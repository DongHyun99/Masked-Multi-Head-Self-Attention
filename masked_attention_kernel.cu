#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Constants for optimization
#define WARP_SIZE 32
#define TILE_SIZE 16
#define MAX_THREADS_PER_BLOCK 1024
#define SHARED_MEM_SIZE 48000

// Helper functions for generic operations
template<typename T>
__device__ __forceinline__ T add_op(T a, T b) { return a + b; }

template<typename T>
__device__ __forceinline__ T mul_op(T a, T b) { return a * b; }

template<typename T>
__device__ __forceinline__ T max_op(T a, T b) { return max(a, b); }

template<typename T>
__device__ __forceinline__ T exp_op(T x) { return exp(x); }

template<typename T>
__device__ __forceinline__ T div_op(T a, T b) { return a / b; }

// Specializations for half precision
#ifdef __CUDA_ARCH__ 
#if __CUDA_ARCH__ >= 530
template<>
__device__ __forceinline__ half add_op<half>(half a, half b) { return __hadd(a, b); }

template<>
__device__ __forceinline__ half mul_op<half>(half a, half b) { return __hmul(a, b); }

template<>
__device__ __forceinline__ half max_op<half>(half a, half b) { return __hmax(a, b); }

template<>
__device__ __forceinline__ half exp_op<half>(half x) { return hexp(x); }

template<>
__device__ __forceinline__ half div_op<half>(half a, half b) { return __hdiv(a, b); }
#endif
#endif

// CUDA kernel for masked QKV projection
template<typename T>
__global__ void masked_qkv_projection_kernel(
    const T* input,           // [batch_size, seq_len, d_model]
    const T* weight_q,        // [d_model, d_model]  
    const T* weight_k,        // [d_model, d_model]
    const T* weight_v,        // [d_model, d_model]
    const T* bias_q,          // [d_model]
    const T* bias_k,          // [d_model]
    const T* bias_v,          // [d_model]
    const bool* mask,         // [batch_size, seq_len]
    T* query,                 // [batch_size, num_heads, seq_len, head_dim]
    T* key,                   // [batch_size, num_heads, seq_len, head_dim]
    T* value,                 // [batch_size, num_heads, seq_len, head_dim]
    int batch_size,
    int seq_len,
    int d_model,
    int num_heads,
    int head_dim
) {
    // Use dynamic shared memory
    extern __shared__ char shared_mem[];
    T* s_input = reinterpret_cast<T*>(shared_mem);
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int batch_idx = bid / seq_len;
    int seq_idx = bid % seq_len;
    
    // Early exit if token is masked
    if (!mask[batch_idx * seq_len + seq_idx]) {
        return;
    }
    
    // Load input token to shared memory
    if (tid < d_model) {
        s_input[tid] = input[batch_idx * seq_len * d_model + seq_idx * d_model + tid];
    }
    __syncthreads();
    
    // Compute Q, K, V for each head
    for (int head = 0; head < num_heads; head++) {
        int head_offset = head * head_dim;
        
        // Query computation
        if (tid < head_dim) {
            T q_val = bias_q[head_offset + tid];
            for (int i = 0; i < d_model; i++) {
                q_val = add_op(q_val, mul_op(s_input[i], weight_q[i * d_model + head_offset + tid]));
            }
            query[batch_idx * num_heads * seq_len * head_dim + 
                  head * seq_len * head_dim + seq_idx * head_dim + tid] = q_val;
        }
        
        // Key computation  
        if (tid < head_dim) {
            T k_val = bias_k[head_offset + tid];
            for (int i = 0; i < d_model; i++) {
                k_val = add_op(k_val, mul_op(s_input[i], weight_k[i * d_model + head_offset + tid]));
            }
            key[batch_idx * num_heads * seq_len * head_dim + 
                head * seq_len * head_dim + seq_idx * head_dim + tid] = k_val;
        }
        
        // Value computation
        if (tid < head_dim) {
            T v_val = bias_v[head_offset + tid];
            for (int i = 0; i < d_model; i++) {
                v_val = add_op(v_val, mul_op(s_input[i], weight_v[i * d_model + head_offset + tid]));
            }
            value[batch_idx * num_heads * seq_len * head_dim + 
                  head * seq_len * head_dim + seq_idx * head_dim + tid] = v_val;
        }
    }
}

// Optimized masked attention computation
template<typename T>
__global__ void masked_attention_kernel(
    const T* query,           // [batch_size, num_heads, seq_len, head_dim]
    const T* key,             // [batch_size, num_heads, seq_len, head_dim]  
    const T* value,           // [batch_size, num_heads, seq_len, head_dim]
    const bool* mask,         // [batch_size, seq_len]
    T* output,                // [batch_size, num_heads, seq_len, head_dim]
    T* attention_weights,     // [batch_size, num_heads, seq_len, seq_len] (optional)
    int batch_size,
    int num_heads, 
    int seq_len,
    int head_dim,
    float scale
) {
    // Use cooperative groups for better warp-level coordination
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    
    extern __shared__ char shared_mem[];
    T* s_query = reinterpret_cast<T*>(shared_mem);
    T* s_key = s_query + TILE_SIZE * head_dim;
    T* s_value = s_key + TILE_SIZE * head_dim;
    T* s_scores = s_value + TILE_SIZE * head_dim;
    
    int batch_idx = blockIdx.x / num_heads;
    int head_idx = blockIdx.x % num_heads;
    int query_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    
    // Early exit if query token is masked
    if (query_idx >= seq_len || !mask[batch_idx * seq_len + query_idx]) {
        return;
    }
    
    // Load query to shared memory
    if (tid_x < head_dim && query_idx < seq_len) {
        s_query[tid_y * head_dim + tid_x] = 
            query[batch_idx * num_heads * seq_len * head_dim + 
                  head_idx * seq_len * head_dim + query_idx * head_dim + tid_x];
    }
    block.sync();
    
    T attention_sum = T(0.0f);
    T max_score = T(-1e20f);  // Use large negative value instead of -INFINITY
    
    // Process keys in tiles for memory efficiency
    for (int key_tile = 0; key_tile < (seq_len + TILE_SIZE - 1) / TILE_SIZE; key_tile++) {
        int key_start = key_tile * TILE_SIZE;
        int key_idx = key_start + tid_y;
        
        // Load key tile to shared memory (only unmasked tokens)
        if (tid_x < head_dim && key_idx < seq_len && mask[batch_idx * seq_len + key_idx]) {
            s_key[tid_y * head_dim + tid_x] = 
                key[batch_idx * num_heads * seq_len * head_dim + 
                    head_idx * seq_len * head_dim + key_idx * head_dim + tid_x];
        } else {
            s_key[tid_y * head_dim + tid_x] = T(0.0f);
        }
        block.sync();
        
        // Compute attention scores for this tile
        for (int k = 0; k < min(TILE_SIZE, seq_len - key_start); k++) {
            int actual_key_idx = key_start + k;
            if (!mask[batch_idx * seq_len + actual_key_idx]) continue;
            
            T score = T(0.0f);
            // Dot product with vectorized operations
            for (int d = tid_x; d < head_dim; d += blockDim.x) {
                score = add_op(score, mul_op(s_query[tid_y * head_dim + d], s_key[k * head_dim + d]));
            }
            
            // Warp-level reduction for dot product
            for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
                score = add_op(score, warp.shfl_down(score, offset));
            }
            
            if (warp.thread_rank() == 0) {
                score = mul_op(score, T(scale));
                s_scores[k] = score;
                max_score = max_op(max_score, score);
            }
        }
        block.sync();
        
        // Compute softmax and accumulate weighted values
        for (int k = 0; k < min(TILE_SIZE, seq_len - key_start); k++) {
            int actual_key_idx = key_start + k;
            if (!mask[batch_idx * seq_len + actual_key_idx]) continue;
            
            T exp_score = exp_op(add_op(s_scores[k], mul_op(max_score, T(-1.0f))));
            attention_sum = add_op(attention_sum, exp_score);
            
            // Load value and accumulate
            if (tid_x < head_dim && key_idx < seq_len) {
                s_value[tid_y * head_dim + tid_x] = 
                    value[batch_idx * num_heads * seq_len * head_dim + 
                          head_idx * seq_len * head_dim + actual_key_idx * head_dim + tid_x];
            }
            block.sync();
            
            // Accumulate weighted values
            for (int d = tid_x; d < head_dim; d += blockDim.x) {
                atomicAdd(&output[batch_idx * num_heads * seq_len * head_dim + 
                                head_idx * seq_len * head_dim + query_idx * head_dim + d],
                         mul_op(exp_score, s_value[tid_y * head_dim + d]));
            }
        }
        block.sync();
    }
    
    // Normalize by attention sum
    if (warp.thread_rank() == 0 && attention_sum > T(0.0f)) {
        for (int d = 0; d < head_dim; d++) {
            int idx = batch_idx * num_heads * seq_len * head_dim + 
                     head_idx * seq_len * head_dim + query_idx * head_dim + d;
            output[idx] = div_op(output[idx], attention_sum);
        }
    }
}

// Launch configuration helper
struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem;
    
    LaunchConfig(int batch_size, int seq_len, int num_heads, int head_dim) {
        // Optimize grid/block dimensions based on problem size
        block = dim3(min(head_dim, 32), min(seq_len, 32), 1);
        grid = dim3(batch_size * num_heads, (seq_len + block.y - 1) / block.y, 1);
        shared_mem = (4 * TILE_SIZE * head_dim) * sizeof(float); // Conservative estimate
    }
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)