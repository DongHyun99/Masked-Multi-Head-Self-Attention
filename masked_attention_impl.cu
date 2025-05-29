#include "masked_attention_kernel.cu"

// Template instantiation and kernel launch wrappers

template<typename T>
void launch_masked_qkv_projection(
    const T* input, const T* weight_q, const T* weight_k, const T* weight_v,
    const T* bias_q, const T* bias_k, const T* bias_v, const bool* mask,
    T* query, T* key, T* value,
    int batch_size, int seq_len, int d_model, int num_heads, int head_dim,
    cudaStream_t stream
) {
    // Calculate optimal grid and block dimensions
    dim3 block(min(d_model, MAX_THREADS_PER_BLOCK));
    dim3 grid(batch_size * seq_len);
    
    size_t shared_mem_size = (TILE_SIZE * d_model + d_model * d_model) * sizeof(T);
    
    // Launch QKV projection kernel
    masked_qkv_projection_kernel<T><<<grid, block, shared_mem_size, stream>>>(
        input, weight_q, weight_k, weight_v, bias_q, bias_k, bias_v, mask,
        query, key, value, batch_size, seq_len, d_model, num_heads, head_dim
    );
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void launch_masked_attention(
    const T* query, const T* key, const T* value, const bool* mask,
    T* output, T* attention_weights,
    int batch_size, int num_heads, int seq_len, int head_dim, float scale,
    cudaStream_t stream
) {
    // Optimize for different sequence lengths
    dim3 block, grid;
    size_t shared_mem_size;
    
    if (seq_len <= 512) {
        // Small sequence optimization
        block = dim3(32, min(seq_len, 32));
        grid = dim3(batch_size * num_heads, (seq_len + block.y - 1) / block.y);
        shared_mem_size = (3 * TILE_SIZE * head_dim + TILE_SIZE * TILE_SIZE) * sizeof(T);
    } else {
        // Large sequence optimization with tiling
        block = dim3(32, 16);
        grid = dim3(batch_size * num_heads, (seq_len + 15) / 16);
        shared_mem_size = (3 * 16 * head_dim + 16 * 16) * sizeof(T);
    }
    
    // Initialize output to zero for masked positions
    CUDA_CHECK(cudaMemsetAsync(output, 0, 
        batch_size * num_heads * seq_len * head_dim * sizeof(T), stream));
    
    // Launch attention kernel
    masked_attention_kernel<T><<<grid, block, shared_mem_size, stream>>>(
        query, key, value, mask, output, attention_weights,
        batch_size, num_heads, seq_len, head_dim, scale
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Specialized kernel for FP16 with proper half precision operations
template<>
__global__ void masked_attention_kernel<half>(
    const half* query, const half* key, const half* value, const bool* mask,
    half* output, half* attention_weights,
    int batch_size, int num_heads, int seq_len, int head_dim, float scale
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    extern __shared__ char shared_mem_half[];
    half* s_query = reinterpret_cast<half*>(shared_mem_half);
    half* s_key = s_query + TILE_SIZE * head_dim;
    half* s_value = s_key + TILE_SIZE * head_dim;
    half* s_scores = s_value + TILE_SIZE * head_dim;
    
    int batch_idx = blockIdx.x / num_heads;
    int head_idx = blockIdx.x % num_heads;
    int query_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    if (query_idx >= seq_len || !mask[batch_idx * seq_len + query_idx]) {
        return;
    }
    
    // Use half precision arithmetic with proper CUDA half operations
    half scale_half = __float2half(scale);
    half max_score = __float2half(-65504.0f); // -inf for half
    half attention_sum = __float2half(0.0f);
    
    // Load query
    if (threadIdx.x < head_dim && query_idx < seq_len) {
        s_query[threadIdx.y * head_dim + threadIdx.x] = 
            query[batch_idx * num_heads * seq_len * head_dim + 
                  head_idx * seq_len * head_dim + query_idx * head_dim + threadIdx.x];
    }
    block.sync();
    
    // Process in tiles with optimized half precision operations
    for (int key_tile = 0; key_tile < (seq_len + TILE_SIZE - 1) / TILE_SIZE; key_tile++) {
        int key_start = key_tile * TILE_SIZE;
        int key_idx = key_start + threadIdx.y;
        
        // Load key tile
        if (threadIdx.x < head_dim && key_idx < seq_len && mask[batch_idx * seq_len + key_idx]) {
            s_key[threadIdx.y * head_dim + threadIdx.x] = 
                key[batch_idx * num_heads * seq_len * head_dim + 
                    head_idx * seq_len * head_dim + key_idx * head_dim + threadIdx.x];
        } else {
            s_key[threadIdx.y * head_dim + threadIdx.x] = __float2half(0.0f);
        }
        block.sync();
        
        // Compute attention scores using vectorized half operations
        for (int k = 0; k < min(TILE_SIZE, seq_len - key_start); k++) {
            int actual_key_idx = key_start + k;
            if (!mask[batch_idx * seq_len + actual_key_idx]) continue;
            
            half score = __float2half(0.0f);
            
            // Vectorized dot product for half precision
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                score = __hadd(score, __hmul(s_query[threadIdx.y * head_dim + d], 
                                           s_key[k * head_dim + d]));
            }
            
            // Warp reduction for half precision
            for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
                score = __hadd(score, warp.shfl_down(score, offset));
            }
            
            if (warp.thread_rank() == 0) {
                score = __hmul(score, scale_half);
                s_scores[k] = score;
                max_score = __hmax(max_score, score);
            }
        }
        block.sync();
        
        // Softmax and value accumulation
        for (int k = 0; k < min(TILE_SIZE, seq_len - key_start); k++) {
            int actual_key_idx = key_start + k;
            if (!mask[batch_idx * seq_len + actual_key_idx]) continue;
            
            half exp_score = hexp(__hsub(s_scores[k], max_score));
            attention_sum = __hadd(attention_sum, exp_score);
            
            // Load and accumulate values
            if (threadIdx.x < head_dim) {
                half val = value[batch_idx * num_heads * seq_len * head_dim + 
                               head_idx * seq_len * head_dim + actual_key_idx * head_dim + threadIdx.x];
                
                // Use atomic add for half precision
                #if __CUDA_ARCH__ >= 700
                atomicAdd(&output[batch_idx * num_heads * seq_len * head_dim + 
                                head_idx * seq_len * head_dim + query_idx * head_dim + threadIdx.x],
                         __hmul(exp_score, val));
                #else
                // Fallback for older architectures
                output[batch_idx * num_heads * seq_len * head_dim + 
                      head_idx * seq_len * head_dim + query_idx * head_dim + threadIdx.x] = 
                      __hadd(output[batch_idx * num_heads * seq_len * head_dim + 
                                   head_idx * seq_len * head_dim + query_idx * head_dim + threadIdx.x],
                            __hmul(exp_score, val));
                #endif
            }
        }
        block.sync();
    }
    
    // Final normalization
    if (warp.thread_rank() == 0 && __hgt(attention_sum, __float2half(0.0f))) {
        for (int d = 0; d < head_dim; d++) {
            int idx = batch_idx * num_heads * seq_len * head_dim + 
                     head_idx * seq_len * head_dim + query_idx * head_dim + d;
            output[idx] = __hdiv(output[idx], attention_sum);
        }
    }
}

// Performance monitoring kernel
__global__ void attention_performance_kernel(
    const bool* mask,
    int* active_tokens,
    float* sparsity_ratio,
    int batch_size,
    int seq_len
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * seq_len) return;
    
    __shared__ int s_count[256];
    int local_tid = threadIdx.x;
    s_count[local_tid] = mask[tid] ? 1 : 0;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            s_count[local_tid] += s_count[local_tid + stride];
        }
        __syncthreads();
    }
    
    if (local_tid == 0) {
        atomicAdd(active_tokens, s_count[0]);
    }
    
    if (tid == 0) {
        *sparsity_ratio = 1.0f - (float)(*active_tokens) / (batch_size * seq_len);
    }
}

// Explicit template instantiations
template void launch_masked_qkv_projection<float>(
    const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const bool*,
    float*, float*, float*, int, int, int, int, int, cudaStream_t
);

template void launch_masked_qkv_projection<half>(
    const half*, const half*, const half*, const half*,
    const half*, const half*, const half*, const bool*,
    half*, half*, half*, int, int, int, int, int, cudaStream_t
);

template void launch_masked_attention<float>(
    const float*, const float*, const float*, const bool*,
    float*, float*, int, int, int, int, float, cudaStream_t
);

template void launch_masked_attention<half>(
    const half*, const half*, const half*, const bool*,
    half*, half*, int, int, int, int, float, cudaStream_t
);

// Utility function for optimal kernel configuration
void get_optimal_config(int batch_size, int seq_len, int num_heads, int head_dim,
                       dim3& grid, dim3& block, size_t& shared_mem) {
    // Heuristic-based configuration
    if (seq_len <= 128) {
        block = dim3(32, 4);
        grid = dim3(batch_size * num_heads, (seq_len + 3) / 4);
    } else if (seq_len <= 512) {
        block = dim3(32, 8);
        grid = dim3(batch_size * num_heads, (seq_len + 7) / 8);
    } else {
        block = dim3(32, 16);
        grid = dim3(batch_size * num_heads, (seq_len + 15) / 16);
    }
    
    shared_mem = (3 * block.y * head_dim + block.y * block.y) * sizeof(float);
    
    // Ensure shared memory doesn't exceed device limits
    shared_mem = min(shared_mem, (size_t)SHARED_MEM_SIZE);
}