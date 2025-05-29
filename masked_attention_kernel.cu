#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDABlas.h>

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 1024

// Optimized CUDA kernels for masked multi-head self-attention
template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

template<typename T>
__global__ void fused_masked_qkv_kernel(
    const T* __restrict__ input,     // [B, N, D]
    const T* __restrict__ weight_q,  // [D, D]
    const T* __restrict__ weight_k,  // [D, D] 
    const T* __restrict__ weight_v,  // [D, D]
    const T* __restrict__ bias_q,    // [D]
    const T* __restrict__ bias_k,    // [D]
    const T* __restrict__ bias_v,    // [D]
    const bool* __restrict__ mask,   // [B, N]
    T* __restrict__ query,           // [B, N, D]
    T* __restrict__ key,             // [B, N, D]
    T* __restrict__ value,           // [B, N, D]
    int batch_size,
    int seq_len,
    int dim
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int total_tokens = batch_size * seq_len;
    
    // Each block processes multiple tokens
    for (int token_idx = bid; token_idx < total_tokens; token_idx += gridDim.x) {
        int b = token_idx / seq_len;
        int n = token_idx % seq_len;
        
        // Skip if masked
        if (!mask[token_idx]) {
            if (tid < dim) {
                int out_idx = token_idx * dim + tid;
                query[out_idx] = T(0);
                key[out_idx] = T(0);
                value[out_idx] = T(0);
            }
            continue;
        }
        
        // Process dimensions in parallel
        for (int d = tid; d < dim; d += blockDim.x) {
            int input_offset = token_idx * dim;
            int out_idx = token_idx * dim + d;
            
            T q_sum = bias_q[d];
            T k_sum = bias_k[d];
            T v_sum = bias_v[d];
            
            // Vectorized dot product
            for (int i = 0; i < dim; i++) {
                T inp_val = input[input_offset + i];
                q_sum += inp_val * weight_q[i * dim + d];
                k_sum += inp_val * weight_k[i * dim + d];
                v_sum += inp_val * weight_v[i * dim + d];
            }
            
            query[out_idx] = q_sum;
            key[out_idx] = k_sum;
            value[out_idx] = v_sum;
        }
    }
}

template<typename T>
__global__ void fused_attention_scores_softmax_kernel(
    const T* __restrict__ query,     // [B, H, N, D/H]
    const T* __restrict__ key,       // [B, H, N, D/H]
    const bool* __restrict__ mask,   // [B, N]
    T* __restrict__ scores,          // [B, H, N, N]
    T* __restrict__ softmax_scores,  // [B, H, N, N]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    const int tid = threadIdx.x;
    const int bid_x = blockIdx.x; // batch
    const int bid_y = blockIdx.y; // head
    const int bid_z = blockIdx.z; // query token
    
    if (bid_x >= batch_size || bid_y >= num_heads || bid_z >= seq_len) return;
    
    // Skip masked query tokens
    if (!mask[bid_x * seq_len + bid_z]) {
        for (int j = tid; j < seq_len; j += blockDim.x) {
            int score_idx = bid_x * num_heads * seq_len * seq_len + 
                           bid_y * seq_len * seq_len + bid_z * seq_len + j;
            scores[score_idx] = T(0);
            softmax_scores[score_idx] = T(0);
        }
        return;
    }
    
    __shared__ T s_max;
    __shared__ T s_sum;
    
    // Initialize shared memory
    if (tid == 0) {
        s_max = T(-1e9f);
        s_sum = T(0);
    }
    __syncthreads();
    
    // Compute attention scores for this query
    T thread_max = T(-1e9f);
    for (int j = tid; j < seq_len; j += blockDim.x) {
        if (!mask[bid_x * seq_len + j]) {
            int score_idx = bid_x * num_heads * seq_len * seq_len + 
                           bid_y * seq_len * seq_len + bid_z * seq_len + j;
            scores[score_idx] = T(-1e9f);
            continue;
        }
        
        // Compute dot product
        T score = T(0);
        int q_offset = bid_x * num_heads * seq_len * head_dim + 
                       bid_y * seq_len * head_dim + bid_z * head_dim;
        int k_offset = bid_x * num_heads * seq_len * head_dim + 
                       bid_y * seq_len * head_dim + j * head_dim;
        
        for (int d = 0; d < head_dim; d++) {
            score += query[q_offset + d] * key[k_offset + d];
        }
        score *= T(scale);
        
        int score_idx = bid_x * num_heads * seq_len * seq_len + 
                       bid_y * seq_len * seq_len + bid_z * seq_len + j;
        scores[score_idx] = score;
        thread_max = fmaxf(thread_max, score);
    }
    
    // Reduce max across threads
    T warp_max = warpReduceMax(thread_max);
    if ((tid % WARP_SIZE) == 0) {
        atomicMaxFloat(&s_max, warp_max);
    }
    __syncthreads();
    
    // Compute softmax
    T thread_sum = T(0);
    for (int j = tid; j < seq_len; j += blockDim.x) {
        int score_idx = bid_x * num_heads * seq_len * seq_len + 
                       bid_y * seq_len * seq_len + bid_z * seq_len + j;
        
        if (mask[bid_x * seq_len + j]) {
            T exp_score = expf(scores[score_idx] - s_max);
            softmax_scores[score_idx] = exp_score;
            thread_sum += exp_score;
        } else {
            softmax_scores[score_idx] = T(0);
        }
    }
    
    // Reduce sum across threads
    T warp_sum = warpReduceSum(thread_sum);
    if ((tid % WARP_SIZE) == 0) {
        atomicAdd(&s_sum, warp_sum);
    }
    __syncthreads();
    
    // Normalize
    for (int j = tid; j < seq_len; j += blockDim.x) {
        int score_idx = bid_x * num_heads * seq_len * seq_len + 
                       bid_y * seq_len * seq_len + bid_z * seq_len + j;
        if (mask[bid_x * seq_len + j]) {
            softmax_scores[score_idx] /= s_sum;
        }
    }
}

template<typename T>
__global__ void fused_attention_output_kernel(
    const T* __restrict__ attention_weights, // [B, H, N, N]
    const T* __restrict__ value,             // [B, H, N, D/H]
    const bool* __restrict__ mask,           // [B, N]
    T* __restrict__ output,                  // [B, H, N, D/H]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int bid_x = blockIdx.x; // batch
    const int bid_y = blockIdx.y; // head  
    const int bid_z = blockIdx.z; // query token
    
    if (bid_x >= batch_size || bid_y >= num_heads || bid_z >= seq_len) return;
    
    // Skip masked tokens
    if (!mask[bid_x * seq_len + bid_z]) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            int out_idx = bid_x * num_heads * seq_len * head_dim + 
                         bid_y * seq_len * head_dim + bid_z * head_dim + d;
            output[out_idx] = T(0);
        }
        return;
    }
    
    // Compute attention output
    for (int d = tid; d < head_dim; d += blockDim.x) {
        T result = T(0);
        
        for (int j = 0; j < seq_len; j++) {
            if (mask[bid_x * seq_len + j]) {
                int weight_idx = bid_x * num_heads * seq_len * seq_len + 
                               bid_y * seq_len * seq_len + bid_z * seq_len + j;
                int value_idx = bid_x * num_heads * seq_len * head_dim + 
                              bid_y * seq_len * head_dim + j * head_dim + d;
                
                result += attention_weights[weight_idx] * value[value_idx];
            }
        }
        
        int out_idx = bid_x * num_heads * seq_len * head_dim + 
                     bid_y * seq_len * head_dim + bid_z * head_dim + d;
        output[out_idx] = result;
    }
}

// Atomic max for float
__device__ __forceinline__ void atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
}

// Use cuBLAS for matrix operations
void cublas_masked_gemm(
    cublasHandle_t handle,
    const float* A, const float* B, float* C,
    int m, int n, int k,
    float alpha, float beta
) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                B, n,
                A, k,
                &beta,
                C, n);
}

torch::Tensor masked_multi_head_attention_cuda(
    torch::Tensor input,           // [B, N, D]
    torch::Tensor weight_q,        // [D, D]
    torch::Tensor weight_k,        // [D, D]
    torch::Tensor weight_v,        // [D, D]
    torch::Tensor weight_o,        // [D, D]
    torch::Tensor bias_q,          // [D]
    torch::Tensor bias_k,          // [D]
    torch::Tensor bias_v,          // [D]
    torch::Tensor bias_o,          // [D]
    torch::Tensor mask,            // [B, N]
    int num_heads
) {
    auto options = input.options();
    auto device = input.device();
    
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int dim = input.size(2);
    int head_dim = dim / num_heads;
    
    // Get cuBLAS handle
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    
    // Allocate intermediate tensors
    auto query = torch::zeros({batch_size, seq_len, dim}, options);
    auto key = torch::zeros({batch_size, seq_len, dim}, options);
    auto value = torch::zeros({batch_size, seq_len, dim}, options);
    
    // Step 1: Compute Q, K, V using optimized kernel
    const int qkv_threads = 256;
    const int qkv_blocks = std::min(65535, (batch_size * seq_len + qkv_threads - 1) / qkv_threads);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "fused_masked_qkv", ([&] {
        fused_masked_qkv_kernel<scalar_t><<<qkv_blocks, qkv_threads>>>(
            input.data_ptr<scalar_t>(),
            weight_q.data_ptr<scalar_t>(),
            weight_k.data_ptr<scalar_t>(),
            weight_v.data_ptr<scalar_t>(),
            bias_q.data_ptr<scalar_t>(),
            bias_k.data_ptr<scalar_t>(),
            bias_v.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            batch_size, seq_len, dim
        );
    }));
    
    // Reshape for multi-head attention
    query = query.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2).contiguous();
    key = key.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2).contiguous();
    value = value.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2).contiguous();
    
    // Step 2: Compute attention scores and softmax
    auto scores = torch::zeros({batch_size, num_heads, seq_len, seq_len}, options);
    auto softmax_scores = torch::zeros({batch_size, num_heads, seq_len, seq_len}, options);
    
    dim3 att_blocks(batch_size, num_heads, seq_len);
    const int att_threads = std::min(256, seq_len);
    float scale = 1.0f / sqrt(head_dim);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "fused_attention_scores_softmax", ([&] {
        fused_attention_scores_softmax_kernel<scalar_t><<<att_blocks, att_threads>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            scores.data_ptr<scalar_t>(),
            softmax_scores.data_ptr<scalar_t>(),
            batch_size, num_heads, seq_len, head_dim, scale
        );
    }));
    
    // Step 3: Compute attention output
    auto attention_output = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "fused_attention_output", ([&] {
        fused_attention_output_kernel<scalar_t><<<att_blocks, att_threads>>>(
            softmax_scores.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            attention_output.data_ptr<scalar_t>(),
            batch_size, num_heads, seq_len, head_dim
        );
    }));
    
    // Reshape back and apply output projection using cuBLAS
    attention_output = attention_output.transpose(1, 2).contiguous().view({batch_size, seq_len, dim});
    
    // Final projection using optimized GEMM
    auto output = torch::zeros_like(input);
    
    if (input.dtype() == torch::kFloat32) {
        // Use cuBLAS for better performance
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    dim, batch_size * seq_len, dim,
                    &alpha,
                    weight_o.data_ptr<float>(), dim,
                    attention_output.data_ptr<float>(), dim,
                    &beta,
                    output.data_ptr<float>(), dim);
    } else {
        // Fallback to torch operations for other dtypes
        output = torch::mm(attention_output.view({-1, dim}), weight_o.t()).view({batch_size, seq_len, dim});
    }
    
    // Add bias and residual connection
    output = output + bias_o.unsqueeze(0).unsqueeze(0);
    output = output + input;
    
    // Apply mask to output (preserve original values for masked tokens)
    auto mask_expanded = mask.unsqueeze(-1).expand_as(output);
    output = torch::where(mask_expanded, output, input);
    
    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("masked_multi_head_attention", &masked_multi_head_attention_cuda, "Masked Multi-Head Attention CUDA");
}