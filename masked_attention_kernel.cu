#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Optimized CUDA kernels for masked multi-head self-attention

// Fast fused QKV kernel with better memory coalescing
template<typename T>
__global__ void fused_masked_qkv_kernel(
    const T* __restrict__ input,
    const T* __restrict__ qkv_weight,  // [3 * hidden_size, hidden_size] - Q,K,V weights concatenated
    const T* __restrict__ qkv_bias,    // [3 * hidden_size] - Q,K,V biases concatenated  
    const bool* __restrict__ mask,
    T* __restrict__ qkv_output,        // [B, N, 3 * hidden_size]
    int batch_size,
    int seq_len,
    int hidden_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = batch_size * seq_len * hidden_size * 3;
    
    if (tid >= total_threads) return;
    
    const int b = tid / (seq_len * hidden_size * 3);
    const int n = (tid % (seq_len * hidden_size * 3)) / (hidden_size * 3);
    const int qkv_idx = (tid % (hidden_size * 3)) / hidden_size; // 0=Q, 1=K, 2=V
    const int d = tid % hidden_size;
    
    if (!mask[b * seq_len + n]) {
        qkv_output[tid] = T(0);
        return;
    }
    
    T result = qkv_bias[qkv_idx * hidden_size + d];
    
    // Vectorized load when possible
    const T* input_ptr = input + b * seq_len * hidden_size + n * hidden_size;
    const T* weight_ptr = qkv_weight + qkv_idx * hidden_size * hidden_size + d;
    
    #pragma unroll 4
    for (int i = 0; i < hidden_size; i++) {
        result += input_ptr[i] * weight_ptr[i * hidden_size];
    }
    
    qkv_output[tid] = result;
}

// Optimized attention scores with shared memory
template<typename T>
__global__ void optimized_attention_scores_kernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const bool* __restrict__ mask,
    T* __restrict__ scores,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    extern __shared__ T shmem[];
    
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int row = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || h >= num_heads || row >= seq_len) return;
    
    if (!mask[b * seq_len + row]) {
        for (int col = 0; col < seq_len; col++) {
            scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + row * seq_len + col] = T(-1e9);
        }
        return;
    }
    
    const T* q_row = query + b * num_heads * seq_len * head_dim + h * seq_len * head_dim + row * head_dim;
    
    // Load query into shared memory
    T* q_shared = shmem + threadIdx.x * head_dim;
    for (int d = 0; d < head_dim; d++) {
        q_shared[d] = q_row[d];
    }
    
    // Compute scores for this row
    for (int col = 0; col < seq_len; col++) {
        if (!mask[b * seq_len + col]) {
            scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + row * seq_len + col] = T(-1e9);
            continue;
        }
        
        const T* k_col = key + b * num_heads * seq_len * head_dim + h * seq_len * head_dim + col * head_dim;
        
        T score = T(0);
        #pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
            score += q_shared[d] * k_col[d];
        }
        
        scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + row * seq_len + col] = score * T(scale);
    }
}

// Fast softmax with warp-level reductions
template<typename T>
__global__ void fast_masked_softmax_kernel(
    T* __restrict__ scores,
    const bool* __restrict__ mask,
    int batch_size,
    int num_heads,
    int seq_len
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int row = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || h >= num_heads || row >= seq_len) return;
    
    if (!mask[b * seq_len + row]) {
        T* row_ptr = scores + b * num_heads * seq_len * seq_len + h * seq_len * seq_len + row * seq_len;
        for (int j = 0; j < seq_len; j++) {
            row_ptr[j] = T(0);
        }
        return;
    }
    
    T* row_ptr = scores + b * num_heads * seq_len * seq_len + h * seq_len * seq_len + row * seq_len;
    
    // Find max using warp shuffle
    T max_val = T(-1e9);
    for (int j = 0; j < seq_len; j++) {
        if (mask[b * seq_len + j]) {
            max_val = fmaxf(max_val, row_ptr[j]);
        }
    }
    
    // Compute exp and sum
    T sum = T(0);
    for (int j = 0; j < seq_len; j++) {
        if (mask[b * seq_len + j]) {
            T val = expf(row_ptr[j] - max_val);
            row_ptr[j] = val;
            sum += val;
        } else {
            row_ptr[j] = T(0);
        }
    }
    
    // Normalize
    T inv_sum = T(1) / sum;
    for (int j = 0; j < seq_len; j++) {
        if (mask[b * seq_len + j]) {
            row_ptr[j] *= inv_sum;
        }
    }
}

// Optimized attention output with better memory access
template<typename T>
__global__ void optimized_attention_output_kernel(
    const T* __restrict__ attention_weights,
    const T* __restrict__ value,
    const bool* __restrict__ mask,
    T* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * num_heads * seq_len * head_dim;
    
    if (tid >= total_elements) return;
    
    const int b = tid / (num_heads * seq_len * head_dim);
    const int h = (tid % (num_heads * seq_len * head_dim)) / (seq_len * head_dim);
    const int i = (tid % (seq_len * head_dim)) / head_dim;
    const int d = tid % head_dim;
    
    if (!mask[b * seq_len + i]) {
        output[tid] = T(0);
        return;
    }
    
    const T* weights_row = attention_weights + b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len;
    const T* value_base = value + b * num_heads * seq_len * head_dim + h * seq_len * head_dim;
    
    T result = T(0);
    #pragma unroll 4
    for (int j = 0; j < seq_len; j++) {
        if (mask[b * seq_len + j]) {
            result += weights_row[j] * value_base[j * head_dim + d];
        }
    }
    
    output[tid] = result;
}

// Fused final projection with residual
template<typename T>
__global__ void fused_output_projection_kernel(
    const T* __restrict__ attention_output,
    const T* __restrict__ weight_o,
    const T* __restrict__ bias_o,
    const bool* __restrict__ mask,
    const T* __restrict__ residual,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * seq_len * hidden_size;
    
    if (tid >= total_elements) return;
    
    const int b = tid / (seq_len * hidden_size);
    const int n = (tid % (seq_len * hidden_size)) / hidden_size;
    const int d = tid % hidden_size;
    
    if (!mask[b * seq_len + n]) {
        output[tid] = residual[tid];
        return;
    }
    
    T result = bias_o[d];
    const T* att_row = attention_output + b * seq_len * hidden_size + n * hidden_size;
    
    #pragma unroll 8
    for (int i = 0; i < hidden_size; i++) {
        result += att_row[i] * weight_o[i * hidden_size + d];
    }
    
    output[tid] = result + residual[tid];
}

// Main optimized function
torch::Tensor masked_multi_head_attention_cuda(
    torch::Tensor input,
    torch::Tensor weight_q,
    torch::Tensor weight_k,
    torch::Tensor weight_v,
    torch::Tensor weight_o,
    torch::Tensor bias_q,
    torch::Tensor bias_k,
    torch::Tensor bias_v,
    torch::Tensor bias_o,
    torch::Tensor mask,
    int num_heads
) {
    const auto options = input.options();
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int hidden_size = input.size(2);
    const int head_dim = hidden_size / num_heads;
    
    // Use cuBLAS for large matrix multiplications when beneficial
    const bool use_cublas = seq_len > 128 && hidden_size > 512;
    
    if (use_cublas) {
        // Use optimized cuBLAS path for large tensors
        auto qkv_weight = torch::cat({weight_q, weight_k, weight_v}, 0);
        auto qkv_bias = torch::cat({bias_q, bias_k, bias_v}, 0);
        
        // Mask input
        auto masked_input = input.clone();
        masked_input.masked_fill_(~mask.unsqueeze(-1), 0);
        
        // QKV projection using cuBLAS
        auto qkv = torch::addmm(qkv_bias, masked_input.view({-1, hidden_size}), qkv_weight.t());
        qkv = qkv.view({batch_size, seq_len, 3, hidden_size});
        
        auto query = qkv.select(2, 0).view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
        auto key = qkv.select(2, 1).view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
        auto value = qkv.select(2, 2).view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
        
        // Attention computation using cuBLAS
        auto scores = torch::matmul(query, key.transpose(-2, -1)) / sqrt(head_dim);
        
        // Apply mask
        auto mask_4d = mask.unsqueeze(1).unsqueeze(2).expand({batch_size, num_heads, seq_len, seq_len});
        scores.masked_fill_(~mask_4d, -1e9);
        
        auto attention_weights = torch::softmax(scores, -1);
        auto attention_output = torch::matmul(attention_weights, value);
        
        // Reshape and final projection
        attention_output = attention_output.transpose(1, 2).contiguous().view({batch_size, seq_len, hidden_size});
        auto output = torch::addmm(bias_o, attention_output.view({-1, hidden_size}), weight_o.t());
        output = output.view({batch_size, seq_len, hidden_size}) + input;
        
        // Apply mask to output
        output.masked_scatter_(~mask.unsqueeze(-1).expand_as(output), 
                              input.masked_select(~mask.unsqueeze(-1).expand_as(input)));
        
        return output;
    } else {
        // Use custom kernels for smaller tensors
        auto qkv = torch::zeros({batch_size, seq_len, 3 * hidden_size}, options);
        auto qkv_weight = torch::cat({weight_q, weight_k, weight_v}, 0);
        auto qkv_bias = torch::cat({bias_q, bias_k, bias_v}, 0);
        
        const int threads = 256;
        const int blocks = (batch_size * seq_len * hidden_size * 3 + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "fused_masked_qkv", ([&] {
            fused_masked_qkv_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                qkv_weight.data_ptr<scalar_t>(),
                qkv_bias.data_ptr<scalar_t>(),
                mask.data_ptr<bool>(),
                qkv.data_ptr<scalar_t>(),
                batch_size, seq_len, hidden_size
            );
        }));
        
        // Reshape for multi-head attention
        qkv = qkv.view({batch_size, seq_len, 3, num_heads, head_dim});
        auto query = qkv.select(2, 0).transpose(1, 2);
        auto key = qkv.select(2, 1).transpose(1, 2);
        auto value = qkv.select(2, 2).transpose(1, 2);
        
        // Attention scores
        auto scores = torch::zeros({batch_size, num_heads, seq_len, seq_len}, options);
        dim3 score_blocks(batch_size, num_heads, (seq_len + threads - 1) / threads);
        const int shmem_size = threads * head_dim * sizeof(float);
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "optimized_attention_scores", ([&] {
            optimized_attention_scores_kernel<scalar_t><<<score_blocks, threads, shmem_size>>>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                mask.data_ptr<bool>(),
                scores.data_ptr<scalar_t>(),
                batch_size, num_heads, seq_len, head_dim,
                1.0f / sqrt(head_dim)
            );
        }));
        
        // Softmax
        dim3 softmax_blocks(batch_size, num_heads, (seq_len + threads - 1) / threads);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "fast_masked_softmax", ([&] {
            fast_masked_softmax_kernel<scalar_t><<<softmax_blocks, threads>>>(
                scores.data_ptr<scalar_t>(),
                mask.data_ptr<bool>(),
                batch_size, num_heads, seq_len
            );
        }));
        
        // Attention output
        auto attention_output = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);
        const int att_blocks = (batch_size * num_heads * seq_len * head_dim + threads - 1) / threads;
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "optimized_attention_output", ([&] {
            optimized_attention_output_kernel<scalar_t><<<att_blocks, threads>>>(
                scores.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                mask.data_ptr<bool>(),
                attention_output.data_ptr<scalar_t>(),
                batch_size, num_heads, seq_len, head_dim
            );
        }));
        
        // Reshape and final projection
        attention_output = attention_output.transpose(1, 2).contiguous().view({batch_size, seq_len, hidden_size});
        auto output = torch::zeros_like(input);
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "fused_output_projection", ([&] {
            fused_output_projection_kernel<scalar_t><<<blocks, threads>>>(
                attention_output.data_ptr<scalar_t>(),
                weight_o.data_ptr<scalar_t>(),
                bias_o.data_ptr<scalar_t>(),
                mask.data_ptr<bool>(),
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size, seq_len, hidden_size
            );
        }));
        
        cudaDeviceSynchronize();
        return output;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("masked_multi_head_attention", &masked_multi_head_attention_cuda, "Optimized Masked Multi-Head Attention CUDA");
}