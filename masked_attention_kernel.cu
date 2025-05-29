#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA kernels for masked multi-head self-attention
template<typename T>
__global__ void masked_qkv_projection_kernel(
    const T* input,           // [B, N, D]
    const T* weight_q,        // [D, D]
    const T* weight_k,        // [D, D]
    const T* weight_v,        // [D, D]
    const T* bias_q,          // [D]
    const T* bias_k,          // [D]
    const T* bias_v,          // [D]
    const bool* mask,         // [B, N]
    T* query,                 // [B, N, D]
    T* key,                   // [B, N, D]
    T* value,                 // [B, N, D]
    int batch_size,
    int seq_len,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * dim;
    
    if (idx >= total_elements) return;
    
    int b = idx / (seq_len * dim);
    int n = (idx % (seq_len * dim)) / dim;
    int d = idx % dim;
    
    // Check if this token is masked
    if (!mask[b * seq_len + n]) {
        query[idx] = T(0);
        key[idx] = T(0);
        value[idx] = T(0);
        return;
    }
    
    // Compute Q, K, V projections
    T q_val = bias_q[d];
    T k_val = bias_k[d];
    T v_val = bias_v[d];
    
    for (int i = 0; i < dim; i++) {
        T inp_val = input[b * seq_len * dim + n * dim + i];
        q_val += inp_val * weight_q[i * dim + d];
        k_val += inp_val * weight_k[i * dim + d];
        v_val += inp_val * weight_v[i * dim + d];
    }
    
    query[idx] = q_val;
    key[idx] = k_val;
    value[idx] = v_val;
}

template<typename T>
__global__ void masked_attention_scores_kernel(
    const T* query,           // [B, H, N, D/H]
    const T* key,             // [B, H, N, D/H]
    const bool* mask,         // [B, N]
    T* scores,                // [B, H, N, N]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z * blockDim.x + threadIdx.x;
    int j = blockIdx.z * blockDim.y + threadIdx.y;
    
    if (b >= batch_size || h >= num_heads || i >= seq_len || j >= seq_len) return;
    
    // Check if either token is masked
    if (!mask[b * seq_len + i] || !mask[b * seq_len + j]) {
        scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j] = T(-1e9);
        return;
    }
    
    // Compute attention score
    T score = T(0);
    for (int d = 0; d < head_dim; d++) {
        T q_val = query[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + d];
        T k_val = key[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + j * head_dim + d];
        score += q_val * k_val;
    }
    
    scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j] = score * T(scale);
}

template<typename T>
__global__ void masked_softmax_kernel(
    T* scores,                // [B, H, N, N]
    const bool* mask,         // [B, N]
    int batch_size,
    int num_heads,
    int seq_len
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int i = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || h >= num_heads || i >= seq_len) return;
    
    // Skip masked tokens
    if (!mask[b * seq_len + i]) {
        for (int j = 0; j < seq_len; j++) {
            scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j] = T(0);
        }
        return;
    }
    
    // Find max for numerical stability
    T max_val = T(-1e9);
    for (int j = 0; j < seq_len; j++) {
        if (mask[b * seq_len + j]) {
            T val = scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j];
            max_val = fmaxf(max_val, val);
        }
    }
    
    // Compute exp and sum
    T sum = T(0);
    for (int j = 0; j < seq_len; j++) {
        if (mask[b * seq_len + j]) {
            T val = scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j];
            val = expf(val - max_val);
            scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j] = val;
            sum += val;
        } else {
            scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j] = T(0);
        }
    }
    
    // Normalize
    for (int j = 0; j < seq_len; j++) {
        if (mask[b * seq_len + j]) {
            scores[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j] /= sum;
        }
    }
}

template<typename T>
__global__ void masked_attention_output_kernel(
    const T* attention_weights, // [B, H, N, N]
    const T* value,             // [B, H, N, D/H]
    const bool* mask,           // [B, N]
    T* output,                  // [B, H, N, D/H]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * seq_len * head_dim;
    
    if (idx >= total_elements) return;
    
    int b = idx / (num_heads * seq_len * head_dim);
    int h = (idx % (num_heads * seq_len * head_dim)) / (seq_len * head_dim);
    int i = (idx % (seq_len * head_dim)) / head_dim;
    int d = idx % head_dim;
    
    // Skip masked tokens
    if (!mask[b * seq_len + i]) {
        output[idx] = T(0);
        return;
    }
    
    // Compute weighted sum
    T result = T(0);
    for (int j = 0; j < seq_len; j++) {
        if (mask[b * seq_len + j]) {
            T weight = attention_weights[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j];
            T val = value[b * num_heads * seq_len * head_dim + h * seq_len * head_dim + j * head_dim + d];
            result += weight * val;
        }
    }
    
    output[idx] = result;
}

template<typename T>
__global__ void masked_final_projection_kernel(
    const T* attention_output, // [B, N, D]
    const T* weight_o,         // [D, D]
    const T* bias_o,           // [D]
    const bool* mask,          // [B, N]
    const T* residual,         // [B, N, D] - original input for residual connection
    T* output,                 // [B, N, D]
    int batch_size,
    int seq_len,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * dim;
    
    if (idx >= total_elements) return;
    
    int b = idx / (seq_len * dim);
    int n = (idx % (seq_len * dim)) / dim;
    int d = idx % dim;
    
    // For masked tokens, preserve original input
    if (!mask[b * seq_len + n]) {
        output[idx] = residual[idx];
        return;
    }
    
    // Compute final projection
    T result = bias_o[d];
    for (int i = 0; i < dim; i++) {
        T att_val = attention_output[b * seq_len * dim + n * dim + i];
        result += att_val * weight_o[i * dim + d];
    }
    
    // Add residual connection
    output[idx] = result + residual[idx];
}

// Main function implementations
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
    
    // Allocate intermediate tensors
    auto query = torch::zeros({batch_size, seq_len, dim}, options);
    auto key = torch::zeros({batch_size, seq_len, dim}, options);
    auto value = torch::zeros({batch_size, seq_len, dim}, options);
    auto scores = torch::zeros({batch_size, num_heads, seq_len, seq_len}, options);
    auto attention_output = torch::zeros({batch_size, seq_len, dim}, options);
    auto output = torch::zeros_like(input);
    
    // Launch kernels
    const int threads = 256;
    const int blocks = (batch_size * seq_len * dim + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "masked_qkv_projection", ([&] {
        masked_qkv_projection_kernel<scalar_t><<<blocks, threads>>>(
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
    query = query.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
    key = key.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
    value = value.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
    
    // Attention scores
    dim3 score_blocks(batch_size, num_heads, (seq_len + 15) / 16);
    dim3 score_threads(16, 16);
    float scale = 1.0f / sqrt(head_dim);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "masked_attention_scores", ([&] {
        masked_attention_scores_kernel<scalar_t><<<score_blocks, score_threads>>>(
            query.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            scores.data_ptr<scalar_t>(),
            batch_size, num_heads, seq_len, head_dim, scale
        );
    }));
    
    // Softmax
    dim3 softmax_blocks(batch_size, num_heads, (seq_len + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "masked_softmax", ([&] {
        masked_softmax_kernel<scalar_t><<<softmax_blocks, threads>>>(
            scores.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            batch_size, num_heads, seq_len
        );
    }));
    
    // Attention output
    auto att_out = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);
    const int att_blocks = (batch_size * num_heads * seq_len * head_dim + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "masked_attention_output", ([&] {
        masked_attention_output_kernel<scalar_t><<<att_blocks, threads>>>(
            scores.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            att_out.data_ptr<scalar_t>(),
            batch_size, num_heads, seq_len, head_dim
        );
    }));
    
    // Reshape back
    att_out = att_out.transpose(1, 2).contiguous().view({batch_size, seq_len, dim});
    
    // Final projection
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "masked_final_projection", ([&] {
        masked_final_projection_kernel<scalar_t><<<blocks, threads>>>(
            att_out.data_ptr<scalar_t>(),
            weight_o.data_ptr<scalar_t>(),
            bias_o.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(),
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, seq_len, dim
        );
    }));
    
    cudaDeviceSynchronize();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("masked_multi_head_attention", &masked_multi_head_attention_cuda, "Masked Multi-Head Attention CUDA");
}