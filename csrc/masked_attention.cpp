#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <stdexcept>
#include "masked_attention_kernel.cuh"

// Global cuBLAS handle
static cublasHandle_t cublas_handle = nullptr;

// Initialize cuBLAS handle
void init_cublas() {
    if (cublas_handle == nullptr) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
    }
}

// Cleanup cuBLAS handle
void cleanup_cublas() {
    if (cublas_handle != nullptr) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
}

namespace masked_attention {

// Create attention mask from token mask
torch::Tensor create_attention_mask(
    const torch::Tensor& mask,
    int num_heads
) {
    TORCH_CHECK(mask.dim() == 2, "Mask must be 2D [batch_size, seq_len]");
    TORCH_CHECK(mask.dtype() == torch::kBool, "Mask must be boolean");
    
    const int batch_size = mask.size(0);
    const int seq_len = mask.size(1);
    
    // Create attention mask [batch_size, seq_len, seq_len]
    auto attention_mask = torch::zeros({batch_size, seq_len, seq_len}, 
                                     torch::TensorOptions().dtype(torch::kBool).device(mask.device()));
    
    // Launch kernel
    dim3 grid(batch_size, (seq_len + 15) / 16, (seq_len + 15) / 16);
    dim3 block(16, 16);
    
    create_attention_mask_kernel<<<grid, block>>>(
        mask.data_ptr<bool>(),
        attention_mask.data_ptr<bool>(),
        batch_size,
        seq_len
    );
    
    CUDA_CHECK(cudaGetLastError());
    return attention_mask;
}

// Optimized QKV projection with cuBLAS
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> masked_qkv_projection(
    const torch::Tensor& input,
    const torch::Tensor& mask,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& qkv_bias,
    int num_heads
) {
    TORCH_CHECK(input.dim() == 3, "Input must be 3D [batch_size, seq_len, d_model]");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA device");
    
    init_cublas();
    
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int d_model = input.size(2);
    const int head_dim = d_model / num_heads;
    
    // Create output tensors
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto q = torch::zeros({batch_size, seq_len, d_model}, options);
    auto k = torch::zeros({batch_size, seq_len, d_model}, options);
    auto v = torch::zeros({batch_size, seq_len, d_model}, options);
    
    // Use cuBLAS for matrix multiplication when possible, fallback to custom kernel
    if (input.dtype() == torch::kFloat32) {
        // Reshape for batch matrix multiply
        auto input_2d = input.view({batch_size * seq_len, d_model});
        auto qkv_out = torch::addmm(qkv_bias, input_2d, qkv_weight.t());
        qkv_out = qkv_out.view({batch_size, seq_len, 3, d_model});
        
        // Split Q, K, V
        q = qkv_out.select(2, 0).clone();
        k = qkv_out.select(2, 1).clone();
        v = qkv_out.select(2, 2).clone();
        
        // Apply masking
        auto expanded_mask = mask.unsqueeze(2).expand({batch_size, seq_len, d_model});
        q.masked_fill_(~expanded_mask, 0.0);
        k.masked_fill_(~expanded_mask, 0.0);
        v.masked_fill_(~expanded_mask, 0.0);
    } else {
        // Use custom kernel for half precision
        dim3 grid(batch_size, seq_len);
        dim3 block(min(d_model, MAX_THREADS_PER_BLOCK));
        
        size_t shared_mem_size = 3 * d_model * sizeof(float);
        
        if (input.dtype() == torch::kFloat16) {
            masked_qkv_kernel<at::Half><<<grid, block, shared_mem_size>>>(
                reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
                mask.data_ptr<bool>(),
                reinterpret_cast<const __half*>(qkv_weight.data_ptr<at::Half>()),
                reinterpret_cast<const __half*>(qkv_bias.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(q.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(k.data_ptr<at::Half>()),
                reinterpret_cast<__half*>(v.data_ptr<at::Half>()),
                batch_size, seq_len, d_model, head_dim
            );
        }
        
        CUDA_CHECK(cudaGetLastError());
    }
    
    return std::make_tuple(q, k, v);
}

// Masked attention computation
torch::Tensor masked_attention_scores(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& attention_mask,
    float scale
) {
    TORCH_CHECK(q.dim() == 4, "Q must be 4D [batch_size, num_heads, seq_len, head_dim]");
    TORCH_CHECK(k.dim() == 4, "K must be 4D [batch_size, num_heads, seq_len, head_dim]");
    
    const int batch_size = q.size(0);
    const int num_heads = q.size(1);
    const int seq_len = q.size(2);
    const int head_dim = q.size(3);
    
    // Use PyTorch's optimized attention when possible
    auto scores = torch::matmul(q, k.transpose(-2, -1)) * scale;
    
    // Apply attention mask
    if (attention_mask.defined()) {
        auto mask_expanded = attention_mask.unsqueeze(1).expand({batch_size, num_heads, seq_len, seq_len});
        scores.masked_fill_(~mask_expanded, -std::numeric_limits<float>::infinity());
    }
    
    return torch::softmax(scores, -1);
}

// Masked attention output computation
torch::Tensor masked_attention_output(
    const torch::Tensor& attention_probs,
    const torch::Tensor& v,
    const torch::Tensor& mask
) {
    TORCH_CHECK(attention_probs.dim() == 4, "Attention probs must be 4D");
    TORCH_CHECK(v.dim() == 4, "V must be 4D");
    
    auto output = torch::matmul(attention_probs, v);
    
    // Apply output masking
    if (mask.defined()) {
        const int batch_size = output.size(0);
        const int num_heads = output.size(1);
        const int seq_len = output.size(2);
        const int head_dim = output.size(3);
        
        auto expanded_mask = mask.unsqueeze(1).unsqueeze(3).expand({batch_size, num_heads, seq_len, head_dim});
        output.masked_fill_(~expanded_mask, 0.0);
    }
    
    return output;
}

// Masked output projection
torch::Tensor masked_output_projection(
    const torch::Tensor& attention_output,
    const torch::Tensor& mask,
    const torch::Tensor& proj_weight,
    const torch::Tensor& proj_bias
) {
    TORCH_CHECK(attention_output.dim() == 3, "Attention output must be 3D");
    
    const int batch_size = attention_output.size(0);
    const int seq_len = attention_output.size(1);
    const int d_model = attention_output.size(2);
    
    // Use cuBLAS for matrix multiplication
    auto output_2d = attention_output.view({batch_size * seq_len, d_model});
    auto proj_out = torch::addmm(proj_bias, output_2d, proj_weight.t());
    auto output = proj_out.view({batch_size, seq_len, d_model});
    
    // Apply masking
    auto expanded_mask = mask.unsqueeze(2).expand({batch_size, seq_len, d_model});
    output.masked_fill_(~expanded_mask, 0.0);
    
    return output;
}

// Main forward function
torch::Tensor masked_multihead_attention_forward(
    const torch::Tensor& input,
    const torch::Tensor& mask,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& qkv_bias,
    const torch::Tensor& proj_weight,
    const torch::Tensor& proj_bias,
    int num_heads,
    float dropout_p,
    bool is_training
) {
    TORCH_CHECK(input.device().is_cuda(), "All tensors must be on CUDA device");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D [batch_size, seq_len, d_model]");
    
    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int d_model = input.size(2);
    const int head_dim = d_model / num_heads;
    
    TORCH_CHECK(d_model % num_heads == 0, "d_model must be divisible by num_heads");
    
    // 1. QKV projection
    auto [q, k, v] = masked_qkv_projection(input, mask, qkv_weight, qkv_bias, num_heads);
    
    // 2. Reshape for multi-head attention
    q = q.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
    k = k.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
    v = v.view({batch_size, seq_len, num_heads, head_dim}).transpose(1, 2);
    
    // 3. Create attention mask
    auto attention_mask = create_attention_mask(mask, num_heads);
    
    // 4. Compute attention scores
    float scale = 1.0f / sqrt(static_cast<float>(head_dim));
    auto attention_probs = masked_attention_scores(q, k, attention_mask, scale);
    
    // 5. Apply dropout if training
    if (is_training && dropout_p > 0.0) {
        attention_probs = torch::dropout(attention_probs, dropout_p, true);
    }
    
    // 6. Compute attention output
    auto attention_output = masked_attention_output(attention_probs, v, mask);
    
    // 7. Reshape and project output
    attention_output = attention_output.transpose(1, 2).contiguous().view({batch_size, seq_len, d_model});
    auto final_output = masked_output_projection(attention_output, mask, proj_weight, proj_bias);
    
    return final_output;
}

} // namespace masked_attention

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("masked_multihead_attention_forward", &masked_attention::masked_multihead_attention_forward,
          "Masked multi-head attention forward pass");
    m.def("create_attention_mask", &masked_attention::create_attention_mask,
          "Create attention mask from token mask");
    m.def("init_cublas", &init_cublas, "Initialize cuBLAS handle");
    m.def("cleanup_cublas", &cleanup_cublas, "Cleanup cuBLAS handle");
}