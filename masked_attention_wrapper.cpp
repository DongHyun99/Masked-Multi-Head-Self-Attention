#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

// Forward declarations of CUDA kernels
void launch_masked_qkv_projection_float(
    const float* input, const float* weight_q, const float* weight_k, const float* weight_v,
    const float* bias_q, const float* bias_k, const float* bias_v, const bool* mask,
    float* query, float* key, float* value,
    int batch_size, int seq_len, int d_model, int num_heads, int head_dim,
    cudaStream_t stream
);

void launch_masked_attention_float(
    const float* query, const float* key, const float* value, const bool* mask,
    float* output, float* attention_weights,
    int batch_size, int num_heads, int seq_len, int head_dim, float scale,
    cudaStream_t stream
);

// Simplified masked multi-head attention implementation
torch::Tensor masked_attention_forward(
    torch::Tensor input,           // [batch_size, seq_len, d_model]
    torch::Tensor weight_q,        // [d_model, d_model]
    torch::Tensor weight_k,        // [d_model, d_model] 
    torch::Tensor weight_v,        // [d_model, d_model]
    torch::Tensor bias_q,          // [d_model]
    torch::Tensor bias_k,          // [d_model]
    torch::Tensor bias_v,          // [d_model] 
    torch::Tensor mask,            // [batch_size, seq_len]
    int num_heads
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(mask.dtype() == torch::kBool, "Mask must be boolean tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 supported in this simplified version");
    
    const auto batch_size = input.size(0);
    const auto seq_len = input.size(1);
    const auto d_model = input.size(2);
    const auto head_dim = d_model / num_heads;
    
    TORCH_CHECK(d_model % num_heads == 0, 
               "d_model must be divisible by num_heads");
    
    // Create output tensors
    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
        
    auto query = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);
    auto key = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);  
    auto value = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);
    auto output = torch::zeros({batch_size, seq_len, d_model}, options);
    
    // Get CUDA stream
    const auto stream = at::cuda::getCurrentCUDAStream();
    const at::cuda::CUDAGuard device_guard(input.device());
    
    // Scale factor for attention
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    // Launch QKV projection kernel
    launch_masked_qkv_projection_float(
        input.data_ptr<float>(),
        weight_q.data_ptr<float>(),
        weight_k.data_ptr<float>(), 
        weight_v.data_ptr<float>(),
        bias_q.data_ptr<float>(),
        bias_k.data_ptr<float>(),
        bias_v.data_ptr<float>(),
        mask.data_ptr<bool>(),
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        batch_size, seq_len, d_model, num_heads, head_dim,
        stream
    );
    
    // Launch attention computation kernel
    launch_masked_attention_float(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        mask.data_ptr<bool>(),
        output.data_ptr<float>(),
        nullptr, // No attention weights for now
        batch_size, num_heads, seq_len, head_dim, scale,
        stream
    );
    
    // Reshape output to [batch_size, seq_len, d_model]
    output = output.view({batch_size, seq_len, d_model});
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("masked_attention_forward", &masked_attention_forward, 
          "Masked Multi-Head Attention Forward Pass");
}