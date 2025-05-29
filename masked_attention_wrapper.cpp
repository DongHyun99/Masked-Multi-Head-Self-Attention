#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

// Forward declarations of CUDA kernels
template<typename T>
void launch_masked_qkv_projection(
    const T* input, const T* weight_q, const T* weight_k, const T* weight_v,
    const T* bias_q, const T* bias_k, const T* bias_v, const bool* mask,
    T* query, T* key, T* value,
    int batch_size, int seq_len, int d_model, int num_heads, int head_dim,
    cudaStream_t stream
);

template<typename T>
void launch_masked_attention(
    const T* query, const T* key, const T* value, const bool* mask,
    T* output, T* attention_weights,
    int batch_size, int num_heads, int seq_len, int head_dim, float scale,
    cudaStream_t stream
);

// Optimized masked multi-head attention implementation
class MaskedMultiHeadAttention {
public:
    static torch::Tensor forward(
        torch::Tensor input,           // [batch_size, seq_len, d_model]
        torch::Tensor weight_q,        // [d_model, d_model]
        torch::Tensor weight_k,        // [d_model, d_model] 
        torch::Tensor weight_v,        // [d_model, d_model]
        torch::Tensor bias_q,          // [d_model]
        torch::Tensor bias_k,          // [d_model]
        torch::Tensor bias_v,          // [d_model] 
        torch::Tensor mask,            // [batch_size, seq_len]
        int num_heads,
        bool return_attention_weights = false
    ) {
        TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
        TORCH_CHECK(mask.dtype() == torch::kBool, "Mask must be boolean tensor");
        
        const auto batch_size = input.size(0);
        const auto seq_len = input.size(1);
        const auto d_model = input.size(2);
        const auto head_dim = d_model / num_heads;
        
        TORCH_CHECK(d_model % num_heads == 0, 
                   "d_model must be divisible by num_heads");
        
        // Create output tensors
        auto options = torch::TensorOptions()
            .dtype(input.dtype())
            .device(input.device())
            .requires_grad(input.requires_grad());
            
        auto query = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);
        auto key = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);  
        auto value = torch::zeros({batch_size, num_heads, seq_len, head_dim}, options);
        auto output = torch::zeros({batch_size, seq_len, d_model}, options);
        
        torch::Tensor attention_weights;
        if (return_attention_weights) {
            attention_weights = torch::zeros({batch_size, num_heads, seq_len, seq_len}, options);
        }
        
        // Get CUDA stream
        const auto stream = at::cuda::getCurrentCUDAStream();
        const at::cuda::CUDAGuard device_guard(input.device());
        
        // Scale factor for attention
        const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        
        // Launch QKV projection kernel
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "masked_qkv_projection", [&] {
            launch_masked_qkv_projection<scalar_t>(
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
                batch_size, seq_len, d_model, num_heads, head_dim,
                stream
            );
        });
        
        // Launch attention computation kernel
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "masked_attention", [&] {
            launch_masked_attention<scalar_t>(
                query.data_ptr<scalar_t>(),
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                mask.data_ptr<bool>(),
                output.data_ptr<scalar_t>(),
                return_attention_weights ? attention_weights.data_ptr<scalar_t>() : nullptr,
                batch_size, num_heads, seq_len, head_dim, scale,
                stream
            );
        });
        
        // Reshape output to [batch_size, seq_len, d_model]
        output = output.view({batch_size, seq_len, d_model});
        
        if (return_attention_weights) {
            return std::make_tuple(output, attention_weights);
        }
        return output;
    }
    
    static std::vector<torch::Tensor> backward(
        torch::Tensor grad_output,
        torch::Tensor input,
        torch::Tensor weight_q,
        torch::Tensor weight_k, 
        torch::Tensor weight_v,
        torch::Tensor mask,
        torch::Tensor attention_weights,
        int num_heads
    ) {
        // Implement backward pass for gradient computation
        // This would include gradients w.r.t input, weights, and biases
        
        const auto batch_size = input.size(0);
        const auto seq_len = input.size(1); 
        const auto d_model = input.size(2);
        
        auto grad_input = torch::zeros_like(input);
        auto grad_weight_q = torch::zeros_like(weight_q);
        auto grad_weight_k = torch::zeros_like(weight_k);
        auto grad_weight_v = torch::zeros_like(weight_v);
        auto grad_bias_q = torch::zeros({d_model}, input.options());
        auto grad_bias_k = torch::zeros({d_model}, input.options());
        auto grad_bias_v = torch::zeros({d_model}, input.options());
        
        // Implement backward kernel launches here...
        
        return {grad_input, grad_weight_q, grad_weight_k, grad_weight_v, 
                grad_bias_q, grad_bias_k, grad_bias_v};
    }
};

// Python binding functions
torch::Tensor masked_attention_forward(
    torch::Tensor input,
    torch::Tensor weight_q,
    torch::Tensor weight_k,
    torch::Tensor weight_v, 
    torch::Tensor bias_q,
    torch::Tensor bias_k,
    torch::Tensor bias_v,
    torch::Tensor mask,
    int num_heads,
    bool return_attention_weights
) {
    return MaskedMultiHeadAttention::forward(
        input, weight_q, weight_k, weight_v, bias_q, bias_k, bias_v, 
        mask, num_heads, return_attention_weights
    );
}

std::vector<torch::Tensor> masked_attention_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weight_q,
    torch::Tensor weight_k,
    torch::Tensor weight_v,
    torch::Tensor mask,
    torch::Tensor attention_weights,
    int num_heads
) {
    return MaskedMultiHeadAttention::backward(
        grad_output, input, weight_q, weight_k, weight_v, 
        mask, attention_weights, num_heads
    );
}

// Memory usage estimation
size_t estimate_memory_usage(int batch_size, int seq_len, int d_model, int num_heads) {
    const size_t element_size = sizeof(float); // Assuming float32
    const int head_dim = d_model / num_heads;
    
    size_t qkv_memory = 3 * batch_size * num_heads * seq_len * head_dim * element_size;
    size_t attention_memory = batch_size * num_heads * seq_len * seq_len * element_size; 
    size_t output_memory = batch_size * seq_len * d_model * element_size;
    size_t shared_memory = 48000 * batch_size * num_heads; // Approximate shared memory per block
    
    return qkv_memory + attention_memory + output_memory + shared_memory;
}

// Performance optimization suggestions
std::string get_optimization_suggestions(int batch_size, int seq_len, int d_model, int num_heads) {
    std::stringstream ss;
    ss << "Optimization Suggestions:\n";
    
    if (seq_len > 2048) {
        ss << "- Consider using gradient checkpointing for long sequences\n";
        ss << "- Use mixed precision training (FP16) to reduce memory usage\n";
    }
    
    if (batch_size * num_heads > 32) {
        ss << "- Consider using multiple GPU streams for parallel processing\n";
    }
    
    if (d_model % 128 != 0) {
        ss << "- Pad d_model to multiple of 128 for better memory alignment\n";
    }
    
    return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("masked_attention_forward", &masked_attention_forward, 
          "Masked Multi-Head Attention Forward Pass");
    m.def("masked_attention_backward", &masked_attention_backward,
          "Masked Multi-Head Attention Backward Pass");  
    m.def("estimate_memory_usage", &estimate_memory_usage,
          "Estimate memory usage for given parameters");
    m.def("get_optimization_suggestions", &get_optimization_suggestions,
          "Get optimization suggestions");
}