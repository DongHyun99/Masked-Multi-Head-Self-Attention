#include <torch/extension.h>

// Forward declarations of CUDA functions
torch::Tensor masked_qkv_projection_cuda(
    torch::Tensor input,
    torch::Tensor weight_q,
    torch::Tensor weight_k,
    torch::Tensor weight_v,
    torch::Tensor bias_q,
    torch::Tensor bias_k,
    torch::Tensor bias_v,
    torch::Tensor mask
);

torch::Tensor masked_attention_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask
);

torch::Tensor combine_heads_cuda(
    torch::Tensor multi_head_output,
    torch::Tensor mask
);

// CPU fallback implementations (for debugging)
torch::Tensor masked_qkv_projection_cpu(
    torch::Tensor input,
    torch::Tensor weight_q,
    torch::Tensor weight_k,
    torch::Tensor weight_v,
    torch::Tensor bias_q,
    torch::Tensor bias_k,
    torch::Tensor bias_v,
    torch::Tensor mask
) {
    // Simple CPU implementation for comparison
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto hidden_dim = input.size(2);
    
    auto query = torch::mm(input.view({-1, hidden_dim}), weight_q.t()).view({batch_size, seq_len, hidden_dim}) + bias_q;
    auto key = torch::mm(input.view({-1, hidden_dim}), weight_k.t()).view({batch_size, seq_len, hidden_dim}) + bias_k;
    auto value = torch::mm(input.view({-1, hidden_dim}), weight_v.t()).view({batch_size, seq_len, hidden_dim}) + bias_v;
    
    // Apply mask
    auto mask_expanded = mask.unsqueeze(-1).expand_as(query);
    query = query * mask_expanded.to(query.dtype());
    key = key * mask_expanded.to(key.dtype());
    value = value * mask_expanded.to(value.dtype());
    
    return torch::stack({query, key, value}, 0);
}

// Wrapper functions that choose between CUDA and CPU
torch::Tensor masked_qkv_projection(
    torch::Tensor input,
    torch::Tensor weight_q,
    torch::Tensor weight_k,
    torch::Tensor weight_v,
    torch::Tensor bias_q,
    torch::Tensor bias_k,
    torch::Tensor bias_v,
    torch::Tensor mask
) {
    if (input.is_cuda()) {
        return masked_qkv_projection_cuda(input, weight_q, weight_k, weight_v, bias_q, bias_k, bias_v, mask);
    } else {
        return masked_qkv_projection_cpu(input, weight_q, weight_k, weight_v, bias_q, bias_k, bias_v, mask);
    }
}

torch::Tensor masked_attention(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask
) {
    if (query.is_cuda()) {
        return masked_attention_cuda(query, key, value, mask);
    } else {
        // CPU fallback - basic implementation
        auto batch_size = query.size(0);
        auto num_heads = query.size(1);
        auto seq_len = query.size(2);
        auto head_dim = query.size(3);
        
        auto scores = torch::matmul(query, key.transpose(-2, -1)) / sqrt(head_dim);
        
        // Apply mask
        auto mask_expanded = mask.unsqueeze(1).unsqueeze(1).expand({batch_size, num_heads, seq_len, seq_len});
        scores = scores.masked_fill(~mask_expanded, -std::numeric_limits<float>::infinity());
        
        auto attention_weights = torch::softmax(scores, -1);
        attention_weights = attention_weights.masked_fill(torch::isnan(attention_weights), 0.0);
        
        return torch::matmul(attention_weights, value);
    }
}

torch::Tensor combine_heads(
    torch::Tensor multi_head_output,
    torch::Tensor mask
) {
    if (multi_head_output.is_cuda()) {
        return combine_heads_cuda(multi_head_output, mask);
    } else {
        // CPU fallback
        auto batch_size = multi_head_output.size(0);
        auto num_heads = multi_head_output.size(1);
        auto seq_len = multi_head_output.size(2);
        auto head_dim = multi_head_output.size(3);
        
        auto combined = multi_head_output.transpose(1, 2).contiguous().view({batch_size, seq_len, num_heads * head_dim});
        
        // Apply mask
        auto mask_expanded = mask.unsqueeze(-1).expand_as(combined);
        return combined * mask_expanded.to(combined.dtype());
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("masked_qkv_projection", &masked_qkv_projection, "Masked QKV Projection");
    m.def("masked_attention", &masked_attention, "Masked Multi-Head Attention");
    m.def("combine_heads", &combine_heads, "Combine Multi-Head Outputs");
}