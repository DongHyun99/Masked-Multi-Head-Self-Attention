#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cmath>

// CUDA kernel for masked QKV projection
__global__ void masked_qkv_projection_kernel(
    const float* input,           // [batch_size, seq_len, hidden_dim]
    const float* weight_q,        // [hidden_dim, hidden_dim]
    const float* weight_k,        // [hidden_dim, hidden_dim]
    const float* weight_v,        // [hidden_dim, hidden_dim]
    const float* bias_q,          // [hidden_dim]
    const float* bias_k,          // [hidden_dim]
    const float* bias_v,          // [hidden_dim]
    const bool* mask,             // [batch_size, seq_len]
    float* query,                 // [batch_size, seq_len, hidden_dim]
    float* key,                   // [batch_size, seq_len, hidden_dim]
    float* value,                 // [batch_size, seq_len, hidden_dim]
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int dim_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || dim_idx >= hidden_dim) {
        return;
    }
    
    // Check if current token is masked
    bool is_masked = mask[batch_idx * seq_len + seq_idx];
    
    if (!is_masked) {
        // Zero out masked positions
        query[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + dim_idx] = 0.0f;
        key[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + dim_idx] = 0.0f;
        value[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + dim_idx] = 0.0f;
        return;
    }
    
    // Compute QKV projections for non-masked tokens
    float input_val = input[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + dim_idx];
    
    float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
    
    // Matrix multiplication with weight matrices
    for (int i = 0; i < hidden_dim; i++) {
        float inp = input[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + i];
        q_sum += inp * weight_q[i * hidden_dim + dim_idx];
        k_sum += inp * weight_k[i * hidden_dim + dim_idx];
        v_sum += inp * weight_v[i * hidden_dim + dim_idx];
    }
    
    // Add bias
    query[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + dim_idx] = q_sum + bias_q[dim_idx];
    key[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + dim_idx] = k_sum + bias_k[dim_idx];
    value[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + dim_idx] = v_sum + bias_v[dim_idx];
}

// Optimized CUDA kernel for masked attention computation
__global__ void masked_attention_kernel(
    const float* query,           // [batch_size, num_heads, seq_len, head_dim]
    const float* key,             // [batch_size, num_heads, seq_len, head_dim]
    const float* value,           // [batch_size, num_heads, seq_len, head_dim]
    const bool* mask,             // [batch_size, seq_len]
    float* attention_output,      // [batch_size, num_heads, seq_len, head_dim]
    float* attention_weights,     // [batch_size, num_heads, seq_len, seq_len]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int query_idx = blockIdx.z;
    int key_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || 
        query_idx >= seq_len || key_idx >= seq_len) {
        return;
    }
    
    // Check if query token is masked
    bool query_masked = !mask[batch_idx * seq_len + query_idx];
    bool key_masked = !mask[batch_idx * seq_len + key_idx];
    
    int q_offset = batch_idx * num_heads * seq_len * head_dim + 
                   head_idx * seq_len * head_dim + 
                   query_idx * head_dim;
    int k_offset = batch_idx * num_heads * seq_len * head_dim + 
                   head_idx * seq_len * head_dim + 
                   key_idx * head_dim;
    
    // Compute attention score
    float score = 0.0f;
    if (!query_masked && !key_masked) {
        for (int d = 0; d < head_dim; d++) {
            score += query[q_offset + d] * key[k_offset + d];
        }
        score *= scale;
    } else {
        score = -INFINITY; // Mask out invalid positions
    }
    
    // Store attention weight
    int weight_idx = batch_idx * num_heads * seq_len * seq_len + 
                     head_idx * seq_len * seq_len + 
                     query_idx * seq_len + key_idx;
    attention_weights[weight_idx] = score;
    
    __syncthreads();
    
    // Softmax computation (only for thread 0 of each query)
    if (key_idx == 0 && !query_masked) {
        // Find maximum for numerical stability
        float max_score = -INFINITY;
        for (int k = 0; k < seq_len; k++) {
            int idx = batch_idx * num_heads * seq_len * seq_len + 
                     head_idx * seq_len * seq_len + 
                     query_idx * seq_len + k;
            if (attention_weights[idx] > max_score) {
                max_score = attention_weights[idx];
            }
        }
        
        // Compute exponentials and sum
        float sum_exp = 0.0f;
        for (int k = 0; k < seq_len; k++) {
            int idx = batch_idx * num_heads * seq_len * seq_len + 
                     head_idx * seq_len * seq_len + 
                     query_idx * seq_len + k;
            if (attention_weights[idx] != -INFINITY) {
                attention_weights[idx] = expf(attention_weights[idx] - max_score);
                sum_exp += attention_weights[idx];
            } else {
                attention_weights[idx] = 0.0f;
            }
        }
        
        // Normalize
        if (sum_exp > 0.0f) {
            for (int k = 0; k < seq_len; k++) {
                int idx = batch_idx * num_heads * seq_len * seq_len + 
                         head_idx * seq_len * seq_len + 
                         query_idx * seq_len + k;
                attention_weights[idx] /= sum_exp;
            }
        }
    }
    
    __syncthreads();
    
    // Compute attention output
    if (key_idx < head_dim) {
        float output_val = 0.0f;
        
        if (!query_masked) {
            for (int k = 0; k < seq_len; k++) {
                int weight_idx = batch_idx * num_heads * seq_len * seq_len + 
                               head_idx * seq_len * seq_len + 
                               query_idx * seq_len + k;
                int v_offset = batch_idx * num_heads * seq_len * head_dim + 
                              head_idx * seq_len * head_dim + 
                              k * head_dim + key_idx;
                
                output_val += attention_weights[weight_idx] * value[v_offset];
            }
        }
        
        int output_offset = batch_idx * num_heads * seq_len * head_dim + 
                           head_idx * seq_len * head_dim + 
                           query_idx * head_dim + key_idx;
        attention_output[output_offset] = output_val;
    }
}

// CUDA kernel for reshaping and combining heads
__global__ void combine_heads_kernel(
    const float* multi_head_output,   // [batch_size, num_heads, seq_len, head_dim]
    const bool* mask,                 // [batch_size, seq_len]
    float* combined_output,           // [batch_size, seq_len, hidden_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int dim_idx = threadIdx.x;
    
    int hidden_dim = num_heads * head_dim;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || dim_idx >= hidden_dim) {
        return;
    }
    
    bool is_masked = !mask[batch_idx * seq_len + seq_idx];
    
    if (is_masked) {
        combined_output[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + dim_idx] = 0.0f;
        return;
    }
    
    int head_idx = dim_idx / head_dim;
    int head_dim_idx = dim_idx % head_dim;
    
    int input_idx = batch_idx * num_heads * seq_len * head_dim + 
                   head_idx * seq_len * head_dim + 
                   seq_idx * head_dim + head_dim_idx;
    
    combined_output[batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + dim_idx] = 
        multi_head_output[input_idx];
}

// Host functions
torch::Tensor masked_qkv_projection_cuda(
    torch::Tensor input,
    torch::Tensor weight_q,
    torch::Tensor weight_k,
    torch::Tensor weight_v,
    torch::Tensor bias_q,
    torch::Tensor bias_k,
    torch::Tensor bias_v,
    torch::Tensor mask
) {
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto hidden_dim = input.size(2);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto query = torch::zeros({batch_size, seq_len, hidden_dim}, options);
    auto key = torch::zeros({batch_size, seq_len, hidden_dim}, options);
    auto value = torch::zeros({batch_size, seq_len, hidden_dim}, options);
    
    dim3 grid_size(batch_size, seq_len);
    dim3 block_size(hidden_dim);
    
    masked_qkv_projection_kernel<<<grid_size, block_size>>>(
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
        batch_size, seq_len, hidden_dim
    );
    
    return torch::stack({query, key, value}, 0);
}

torch::Tensor masked_attention_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor mask
) {
    auto batch_size = query.size(0);
    auto num_heads = query.size(1);
    auto seq_len = query.size(2);
    auto head_dim = query.size(3);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
    auto attention_output = torch::zeros_like(query);
    auto attention_weights = torch::zeros({batch_size, num_heads, seq_len, seq_len}, options);
    
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    dim3 grid_size(batch_size, num_heads, seq_len);
    dim3 block_size(seq_len);
    
    masked_attention_kernel<<<grid_size, block_size>>>(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        mask.data_ptr<bool>(),
        attention_output.data_ptr<float>(),
        attention_weights.data_ptr<float>(),
        batch_size, num_heads, seq_len, head_dim, scale
    );
    
    return attention_output;
}

torch::Tensor combine_heads_cuda(
    torch::Tensor multi_head_output,
    torch::Tensor mask
) {
    auto batch_size = multi_head_output.size(0);
    auto num_heads = multi_head_output.size(1);
    auto seq_len = multi_head_output.size(2);
    auto head_dim = multi_head_output.size(3);
    auto hidden_dim = num_heads * head_dim;
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(multi_head_output.device());
    auto combined_output = torch::zeros({batch_size, seq_len, hidden_dim}, options);
    
    dim3 grid_size(batch_size, seq_len);
    dim3 block_size(hidden_dim);
    
    combine_heads_kernel<<<grid_size, block_size>>>(
        multi_head_output.data_ptr<float>(),
        mask.data_ptr<bool>(),
        combined_output.data_ptr<float>(),
        batch_size, num_heads, seq_len, head_dim
    );
    
    return combined_output;
}