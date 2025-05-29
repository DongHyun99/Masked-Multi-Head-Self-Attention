#include "masked_attention_kernel.cu"

// Launch wrapper for QKV projection
void launch_masked_qkv_projection_float(
    const float* input, const float* weight_q, const float* weight_k, const float* weight_v,
    const float* bias_q, const float* bias_k, const float* bias_v, const bool* mask,
    float* query, float* key, float* value,
    int batch_size, int seq_len, int d_model, int num_heads, int head_dim,
    cudaStream_t stream
) {
    // Calculate optimal grid and block dimensions
    dim3 block(min(d_model, MAX_THREADS_PER_BLOCK));
    dim3 grid(batch_size * seq_len);
    
    size_t shared_mem_size = d_model * sizeof(float);
    
    // Launch QKV projection kernel
    masked_qkv_projection_kernel<float><<<grid, block, shared_mem_size, stream>>>(
        input, weight_q, weight_k, weight_v, bias_q, bias_k, bias_v, mask,
        query, key, value, batch_size, seq_len, d_model, num_heads, head_dim
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Launch wrapper for attention computation
void launch_masked_attention_float(
    const float* query, const float* key, const float* value, const bool* mask,
    float* output, float* attention_weights,
    int batch_size, int num_heads, int seq_len, int head_dim, float scale,
    cudaStream_t stream
) {
    // Optimize for different sequence lengths
    dim3 block, grid;
    size_t shared_mem_size;
    
    if (seq_len <= 512) {
        // Small sequence optimization
        block = dim3(32, min(seq_len, 16));
        grid = dim3(batch_size * num_heads, (seq_len + block.y - 1) / block.y);
        shared_mem_size = (4 * 16 * head_dim) * sizeof(float);
    } else {
        // Large sequence optimization with tiling
        block = dim3(32, 16);
        grid = dim3(batch_size * num_heads, (seq_len + 15) / 16);
        shared_mem_size = (4 * 16 * head_dim) * sizeof(float);
    }
    
    // Initialize output to zero for masked positions
    CUDA_CHECK(cudaMemsetAsync(output, 0, 
        batch_size * num_heads * seq_len * head_dim * sizeof(float), stream));
    
    // Launch attention kernel
    masked_attention_kernel<float><<<grid, block, shared_mem_size, stream>>>(
        query, key, value, mask, output, attention_weights,
        batch_size, num_heads, seq_len, head_dim, scale
    );
    
    CUDA_CHECK(cudaGetLastError());
}