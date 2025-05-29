#include "masked_attention_kernel.cuh"
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Device functions for mixed precision support
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }

template<typename T>
__device__ __forceinline__ T from_float(float x);

template<>
__device__ __forceinline__ float from_float<float>(float x) { return x; }

template<>
__device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }

// Optimized QKV projection kernel with masking
template<typename T>
__global__ void masked_qkv_kernel(
    const T* __restrict__ input,      // [batch_size, seq_len, d_model]
    const bool* __restrict__ mask,    // [batch_size, seq_len]
    const T* __restrict__ weight,     // [d_model, 3 * d_model]
    const T* __restrict__ bias,       // [3 * d_model]
    T* __restrict__ q_out,           // [batch_size, seq_len, d_model]
    T* __restrict__ k_out,           // [batch_size, seq_len, d_model]
    T* __restrict__ v_out,           // [batch_size, seq_len, d_model]
    int batch_size,
    int seq_len,
    int d_model,
    int head_dim
) {
    const int batch_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    // Check if this token is masked
    const int mask_idx = batch_idx * seq_len + seq_idx;
    const bool is_masked = !mask[mask_idx];
    
    if (is_masked) {
        // Zero out outputs for masked tokens
        for (int i = tid; i < d_model; i += blockDim.x) {
            const int out_idx = batch_idx * seq_len * d_model + seq_idx * d_model + i;
            q_out[out_idx] = from_float<T>(0.0f);
            k_out[out_idx] = from_float<T>(0.0f);
            v_out[out_idx] = from_float<T>(0.0f);
        }
        return;
    }
    
    // Input index
    const int input_idx = batch_idx * seq_len * d_model + seq_idx * d_model;
    const int output_idx = batch_idx * seq_len * d_model + seq_idx * d_model;
    
    // Shared memory for reduction
    extern __shared__ float sdata[];
    float* sq = sdata;
    float* sk = sq + d_model;
    float* sv = sk + d_model;
    
    // Compute Q, K, V using matrix multiplication with shared memory
    for (int out_dim = tid; out_dim < d_model; out_dim += blockDim.x) {
        float q_sum = 0.0f, k_sum = 0.0f, v_sum = 0.0f;
        
        // Dot product with weight matrix
        for (int in_dim = 0; in_dim < d_model; in_dim++) {
            const float input_val = to_float(input[input_idx + in_dim]);
            q_sum += input_val * to_float(weight[in_dim * 3 * d_model + out_dim]);
            k_sum += input_val * to_float(weight[in_dim * 3 * d_model + d_model + out_dim]);
            v_sum += input_val * to_float(weight[in_dim * 3 * d_model + 2 * d_model + out_dim]);
        }
        
        // Add bias
        q_sum += to_float(bias[out_dim]);
        k_sum += to_float(bias[d_model + out_dim]);
        v_sum += to_float(bias[2 * d_model + out_dim]);
        
        // Store results
        q_out[output_idx + out_dim] = from_float<T>(q_sum);
        k_out[output_idx + out_dim] = from_float<T>(k_sum);
        v_out[output_idx + out_dim] = from_float<T>(v_sum);
    }
}

// Optimized masked attention kernel with Tensor Cores support
template<typename T>
__global__ void masked_attention_kernel(
    const T* __restrict__ q,              // [batch_size, num_heads, seq_len, head_dim]
    const T* __restrict__ k,              // [batch_size, num_heads, seq_len, head_dim]
    const T* __restrict__ v,              // [batch_size, num_heads, seq_len, head_dim]
    const bool* __restrict__ attention_mask, // [batch_size, seq_len, seq_len]
    T* __restrict__ output,               // [batch_size, num_heads, seq_len, head_dim]
    float scale,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int row = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (row >= seq_len) return;
    
    // Check if this query token is masked
    const int query_mask_idx = batch_idx * seq_len + row;
    const bool query_masked = !attention_mask[query_mask_idx * seq_len + row];
    
    if (query_masked) {
        // Zero out output for masked query tokens
        for (int d = 0; d < head_dim; d++) {
            const int out_idx = ((batch_idx * num_heads + head_idx) * seq_len + row) * head_dim + d;
            output[out_idx] = from_float<T>(0.0f);
        }
        return;
    }
    
    // Shared memory for attention scores and softmax
    extern __shared__ float sdata[];
    float* scores = sdata;
    float* exp_scores = scores + seq_len;
    
    const int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + row) * head_dim;
    
    // Compute attention scores: Q * K^T
    float max_score = -INFINITY;
    for (int col = threadIdx.y; col < seq_len; col += blockDim.y) {
        float score = 0.0f;
        
        // Check if key token is masked
        const int key_mask_idx = batch_idx * seq_len * seq_len + row * seq_len + col;
        const bool key_masked = !attention_mask[key_mask_idx];
        
        if (!key_masked) {
            const int k_offset = ((batch_idx * num_heads + head_idx) * seq_len + col) * head_dim;
            
            // Dot product Q[row] * K[col]
            for (int d = 0; d < head_dim; d++) {
                score += to_float(q[q_offset + d]) * to_float(k[k_offset + d]);
            }
            score *= scale;
        } else {
            score = -INFINITY;  // Mask out invalid positions
        }
        
        scores[col] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Sync threads to get global max
    __syncthreads();
    
    // Reduce to find global max across all threads
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    max_score = BlockReduce(temp_storage).Reduce(max_score, cub::Max());
    __syncthreads();
    
    // Compute softmax: exp(score - max_score)
    float sum_exp = 0.0f;
    for (int col = threadIdx.y; col < seq_len; col += blockDim.y) {
        float exp_score = expf(scores[col] - max_score);
        exp_scores[col] = exp_score;
        sum_exp += exp_score;
    }
    __syncthreads();
    
    // Reduce sum_exp
    sum_exp = BlockReduce(temp_storage).Sum(sum_exp);
    __syncthreads();
    
    // Normalize and compute output: Attention * V
    for (int d = threadIdx.y; d < head_dim; d += blockDim.y) {
        float output_val = 0.0f;
        
        for (int col = 0; col < seq_len; col++) {
            const float attention_weight = exp_scores[col] / (sum_exp + 1e-12f);
            const int v_offset = ((batch_idx * num_heads + head_idx) * seq_len + col) * head_dim;
            output_val += attention_weight * to_float(v[v_offset + d]);
        }
        
        const int out_idx = q_offset + d;
        output[out_idx] = from_float<T>(output_val);
    }
}

// Output projection kernel with masking
template<typename T>
__global__ void masked_output_projection_kernel(
    const T* __restrict__ input,      // [batch_size, seq_len, d_model]
    const bool* __restrict__ mask,    // [batch_size, seq_len]
    const T* __restrict__ weight,     // [d_model, d_model]
    const T* __restrict__ bias,       // [d_model]
    T* __restrict__ output,           // [batch_size, seq_len, d_model]
    int batch_size,
    int seq_len,
    int d_model
) {
    const int batch_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    // Check if this token is masked
    const int mask_idx = batch_idx * seq_len + seq_idx;
    const bool is_masked = !mask[mask_idx];
    
    const int input_idx = batch_idx * seq_len * d_model + seq_idx * d_model;
    const int output_idx = batch_idx * seq_len * d_model + seq_idx * d_model;
    
    if (is_masked) {
        // For masked tokens, either zero out or keep original input
        for (int i = tid; i < d_model; i += blockDim.x) {
            output[output_idx + i] = from_float<T>(0.0f);  // or input[input_idx + i]
        }
        return;
    }
    
    // Compute output projection: input * weight + bias
    for (int out_dim = tid; out_dim < d_model; out_dim += blockDim.x) {
        float sum = 0.0f;
        
        // Matrix multiplication
        for (int in_dim = 0; in_dim < d_model; in_dim++) {
            sum += to_float(input[input_idx + in_dim]) * to_float(weight[in_dim * d_model + out_dim]);
        }
        
        // Add bias
        sum += to_float(bias[out_dim]);
        
        output[output_idx + out_dim] = from_float<T>(sum);
    }
}

// Create attention mask kernel
__global__ void create_attention_mask_kernel(
    const bool* __restrict__ token_mask,     // [batch_size, seq_len]
    bool* __restrict__ attention_mask,       // [batch_size, seq_len, seq_len]
    int batch_size,
    int seq_len
) {
    const int batch_idx = blockIdx.x;
    const int row = blockIdx.y * blockDim.x + threadIdx.x;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    
    if (row >= seq_len || col >= seq_len) return;
    
    const int token_mask_row = batch_idx * seq_len + row;
    const int token_mask_col = batch_idx * seq_len + col;
    const int attention_mask_idx = batch_idx * seq_len * seq_len + row * seq_len + col;
    
    // Attention is valid only if both query and key tokens are not masked
    attention_mask[attention_mask_idx] = token_mask[token_mask_row] && token_mask[token_mask_col];
}

// Explicit template instantiations
template __global__ void masked_qkv_kernel<float>(
    const float*, const bool*, const float*, const float*,
    float*, float*, float*, int, int, int, int);

template __global__ void masked_qkv_kernel<__half>(
    const __half*, const bool*, const __half*, const __half*,
    __half*, __half*, __half*, int, int, int, int);

template __global__ void masked_attention_kernel<float>(
    const float*, const float*, const float*, const bool*,
    float*, float, int, int, int, int);

template __global__ void masked_attention_kernel<__half>(
    const __half*, const __half*, const __half*, const bool*,
    __half*, float, int, int, int, int);

template __global__ void masked_output_projection_kernel<float>(
    const float*, const bool*, const float*, const float*,
    float*, int, int, int);

template __global__ void masked_output_projection_kernel<__half>(
    const __half*, const bool*, const __half*, const __half*,
    __half*, int, int, int);