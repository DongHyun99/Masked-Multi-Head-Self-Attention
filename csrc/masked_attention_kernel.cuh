#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <torch/extension.h>

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int TILE_SIZE = 16;

// CUDA error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error: " + std::to_string(status)); \
        } \
    } while(0)

// Forward declarations
namespace masked_attention {

// Main attention function
torch::Tensor masked_multihead_attention_forward(
    const torch::Tensor& input,           // [batch_size, seq_len, d_model]
    const torch::Tensor& mask,            // [batch_size, seq_len] boolean mask
    const torch::Tensor& qkv_weight,      // [d_model, 3 * d_model]
    const torch::Tensor& qkv_bias,        // [3 * d_model]
    const torch::Tensor& proj_weight,     // [d_model, d_model]
    const torch::Tensor& proj_bias,       // [d_model]
    int num_heads,
    float dropout_p = 0.0,
    bool is_training = true
);

// Utility functions
torch::Tensor create_attention_mask(
    const torch::Tensor& mask,
    int num_heads
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> masked_qkv_projection(
    const torch::Tensor& input,
    const torch::Tensor& mask,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& qkv_bias,
    int num_heads
);

torch::Tensor masked_attention_scores(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& attention_mask,
    float scale
);

torch::Tensor masked_attention_output(
    const torch::Tensor& attention_probs,
    const torch::Tensor& v,
    const torch::Tensor& mask
);

torch::Tensor masked_output_projection(
    const torch::Tensor& attention_output,
    const torch::Tensor& mask,
    const torch::Tensor& proj_weight,
    const torch::Tensor& proj_bias
);

} // namespace masked_attention

// CUDA kernel declarations
extern "C" {

// Optimized QKV projection kernel
template<typename T>
__global__ void masked_qkv_kernel(
    const T* __restrict__ input,
    const bool* __restrict__ mask,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    T* __restrict__ v_out,
    int batch_size,
    int seq_len,
    int d_model,
    int head_dim
);

// Optimized attention computation kernel
template<typename T>
__global__ void masked_attention_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    const bool* __restrict__ attention_mask,
    T* __restrict__ output,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
);

// Output projection kernel
template<typename T>
__global__ void masked_output_projection_kernel(
    const T* __restrict__ input,
    const bool* __restrict__ mask,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int d_model
);

// Mask creation kernel
__global__ void create_attention_mask_kernel(
    const bool* __restrict__ token_mask,
    bool* __restrict__ attention_mask,
    int batch_size,
    int seq_len
);

} // extern "C"

// Template specializations
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