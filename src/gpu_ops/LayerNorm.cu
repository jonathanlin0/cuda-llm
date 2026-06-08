#include "LayerNorm.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include "../ErrorCheck.h"

namespace {
constexpr int32_t kThreadsPerBlock = 256;
constexpr int32_t kMaxBlocks = 1024;

int32_t layernorm_num_blocks(int32_t len) {
    if (len <= 0) {
        return 1;
    }

    int32_t blocks = (len + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return blocks < kMaxBlocks ? blocks : kMaxBlocks;
}

/*
Computes the partial sums of squares (sum of squares for each block)
*/
__global__ void layernorm_partial_sum_kernel(const __nv_bfloat16 *hidden_state,
        int32_t len, float *partial_sums) {
    float sum = 0.0f;

    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;
    while (idx < len) {
        float value = __bfloat162float(hidden_state[idx]);
        sum += value * value;
        idx += stride;
    }

    extern __shared__ float shared_sums[];
    shared_sums[threadIdx.x] = sum;
    __syncthreads();

    for (int32_t offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = shared_sums[0];
    }
}

/*
Reduces the partial sums (one from each block in the prev kernel) into the single scale factor
*/
__global__ void layernorm_scale_kernel(const float *partial_sums, int32_t num_partials,
        int32_t len, float *scale) {
    float sum = 0.0f;

    for (int32_t i = threadIdx.x; i < num_partials; i += blockDim.x) {
        sum += partial_sums[i];
    }

    extern __shared__ float shared_sums[];
    shared_sums[threadIdx.x] = sum;
    __syncthreads();

    for (int32_t offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared_sums[threadIdx.x] += shared_sums[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *scale = rsqrtf(shared_sums[0] / static_cast<float>(len) + LayerNorm::EPS);
    }
}

/*
Applies the scale and writes the values to output
*/
__global__ void layernorm_apply_kernel(const __nv_bfloat16 *hidden_state,
        const __nv_bfloat16 *weights, const float *scale, int32_t len,
        __nv_bfloat16 *output) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;
    while (idx < len) {
        float value = __bfloat162float(hidden_state[idx]);
        float weight = __bfloat162float(weights[idx]);
        output[idx] = __float2bfloat16(weight * value * *scale);
        idx += stride;
    }
}
}

LayerNorm::LayerNorm(int32_t len): len(len) {
    int32_t num_blocks = layernorm_num_blocks(len);
    size_t temp_size = num_blocks * sizeof(float) + sizeof(float);
    temp_space = std::make_shared<CudaBuffer>(temp_size);
}

void LayerNorm::normalize_hidden_state(const std::shared_ptr<CudaBuffer> &hidden_state, const std::shared_ptr<CudaBuffer> &output, cudaStream_t stream) {
    int32_t num_blocks = layernorm_num_blocks(len);
    auto *base = static_cast<uint8_t*>(temp_space->data);
    auto *partial_sums = reinterpret_cast<float*>(base);
    auto *scale = reinterpret_cast<float*>(base + num_blocks * sizeof(float));

    size_t shared_size = kThreadsPerBlock * sizeof(float);
    layernorm_partial_sum_kernel<<<num_blocks, kThreadsPerBlock, shared_size, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden_state->data),
        len,
        partial_sums);
    checkCuda(cudaGetLastError());

    layernorm_scale_kernel<<<1, kThreadsPerBlock, shared_size, stream>>>(
        partial_sums,
        num_blocks,
        len,
        scale);
    checkCuda(cudaGetLastError());

    layernorm_apply_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden_state->data),
        static_cast<const __nv_bfloat16*>(weights->data),
        scale,
        len,
        static_cast<__nv_bfloat16*>(output->data));
    checkCuda(cudaGetLastError());
}
