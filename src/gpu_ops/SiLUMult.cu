#include "SiLUMult.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include "../ErrorCheck.h"

namespace {
constexpr int32_t kThreadsPerBlock = 256;
constexpr int32_t kMaxBlocks = 1024;

int32_t silu_mult_num_blocks(int32_t len) {
    if (len <= 0) {
        return 1;
    }

    int32_t blocks = (len + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return blocks < kMaxBlocks ? blocks : kMaxBlocks;
}

__global__ void silu_mult_kernel(__nv_bfloat16 *x, const __nv_bfloat16 *y, int32_t len) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;

    while (idx < len) {
        float x_value = __bfloat162float(x[idx]);
        float y_value = __bfloat162float(y[idx]);
        float silu = x_value / (1.0f + expf(-x_value));
        x[idx] = __float2bfloat16(silu * y_value);
        idx += stride;
    }
}
}

void SiLUMult::silu_mult_in_place(const std::shared_ptr<CudaBuffer> &x, const std::shared_ptr<CudaBuffer> &y, cudaStream_t stream) {
    int32_t len = static_cast<int32_t>(x->size / sizeof(__nv_bfloat16));
    int32_t num_blocks = silu_mult_num_blocks(len);

    silu_mult_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x->data),
        static_cast<const __nv_bfloat16*>(y->data),
        len);
    checkCuda(cudaGetLastError());
}
