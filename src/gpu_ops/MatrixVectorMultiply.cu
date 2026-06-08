#include "MatrixVectorMultiply.cuh"
#include "../ErrorCheck.h"
#include <cstddef>
#include <cstdint>

namespace {
constexpr int32_t kThreadsPerBlock = 256;

__device__ inline float matvec_to_float(float value) {
    return value;
}

__device__ inline float matvec_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template<typename input_float_t>
__global__ void bf16_matmul_kernel(int32_t m, int32_t k, const __nv_bfloat16 *mat,
        const __nv_bfloat16 *bias, const input_float_t *vec, __nv_bfloat16 *out) {
    /*
    launches 1 block per dimension of the output vector

    (m x k) * (k) -> (m)

    not most efficient implementation, because kernel significantly slows down with large k, since
    parallelism scales with m, and not k. large k causes more serial work by the threads in the initial sweep.
    but this is fine, because the ratio of m to k for the matrix vector multplications isn't terrible
    in the qwen2 0.5b model.
    */
    int32_t row = blockIdx.x;
    if (row >= m) {
        return;
    }

    float sum = 0.0f;
    for (int32_t col = threadIdx.x; col < k; col += blockDim.x) {
        sum += __bfloat162float(mat[row * k + col]) * matvec_to_float(vec[col]);
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
        float result = shared_sums[0];
        if (bias != nullptr) {
            result += __bfloat162float(bias[row]);
        }
        out[row] = __float2bfloat16(result);
    }
}
}

template<typename input_float_t>
void MatrixVectorMultiply::bf16_matmul(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, input_float_t *vec, __nv_bfloat16 *out, cudaStream_t stream) {
    if (m <= 0) {
        return;
    }

    size_t shared_size = kThreadsPerBlock * sizeof(float);
    bf16_matmul_kernel<<<m, kThreadsPerBlock, shared_size, stream>>>(m, k, mat, bias, vec, out);
    checkCuda(cudaGetLastError());
}

// explicit instantiations
template void MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, __nv_bfloat16 *vec, __nv_bfloat16 *out, cudaStream_t stream);
template void MatrixVectorMultiply::bf16_matmul<float>(int32_t m, int32_t k, __nv_bfloat16 *mat, __nv_bfloat16* bias, float *vec, __nv_bfloat16 *out, cudaStream_t stream);
