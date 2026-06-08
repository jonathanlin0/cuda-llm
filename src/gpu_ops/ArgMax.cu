#include "ArgMax.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include "../ErrorCheck.h"

namespace {
// Keep a fixed block size so the shared-memory reductions are simple.
constexpr int32_t kThreadsPerBlock = 256;
constexpr int32_t kMaxBlocks = 1024;

// Launch enough blocks to expose parallelism, but cap the first-stage output so
// the second-stage reduction remains small and fits in one block.
int32_t argmax_num_blocks(int32_t len) {
    if (len <= 0) {
        return 1;
    }

    int32_t blocks = (len + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return blocks < kMaxBlocks ? blocks : kMaxBlocks;
}

// Compare (value, index) pairs. Higher value wins; ties go to the lower index,
// matching the ArgMax contract in the header.
__device__ bool argmax_is_better(float candidate_value, int32_t candidate_index,
        float best_value, int32_t best_index) {
    if (candidate_index < 0) {
        return false;
    }
    return best_index < 0
        || candidate_value > best_value
        || (candidate_value == best_value && candidate_index < best_index);
}

__global__ void bf16_argmax_partial_kernel(const __nv_bfloat16 *data, int32_t len,
        float *partial_values, int32_t *partial_indices) {
    // Each thread scans a grid-stride slice of the input and keeps its local
    // best (value, original input index) pair.

    /*
    Then, a reduction is done between the threads in the block (in shared memory) to find the block's max value and corresponding index
    */
    float best_value = -INFINITY;
    int32_t best_index = -1;

    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;
    while (idx < len) {
        float value = __bfloat162float(data[idx]);
        if (argmax_is_better(value, idx, best_value, best_index)) {
            best_value = value;
            best_index = idx;
        }
        idx += stride;
    }

    // Store one candidate per thread in shared memory, then reduce within the
    // block. Values and indices are kept in parallel arrays.
    extern __shared__ float shared_values[];
    int32_t *shared_indices = reinterpret_cast<int32_t*>(&shared_values[blockDim.x]);
    shared_values[threadIdx.x] = best_value;
    shared_indices[threadIdx.x] = best_index;
    __syncthreads();

    // we know that block dim is fixed at 256, so we can simplify this from arbitrary-length arrs
    for (int32_t offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset && argmax_is_better(shared_values[threadIdx.x + offset],
                shared_indices[threadIdx.x + offset], shared_values[threadIdx.x],
                shared_indices[threadIdx.x])) {
            shared_values[threadIdx.x] = shared_values[threadIdx.x + offset];
            shared_indices[threadIdx.x] = shared_indices[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // One partial result per block is written to global scratch space.
    if (threadIdx.x == 0) {
        partial_values[blockIdx.x] = shared_values[0];
        partial_indices[blockIdx.x] = shared_indices[0];
    }
}

__global__ void argmax_final_kernel(const float *partial_values, const int32_t *partial_indices,
        int32_t num_partials, int32_t *output_index) {
    // The first kernel emits at most kMaxBlocks partials, so one block can finish
    // the reduction and write the final device-memory index.

    /*
    First loop strides by blockDim.x and collects the max for the number of indices associated w the current thread.
    The number of loops will be ceil(num blocks from the partial kernel / num threads in the grid for argmax final kernel).
    So this is in case there were more blocks used in the partial kernel than threads in this kernel
    */
    float best_value = -INFINITY;
    int32_t best_index = -1;

    for (int32_t i = threadIdx.x; i < num_partials; i += blockDim.x) {
        if (argmax_is_better(partial_values[i], partial_indices[i], best_value, best_index)) {
            best_value = partial_values[i];
            best_index = partial_indices[i];
        }
    }

    /*
    Simple reduction done across all the threads
    */
    extern __shared__ float shared_values[];
    int32_t *shared_indices = reinterpret_cast<int32_t*>(&shared_values[blockDim.x]);
    shared_values[threadIdx.x] = best_value;
    shared_indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int32_t offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset && argmax_is_better(shared_values[threadIdx.x + offset],
                shared_indices[threadIdx.x + offset], shared_values[threadIdx.x],
                shared_indices[threadIdx.x])) {
            shared_values[threadIdx.x] = shared_values[threadIdx.x + offset];
            shared_indices[threadIdx.x] = shared_indices[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output_index = shared_indices[0];
    }
}
}

ArgMax::ArgMax(int32_t len): len(len) {
    int32_t num_blocks = argmax_num_blocks(len);
    // Scratch layout:
    // [num_blocks floats for partial values]
    // [num_blocks int32s for partial indices]
    // [one int32 final output index]
    size_t temp_size = num_blocks * sizeof(float)
        + num_blocks * sizeof(int32_t)
        + sizeof(int32_t);
    temp_space = std::make_shared<CudaBuffer>(temp_size);
}

int32_t *ArgMax::bf16_argmax(const std::shared_ptr<CudaBuffer> &bf16_data, cudaStream_t stream) {
    int32_t num_blocks = argmax_num_blocks(len);
    // Split the constructor-allocated scratch buffer into typed regions.
    auto *base = static_cast<uint8_t*>(temp_space->data);
    auto *partial_values = reinterpret_cast<float*>(base);
    auto *partial_indices = reinterpret_cast<int32_t*>(base + num_blocks * sizeof(float));
    auto *output_index = reinterpret_cast<int32_t*>(base
        + num_blocks * sizeof(float)
        + num_blocks * sizeof(int32_t)); // last spot in the scratch buffer

    size_t shared_size = kThreadsPerBlock * (sizeof(float) + sizeof(int32_t));
    // Stage 1: reduce the full BF16 input into one partial pair per block.
    bf16_argmax_partial_kernel<<<num_blocks, kThreadsPerBlock, shared_size, stream>>>(
        static_cast<const __nv_bfloat16*>(bf16_data->data),
        len,
        partial_values,
        partial_indices);
    checkCuda(cudaGetLastError());

    // Stage 2: reduce the partial pairs to a single output index.
    argmax_final_kernel<<<1, kThreadsPerBlock, shared_size, stream>>>(
        partial_values,
        partial_indices,
        num_blocks,
        output_index);
    checkCuda(cudaGetLastError());

    return output_index;
}
