#include "RoPE.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include "../ErrorCheck.h"

namespace {
constexpr int32_t kThreadsPerBlock = 256;

__global__ void apply_rope_to_qk_kernel(__nv_bfloat16 *x, int32_t num_heads,
        int32_t head_dim, int32_t position_idx, float theta_base) {
    int32_t half_dim = head_dim / 2;
    int32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t total_pairs = num_heads * half_dim;
    if (pair_idx >= total_pairs) {
        return;
    }

    // verified that these r correct. just hella confusing
    int32_t head_idx = pair_idx / half_dim;
    int32_t theta_idx = pair_idx % half_dim;

    int32_t head_offset = head_idx * head_dim; // wrt x
    int32_t first_idx = head_offset + theta_idx;
    int32_t second_idx = first_idx + half_dim;

    float a = __bfloat162float(x[first_idx]);
    float b = __bfloat162float(x[second_idx]);
    
    float theta = powf(theta_base, -static_cast<float>(theta_idx) / static_cast<float>(half_dim));
    float angle = static_cast<float>(position_idx) * theta;
    
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    x[first_idx] = __float2bfloat16(cos_val * a - sin_val * b);
    x[second_idx] = __float2bfloat16(cos_val * b + sin_val * a);
}
}

void RoPE::apply_rope_to_qk(__nv_bfloat16 *x, int32_t num_heads, int32_t head_dim,
        int32_t position_idx, float theta_base, cudaStream_t stream) {
    int32_t half_dim = head_dim / 2;
    int32_t total_pairs = num_heads * half_dim;
    if (total_pairs <= 0) {
        return;
    }

    int32_t num_blocks = (total_pairs + kThreadsPerBlock - 1) / kThreadsPerBlock;
    apply_rope_to_qk_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
        x,
        num_heads,
        head_dim,
        position_idx,
        theta_base);
    checkCuda(cudaGetLastError());
}
