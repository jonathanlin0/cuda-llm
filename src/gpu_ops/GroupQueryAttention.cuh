#pragma once

#include "../qwen2/Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <cuda_bf16.h>
#include <math_constants.h>
#include <memory>
#include "../ErrorCheck.h"

namespace {

template<Qwen2Size QWEN2_SIZE>
__global__ void group_query_attention_kernel(const __nv_bfloat16 *queries,
        const __nv_bfloat16 *k_cache, const __nv_bfloat16 *v_cache,
        float *weighted_values, int32_t layer_num, int32_t seq_len) {
    /*
    does flash attention (1 pass for attention calculation)

    a block launched for each head

    a thread corresponds with a dimension within the head. so each block will have 64 useful threads
    */
    using Qwen2ConfigT = Qwen2Config<QWEN2_SIZE>;

    constexpr int32_t num_query_heads = Qwen2ConfigT::num_query_heads();
    constexpr int32_t num_kv_heads = Qwen2ConfigT::num_kv_heads();
    constexpr int32_t head_size = Qwen2ConfigT::head_size();
    constexpr int32_t value_size = Qwen2ConfigT::value_size();
    constexpr int32_t num_layers = Qwen2ConfigT::num_layers();
    constexpr int32_t key_token_stride = num_layers * num_kv_heads * head_size;
    constexpr int32_t key_layer_stride = num_kv_heads * head_size;
    constexpr int32_t value_token_stride = num_layers * num_kv_heads * value_size;
    constexpr int32_t value_layer_stride = num_kv_heads * value_size;

    __shared__ float partials[256];
    __shared__ float max_score;
    __shared__ float denominator;
    __shared__ float old_scale;
    __shared__ float new_scale;
    __shared__ float previous_output_scale;
    __shared__ float current_value_scale;

    int32_t q_head_idx = blockIdx.x; // each block = 1 head
    if (q_head_idx >= num_query_heads) {
        return;
    }

    int32_t kv_head_idx = q_head_idx * num_kv_heads / num_query_heads;
    int32_t i = threadIdx.x; // dimension within the current head

    // Load the query row for q_head_idx.
    // queries shape: (num_query_heads, head_size)
    const __nv_bfloat16 *query_vec = queries + q_head_idx * head_size;

    // Initialize the scalar online softmax state for this query head.
    if (i == 0) {
        max_score = -CUDART_INF_F;
        denominator = 0.0f;
        old_scale = 0.0f;
        new_scale = 0.0f;
        previous_output_scale = 0.0f;
        current_value_scale = 0.0f;
    }
    __syncthreads();

    // Each thread i < value_size owns one output component for this query head.
    float out_i = 0.0f;

    // Stream over k_cache/v_cache for sequence positions [0, seq_len).
    // k_cache shape: (seq_len, num_layers, num_kv_heads, head_size)
    // v_cache shape: (seq_len, num_layers, num_kv_heads, value_size)

    // calculate the query key dot product
    // each loop represents the dot product between the query vec and key vec associated w token token_idx
    for (int32_t token_idx = 0; token_idx < seq_len; token_idx++) {
        const __nv_bfloat16 *key_vec =
            k_cache
            + token_idx * key_token_stride
            + layer_num * key_layer_stride
            + kv_head_idx * head_size;

        // Threads 0..63 compute one term of dot(query_vec, key_vec).
        // Threads above head_size write 0 so the block-wide reduction is valid.
        float partial = 0.0f;
        if (i < head_size) {
            partial = __bfloat162float(query_vec[i]) * __bfloat162float(key_vec[i]);
        }
        partials[i] = partial;
        __syncthreads();

        // Reduce the per-dimension products into one QK score for this token.
        for (int32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (i < stride) {
                partials[i] += partials[i + stride];
            }
            __syncthreads();
        }

        // Thread 0 now has this token's scaled QK score.
        if (i == 0) {
            float score = partials[0] * rsqrtf((float) head_size);

            // Online softmax update:
            // max_score tracks max(score[0..token_idx]) for numerical stability.
            // denominator tracks sum(exp(score[j] - max_score)).
            float previous_denominator = denominator;
            float next_max_score = fmaxf(max_score, score);
            old_scale = expf(max_score - next_max_score);
            new_scale = expf(score - next_max_score);
            denominator = denominator * old_scale + new_scale;
            max_score = next_max_score;

            // Coefficients for the normalized output recurrence:
            // out = out * (previous_denominator / denominator) * old_scale
            //     + value * (new_scale / denominator)
            previous_output_scale = previous_denominator * old_scale / denominator;
            current_value_scale = new_scale / denominator;
        }
        __syncthreads();

        const __nv_bfloat16 *value_vec =
            v_cache
            + token_idx * value_token_stride
            + layer_num * value_layer_stride
            + kv_head_idx * value_size;

        // Update one component of the running weighted value vector.
        if (i < value_size) {
            float value_i = __bfloat162float(value_vec[i]);
            out_i = out_i * previous_output_scale + value_i * current_value_scale;
        }
        __syncthreads();
    }

    // weighted_values shape: (num_query_heads, value_size)
    if (i < value_size) {
        weighted_values[q_head_idx * value_size + i] = out_i;
    }
}

}

template<Qwen2Size QWEN2_SIZE>
class GroupQueryAttention {
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;

    /**
     * Allocate temporary space
     */
    explicit GroupQueryAttention(int32_t max_seq_len) {
        // TODO
    }

    /**
     * Scaled dot product attention with grouped queries, see https://arxiv.org/abs/2305.13245.
     * Performs softmax((QK^T)/sqrt(d_k))*V for all queries Q and their associated K and V
     * - dot product each query with its target value throughout the sequence
     * - numerically stable softmax
     * - save a weighted sum of values
     * Does not perform the output projection.
     *
     * All inputs and outputs are row-major
     *
     * @param queries (num_query_heads, head_size)
     * @param k_cache (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache (seq_len, num_layers, num_kv_heads, value_size)
     * @param weighted_values (num_query_heads, value_size) outputs
     * @param layer_num layer index, starting at 0
     * @param seq_len current sequence length
     * @param stream CUDA stream for asynchronous operation
     */
    void sdpa(__nv_bfloat16 *queries, __nv_bfloat16 *k_cache, __nv_bfloat16 *v_cache, float *weighted_values, int32_t layer_num, int32_t seq_len, cudaStream_t stream) {
        // TODO: Tune block size and grid shape. This skeleton starts with one
        // block per query head.
        constexpr int32_t threads_per_block = 256; // using generic 256 size, but only 64 threads per block r doing useful work
        dim3 grid_dim(Qwen2Config::num_query_heads());
        dim3 block_dim(threads_per_block);

        group_query_attention_kernel<QWEN2_SIZE><<<grid_dim, block_dim, 0, stream>>>(
            queries,
            k_cache,
            v_cache,
            weighted_values,
            layer_num,
            seq_len
        );
        checkCuda(cudaGetLastError());
    }
};
