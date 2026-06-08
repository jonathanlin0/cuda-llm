#pragma once

#include <cuda_bf16.h>

#include "Qwen2Config.h"
#include "../CudaBuffer.cuh"
#include <memory>

#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/RoPE.cuh"
#include "../gpu_ops/GroupQueryAttention.cuh"
#include "../gpu_ops/SiLUMult.cuh"

namespace {

__global__ void bf16_add_in_place_kernel(
    __nv_bfloat16 *x,
    const __nv_bfloat16 *y,
    int32_t len
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        float sum = __bfloat162float(x[idx]) + __bfloat162float(y[idx]);
        x[idx] = __float2bfloat16(sum);
    }
}

inline void bf16_add_in_place(
    const std::shared_ptr<CudaBuffer> &x,
    const std::shared_ptr<CudaBuffer> &y,
    int32_t len,
    cudaStream_t stream
) {
    constexpr int32_t threads_per_block = 256;
    // ceiling division
    int32_t blocks = (len + threads_per_block - 1) / threads_per_block;

    bf16_add_in_place_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x->data),
        static_cast<const __nv_bfloat16*>(y->data),
        len);

    checkCuda(cudaGetLastError());
}

}


template<Qwen2Size QWEN2_SIZE>
class Qwen2Layer {
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
    std::shared_ptr<CudaBuffer> attn_input;
    std::shared_ptr<CudaBuffer> queries;
    std::shared_ptr<CudaBuffer> new_keys;
    std::shared_ptr<CudaBuffer> new_values;
    std::shared_ptr<CudaBuffer> attention_values; // referenced as weighted_values from python file qwen2.py
    GroupQueryAttention<QWEN2_SIZE> gqa;
    std::shared_ptr<CudaBuffer> attn_output;
    std::shared_ptr<CudaBuffer> ffn_input;
    std::shared_ptr<CudaBuffer> gate_output;
    std::shared_ptr<CudaBuffer> up_output;
    std::shared_ptr<CudaBuffer> ffn_output;

    Qwen2Layer(uint32_t layer_num, uint32_t max_seq_len):
    layer_num(layer_num), input_layernorm(Qwen2Config::hidden_size()), gqa(max_seq_len), post_attention_layernorm(Qwen2Config::hidden_size()) {
        // TODO
        attn_input = std::make_shared<CudaBuffer>(
            Qwen2Config::hidden_size() * sizeof(__nv_bfloat16)
        );

        queries = std::make_shared<CudaBuffer>(
            Qwen2Config::queries_size() * sizeof(__nv_bfloat16));

        new_keys = std::make_shared<CudaBuffer>(
            Qwen2Config::keys_size() * sizeof(__nv_bfloat16));

        new_values = std::make_shared<CudaBuffer>(
            Qwen2Config::values_size() * sizeof(__nv_bfloat16));

        attention_values = std::make_shared<CudaBuffer>(
            Qwen2Config::num_query_heads()
            * Qwen2Config::value_size()
            * sizeof(float));

        attn_output = std::make_shared<CudaBuffer>(
            Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));

        ffn_input = std::make_shared<CudaBuffer>(
            Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));

        gate_output = std::make_shared<CudaBuffer>(
            Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));

        up_output = std::make_shared<CudaBuffer>(
            Qwen2Config::intermediate_size() * sizeof(__nv_bfloat16));

        ffn_output = std::make_shared<CudaBuffer>(
            Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));

    }

    uint32_t layer_num;
    LayerNorm input_layernorm;                              // (hidden_size,)
    std::shared_ptr<CudaBuffer> q_proj_weight;              // (queries_size, hidden_size)
    std::shared_ptr<CudaBuffer> q_proj_bias;                // (queries_size,)
    std::shared_ptr<CudaBuffer> k_proj_weight;              // (keys_size, hidden_size)
    std::shared_ptr<CudaBuffer> k_proj_bias;                // (keys_size,)
    std::shared_ptr<CudaBuffer> v_proj_weight;              // (values_size, hidden_size)
    std::shared_ptr<CudaBuffer> v_proj_bias;                // (values_size,)
    std::shared_ptr<CudaBuffer> o_proj_weight;              // (hidden_size, queries_size)
    LayerNorm post_attention_layernorm;                     // (hidden_size,)
    std::shared_ptr<CudaBuffer> up_proj_weight;             // (intermediate_size, hidden_size)
    std::shared_ptr<CudaBuffer> gate_proj_weight;           // (intermediate_size, hidden_size)
    std::shared_ptr<CudaBuffer> down_proj_weight;           // (hidden_size, intermediate_size)

    /**
     * Pass the hidden state through this layer. Modifies the hidden state in-place.
     * @param k_cache bf16 keys (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache bf16 values (seq_len, num_layers, num_kv_heads, value_size)
     * @param hidden_state current hidden state bf16 (hidden_size,)
     * @param seq_len current sequence length
     * @param stream CUDA stream for asynchronous operation
     */
    void forward(const std::shared_ptr<CudaBuffer>& k_cache, const std::shared_ptr<CudaBuffer> &v_cache, const std::shared_ptr<CudaBuffer> &hidden_state, int32_t seq_len, cudaStream_t stream) {
        // TODO
        
        // initial layernorm
        input_layernorm.normalize_hidden_state(hidden_state, attn_input, stream);

        // calculate query key value vectors of the current token
        // this includes the bias term
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::queries_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(q_proj_weight->data),
            static_cast<__nv_bfloat16*>(q_proj_bias->data),
            static_cast<__nv_bfloat16*>(attn_input->data),
            static_cast<__nv_bfloat16*>(queries->data),
            stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::keys_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(k_proj_weight->data),
            static_cast<__nv_bfloat16*>(k_proj_bias->data),
            static_cast<__nv_bfloat16*>(attn_input->data),
            static_cast<__nv_bfloat16*>(new_keys->data),
            stream);

        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::values_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(v_proj_weight->data),
            static_cast<__nv_bfloat16*>(v_proj_bias->data),
            static_cast<__nv_bfloat16*>(attn_input->data),
            static_cast<__nv_bfloat16*>(new_values->data),
            stream);

        RoPE::apply_rope_to_qk(
            static_cast<__nv_bfloat16*>(queries->data),
            Qwen2Config::num_query_heads(),
            Qwen2Config::head_size(),
            seq_len - 1,
            Qwen2Config::rope_theta_base(),
            stream);

        RoPE::apply_rope_to_qk(
            static_cast<__nv_bfloat16*>(new_keys->data),
            Qwen2Config::num_kv_heads(),
            Qwen2Config::head_size(),
            seq_len - 1,
            Qwen2Config::rope_theta_base(),
            stream);

        int32_t k_offset =
            (seq_len - 1) * Qwen2Config::num_layers() * Qwen2Config::keys_size()
            + layer_num * Qwen2Config::keys_size();

        int32_t v_offset =
            (seq_len - 1) * Qwen2Config::num_layers() * Qwen2Config::values_size()
            + layer_num * Qwen2Config::values_size();

        checkCuda(cudaMemcpyAsync(
            static_cast<__nv_bfloat16*>(k_cache->data) + k_offset,
            static_cast<__nv_bfloat16*>(new_keys->data),
            Qwen2Config::keys_size() * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice,
            stream));

        checkCuda(cudaMemcpyAsync(
            static_cast<__nv_bfloat16*>(v_cache->data) + v_offset,
            static_cast<__nv_bfloat16*>(new_values->data),
            Qwen2Config::values_size() * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice,
            stream));
        
        // grouped query attention
        gqa.sdpa(
            static_cast<__nv_bfloat16*>(queries->data),
            static_cast<__nv_bfloat16*>(k_cache->data),
            static_cast<__nv_bfloat16*>(v_cache->data),
            static_cast<float*>(attention_values->data),
            layer_num,
            seq_len,
            stream);

        // project group query attention output of shape (num_heads * head_dim,) -> (hidden_dim,)
        MatrixVectorMultiply::bf16_matmul<float>(
            Qwen2Config::hidden_size(),
            Qwen2Config::queries_size(),
            static_cast<__nv_bfloat16*>(o_proj_weight->data),
            nullptr,
            static_cast<float*>(attention_values->data),
            static_cast<__nv_bfloat16*>(attn_output->data),
            stream);
        
        // add residual connection to attn_output
        bf16_add_in_place(
            hidden_state,
            attn_output,
            Qwen2Config::hidden_size(),
            stream);

        // another layer norm
        post_attention_layernorm.normalize_hidden_state(hidden_state, ffn_input, stream);

        // SwiGLU
        // gate_proj_weight (intermediate_size, hidden_size)
        // ffn_input (hidden_size,)
        // gate_proj_weight x ffn_input = gate_output (intermediate_size)
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::intermediate_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(gate_proj_weight->data),
            nullptr,
            static_cast<__nv_bfloat16*>(ffn_input->data),
            static_cast<__nv_bfloat16*>(gate_output->data),
            stream);

        // up_output (intermediate_size, hidden_size)
        // ffn_input (hidden_size,)
        // up_output x ffn_input = gate_output (intermediate_size)
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::intermediate_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(up_proj_weight->data),
            nullptr,
            static_cast<__nv_bfloat16*>(ffn_input->data),
            static_cast<__nv_bfloat16*>(up_output->data),
            stream);

        SiLUMult::silu_mult_in_place(gate_output, up_output, stream);

        // project from (intermediate_size,) back to (hidden_size,)
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::hidden_size(),
            Qwen2Config::intermediate_size(),
            static_cast<__nv_bfloat16*>(down_proj_weight->data),
            nullptr,
            static_cast<__nv_bfloat16*>(gate_output->data),
            static_cast<__nv_bfloat16*>(ffn_output->data),
            stream);

        // final residual
        bf16_add_in_place(
            hidden_state,
            ffn_output,
            Qwen2Config::hidden_size(),
            stream);
    
    }

};
