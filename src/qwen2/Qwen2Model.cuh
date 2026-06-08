#pragma once

#include <memory>

#include "Qwen2Layer.cuh"
#include "Qwen2Config.h"
#include "../ErrorCheck.h"
#include "../gpu_ops/LayerNorm.cuh"
#include "../gpu_ops/ArgMax.cuh"

template<Qwen2Size QWEN2_SIZE>
class Qwen2Model {
    cudaStream_t stream;
public:
    using Qwen2Config = Qwen2Config<QWEN2_SIZE>;
    using Qwen2Layer = Qwen2Layer<QWEN2_SIZE>;

    std::shared_ptr<CudaBuffer> hidden_state; // starts as embedding, then gets passed through the layers (model)
    std::shared_ptr<CudaBuffer> final_hidden_state;
    std::shared_ptr<CudaBuffer> logits;
    ArgMax argmax{Qwen2Config::vocab_size()};

    Qwen2Model() {
        checkCuda(cudaStreamCreate(&stream));
        hidden_state = std::make_shared<CudaBuffer>(
            Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));

        final_hidden_state = std::make_shared<CudaBuffer>(
            Qwen2Config::hidden_size() * sizeof(__nv_bfloat16));

        logits = std::make_shared<CudaBuffer>(
            Qwen2Config::vocab_size() * sizeof(__nv_bfloat16));
    }

    // deconstructor
    ~Qwen2Model() {
        checkCuda(cudaStreamDestroy(stream));
    }

    // these are all populated by Qwen2Loader
    std::shared_ptr<CudaBuffer> embedding_weight; // (vocab_size, hidden_size)
    std::shared_ptr<Qwen2Layer> layers[Qwen2Config::num_layers()];
    LayerNorm final_layernorm{Qwen2Config::hidden_size()}; // (hidden_size,)

    /**
     *
     * @param k_cache bf16 keys (seq_len, num_layers, num_kv_heads, key_size)
     * @param v_cache bf16 values (seq_len, num_layers, num_kv_heads, value_size)
     * @param seq_len current sequence length
     * @param input_tok_id last token in the sequence
     * @param temperature Sampling parameter. Always set to 0, for deterministic (greedy) decoding, see https://www.ibm.com/docs/en/watsonx/saas?topic=lab-model-parameters-prompting.
     *                    You do not need to implement any other sampling methods.
     * @return
     */
    int32_t forward(const std::shared_ptr<CudaBuffer> &k_cache, const std::shared_ptr<CudaBuffer> &v_cache, int32_t seq_len, int32_t input_tok_id, float temperature) {
        // TODO

        // get embedding for token input_tok_id
        checkCuda(cudaMemcpyAsync(
            hidden_state->data,
            static_cast<__nv_bfloat16*>(embedding_weight->data)
                + input_tok_id * Qwen2Config::hidden_size(),
            Qwen2Config::hidden_size() * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice,
            stream));

        // pass through transformer layers
        for (uint32_t layer_idx = 0; layer_idx < Qwen2Config::num_layers(); layer_idx++) {
            layers[layer_idx]->forward(
                k_cache,
                v_cache,
                hidden_state,
                seq_len,
                stream);
        }

        // final layernorm
        final_layernorm.normalize_hidden_state(hidden_state, final_hidden_state, stream);

        // convert from (hidden_size,) to the logits (vocab_size,)
        MatrixVectorMultiply::bf16_matmul<__nv_bfloat16>(
            Qwen2Config::vocab_size(),
            Qwen2Config::hidden_size(),
            static_cast<__nv_bfloat16*>(embedding_weight->data),
            nullptr,
            static_cast<__nv_bfloat16*>(final_hidden_state->data),
            static_cast<__nv_bfloat16*>(logits->data),
            stream);

        int32_t *new_token_ptr = argmax.bf16_argmax(logits, stream);
        checkCuda(cudaStreamSynchronize(stream));
        return *new_token_ptr;
    }
};