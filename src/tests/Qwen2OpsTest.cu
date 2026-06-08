#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "../CudaBuffer.cuh"
#include "../ErrorCheck.h"
#include "../gpu_ops/ArgMax.cuh"
#include "../gpu_ops/LayerNorm.cuh"
#include "../gpu_ops/MatrixVectorMultiply.cuh"
#include "../gpu_ops/RoPE.cuh"
#include "../gpu_ops/SiLUMult.cuh"
#include "../qwen2/Qwen2Config.h"
#include "TestUtils.cuh"

namespace {

using Config = Qwen2Config<QWEN2_0_5B>;

float bf16_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

void fill_bf16(__nv_bfloat16 *data, int64_t len, std::mt19937 &generator, float stddev = 1.0f) {
    std::normal_distribution<float> distribution(0.0f, stddev);
    for (int64_t i = 0; i < len; i++) {
        data[i] = __float2bfloat16(distribution(generator));
    }
}

void fill_layernorm_weights(__nv_bfloat16 *data, int64_t len, std::mt19937 &generator) {
    std::normal_distribution<float> distribution(1.0f, 0.05f);
    for (int64_t i = 0; i < len; i++) {
        data[i] = __float2bfloat16(distribution(generator));
    }
}

void prefetch_to_device(cudaStream_t stream, const std::vector<std::shared_ptr<CudaBuffer>> &buffers) {
    int device = 0;
    checkCuda(cudaGetDevice(&device));
    for (const auto &buffer : buffers) {
        checkCuda(cudaMemPrefetchAsync(buffer->data, buffer->size, device, stream));
    }
    checkCuda(cudaStreamSynchronize(stream));
}

void prefetch_to_cpu(cudaStream_t stream, const std::shared_ptr<CudaBuffer> &buffer) {
    checkCuda(cudaMemPrefetchAsync(buffer->data, buffer->size, cudaCpuDeviceId, stream));
    checkCuda(cudaStreamSynchronize(stream));
}

void test_argmax() {
    constexpr int32_t len = Config::vocab_size();
    constexpr int32_t expected_index = len / 3;

    auto values = std::make_shared<CudaBuffer>(len * sizeof(__nv_bfloat16));
    auto *values_bf16 = static_cast<__nv_bfloat16*>(values->data);

    std::mt19937 generator{123};
    fill_bf16(values_bf16, len, generator, 2.0f);

    values_bf16[expected_index] = __float2bfloat16(42.0f);
    values_bf16[expected_index + 7] = __float2bfloat16(42.0f);

    prefetch_to_device(cudaStreamPerThread, {values});

    ArgMax argmax(len);
    int32_t *calculated_index_ptr = argmax.bf16_argmax(values, cudaStreamPerThread);
    checkCuda(cudaStreamSynchronize(cudaStreamPerThread));

    int32_t calculated_index = *calculated_index_ptr;
    if (calculated_index != expected_index) {
        std::cerr << "ArgMax mismatch: got index " << calculated_index
            << ", expected index " << expected_index << std::endl;
        std::exit(1);
    }

    std::cout << "ArgMax passed with vocab_size=" << len << std::endl;
}

void test_layernorm() {
    constexpr int32_t hidden_size = Config::hidden_size();

    auto weights = std::make_shared<CudaBuffer>(hidden_size * sizeof(__nv_bfloat16));
    auto input = std::make_shared<CudaBuffer>(hidden_size * sizeof(__nv_bfloat16));
    auto output = std::make_shared<CudaBuffer>(hidden_size * sizeof(__nv_bfloat16));

    auto *weights_bf16 = static_cast<__nv_bfloat16*>(weights->data);
    auto *input_bf16 = static_cast<__nv_bfloat16*>(input->data);
    auto *output_bf16 = static_cast<__nv_bfloat16*>(output->data);

    std::mt19937 generator{456};
    fill_layernorm_weights(weights_bf16, hidden_size, generator);
    fill_bf16(input_bf16, hidden_size, generator);

    float variance_sum = 0.0f;
    for (int32_t i = 0; i < hidden_size; i++) {
        float value = bf16_to_float(input_bf16[i]);
        variance_sum += value * value;
        output_bf16[i] = __float2bfloat16(-1.0f);
    }

    float rms = std::sqrt(variance_sum / static_cast<float>(hidden_size) + LayerNorm::EPS);
    std::vector<__nv_bfloat16> expected(hidden_size);
    for (int32_t i = 0; i < hidden_size; i++) {
        expected[i] = __float2bfloat16(bf16_to_float(weights_bf16[i]) * bf16_to_float(input_bf16[i]) / rms);
    }

    prefetch_to_device(cudaStreamPerThread, {weights, input, output});

    LayerNorm layer_norm(hidden_size);
    layer_norm.weights = weights;
    layer_norm.normalize_hidden_state(input, output, cudaStreamPerThread);
    checkCuda(cudaStreamSynchronize(cudaStreamPerThread));

    prefetch_to_cpu(cudaStreamPerThread, output);
    check_bf16_allclose(output_bf16, expected.data(), hidden_size);

    std::cout << "LayerNorm passed with hidden_size=" << hidden_size << std::endl;
}

template<typename input_float_t>
void test_matvec_case(const std::string &label, int32_t m, int32_t k, bool use_bias, bool input_is_float) {
    auto matrix = std::make_shared<CudaBuffer>(static_cast<size_t>(m) * k * sizeof(__nv_bfloat16));
    std::shared_ptr<CudaBuffer> bias{};
    if (use_bias) {
        bias = std::make_shared<CudaBuffer>(m * sizeof(__nv_bfloat16));
    }
    auto input = std::make_shared<CudaBuffer>(k * sizeof(input_float_t));
    auto output = std::make_shared<CudaBuffer>(m * sizeof(__nv_bfloat16));

    auto *matrix_bf16 = static_cast<__nv_bfloat16*>(matrix->data);
    auto *bias_bf16 = bias ? static_cast<__nv_bfloat16*>(bias->data) : nullptr;
    auto *input_values = static_cast<input_float_t*>(input->data);
    auto *output_bf16 = static_cast<__nv_bfloat16*>(output->data);

    std::mt19937 generator{static_cast<uint32_t>(789 + m + k + (input_is_float ? 17 : 0))};
    fill_bf16(matrix_bf16, static_cast<int64_t>(m) * k, generator, 0.02f);
    if constexpr (std::is_same_v<input_float_t, float>) {
        fill_float(input_values, k, generator);
    } else {
        fill_bf16(input_values, k, generator);
    }
    if (bias_bf16 != nullptr) {
        fill_bf16(bias_bf16, m, generator, 0.02f);
    }

    std::vector<__nv_bfloat16> expected(m);
    for (int32_t row = 0; row < m; row++) {
        float sum = bias_bf16 == nullptr ? 0.0f : bf16_to_float(bias_bf16[row]);
        for (int32_t col = 0; col < k; col++) {
            float input_value;
            if constexpr (std::is_same_v<input_float_t, float>) {
                input_value = input_values[col];
            } else {
                input_value = bf16_to_float(input_values[col]);
            }
            sum += bf16_to_float(matrix_bf16[row * k + col]) * input_value;
        }
        expected[row] = __float2bfloat16(sum);
        output_bf16[row] = __float2bfloat16(-1.0f);
    }

    std::vector<std::shared_ptr<CudaBuffer>> buffers{matrix, input, output};
    if (bias != nullptr) {
        buffers.push_back(bias);
    }
    prefetch_to_device(cudaStreamPerThread, buffers);

    MatrixVectorMultiply::bf16_matmul<input_float_t>(
        m, k, matrix_bf16, bias_bf16, input_values, output_bf16, cudaStreamPerThread);
    checkCuda(cudaStreamSynchronize(cudaStreamPerThread));

    prefetch_to_cpu(cudaStreamPerThread, output);
    check_bf16_allclose(output_bf16, expected.data(), m);

    std::cout << label << " passed with matrix shape (" << m << ", " << k << ")" << std::endl;
}

void test_matrix_vector_multiply() {
    test_matvec_case<__nv_bfloat16>(
        "MatVec FFN up projection",
        Config::intermediate_size(),
        Config::hidden_size(),
        false,
        false);
}

void test_rope_case(const std::string &label, int32_t num_heads, int32_t position_idx) {
    constexpr int32_t head_dim = Config::head_size();
    constexpr float theta_base = Config::rope_theta_base();
    static_assert(head_dim % 2 == 0);

    int32_t len = num_heads * head_dim;
    auto values = std::make_shared<CudaBuffer>(len * sizeof(__nv_bfloat16));
    auto *values_bf16 = static_cast<__nv_bfloat16*>(values->data);

    std::mt19937 generator{static_cast<uint32_t>(321 + num_heads)};
    fill_bf16(values_bf16, len, generator);

    std::vector<float> cos_vals(head_dim / 2);
    std::vector<float> sin_vals(head_dim / 2);
    for (int32_t theta_idx = 0; theta_idx < head_dim / 2; theta_idx++) {
        float theta_idx_frac = static_cast<float>(theta_idx) / static_cast<float>(head_dim / 2);
        float theta = std::pow(theta_base, -theta_idx_frac);
        float angle = theta * static_cast<float>(position_idx);
        cos_vals[theta_idx] = std::cos(angle);
        sin_vals[theta_idx] = std::sin(angle);
    }

    std::vector<__nv_bfloat16> expected(len);
    for (int32_t head = 0; head < num_heads; head++) {
        auto *input_row = values_bf16 + head * head_dim;
        for (int32_t i = 0; i < head_dim; i++) {
            int32_t theta_idx = i % (head_dim / 2);
            int32_t rotated_idx = i < head_dim / 2 ? i + head_dim / 2 : i - head_dim / 2;
            float rotated_value = i < head_dim / 2
                ? -bf16_to_float(input_row[rotated_idx])
                : bf16_to_float(input_row[rotated_idx]);
            float result = bf16_to_float(input_row[i]) * cos_vals[theta_idx] + rotated_value * sin_vals[theta_idx];
            expected[head * head_dim + i] = __float2bfloat16(result);
        }
    }

    prefetch_to_device(cudaStreamPerThread, {values});

    RoPE::apply_rope_to_qk(values_bf16, num_heads, head_dim, position_idx, theta_base, cudaStreamPerThread);
    checkCuda(cudaStreamSynchronize(cudaStreamPerThread));

    prefetch_to_cpu(cudaStreamPerThread, values);
    check_bf16_allclose(values_bf16, expected.data(), len);

    std::cout << label << " passed with shape (" << num_heads << ", " << head_dim << ")" << std::endl;
}

void test_rope() {
    constexpr int32_t position_idx = 1233;
    test_rope_case("RoPE queries", Config::num_query_heads(), position_idx);
}

void test_silu_mult() {
    constexpr int32_t len = Config::intermediate_size();

    auto x = std::make_shared<CudaBuffer>(len * sizeof(__nv_bfloat16));
    auto y = std::make_shared<CudaBuffer>(len * sizeof(__nv_bfloat16));
    auto *x_bf16 = static_cast<__nv_bfloat16*>(x->data);
    auto *y_bf16 = static_cast<__nv_bfloat16*>(y->data);

    std::mt19937 generator{654};
    fill_bf16(x_bf16, len, generator, 2.0f);
    fill_bf16(y_bf16, len, generator, 2.0f);

    std::vector<__nv_bfloat16> expected(len);
    for (int32_t i = 0; i < len; i++) {
        float x_value = bf16_to_float(x_bf16[i]);
        float y_value = bf16_to_float(y_bf16[i]);
        expected[i] = __float2bfloat16(x_value / (1.0f + std::exp(-x_value)) * y_value);
    }

    prefetch_to_device(cudaStreamPerThread, {x, y});

    SiLUMult::silu_mult_in_place(x, y, cudaStreamPerThread);
    checkCuda(cudaStreamSynchronize(cudaStreamPerThread));

    prefetch_to_cpu(cudaStreamPerThread, x);
    check_bf16_allclose(x_bf16, expected.data(), len);

    std::cout << "SiLUMult passed with intermediate_size=" << len << std::endl;
}

}

int main() {
    std::cout << "Testing kernels with Qwen2 0.5B dimensions: hidden_size="
        << Config::hidden_size()
        << ", query_heads=" << Config::num_query_heads()
        << ", kv_heads=" << Config::num_kv_heads()
        << ", head_size=" << Config::head_size()
        << ", intermediate_size=" << Config::intermediate_size()
        << ", vocab_size=" << Config::vocab_size()
        << std::endl;

    test_argmax();
    test_layernorm();
    test_matrix_vector_multiply();
    test_rope();
    test_silu_mult();

    checkCuda(cudaDeviceSynchronize());
    std::cout << "All Qwen2 operator tests passed" << std::endl;
}
