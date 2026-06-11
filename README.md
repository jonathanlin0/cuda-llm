# CUDA LLM

CUDA implementation of single-token autoregressive inference for Qwen2.5-0.5B-Instruct.

## CUDA modules
I wrote 10 custom kernels for this project. The following is a brief explanation of each of the kernels, and the associated CUDA files contain more information on implementation and performance.

### [`ArgMax.cu`](src/gpu_ops/ArgMax.cu)
Find the first index of the maximum value in an array.
- `bf16_argmax_partial_kernel`. The maximum value and corresponding index is found for each block.
- `argmax_final_kernel`. Finds the maximum value and corresponding index based on the maximums from each block.

### [`GroupQueryAttention.cuh`](src/gpu_ops/GroupQueryAttention.cuh)
Performs group query attention with flash attention.
- `group_query_attention_kernel`. Performs grouped query attention. A block is launched for each head, and each thread corresponds with a dimension in the head.

### [`LayerNorm.cu`](src/gpu_ops/LayerNorm.cu)
Performs layer normalization $y = \frac{\gamma_i x}{\sqrt{mean(x^2) + \epsilon}}$.
- `layernorm_partial_sum_kernel`. Computes each block's contribution to $mean(x^2)$
- `layernorm_scale_kernel`. Combines the sum of each block's contribution to $mean(x^2)$
- `layernorm_apply_kernel`. Applies the scaling: multiply by $\gamma_i$ weight and divide everything by $\sqrt{mean(x^2) + \epsilon}$

### [`MatrixVectorMultiply.cu`](src/gpu_ops/MatrixVectorMultiply.cu)
Does a matrix vector multiplication.
- `bf16_matmul_kernel`. Performs the matrix vector multiplication. One block for each output dimension.

### [`RoPE.cu`](src/gpu_ops/RoPE.cu)
Applies the Rotary Position Embedding to the embeddings.
- `apply_rope_to_qk_kernel`. Modifies the embeddings in place with the PEs.

### [`SiLUMult.cu`](src/gpu_ops/SiLUMult.cu)
Helps perform the SwiGLU operation.
- `silu_mult_kernel`. Performs SiLU(gate_proj(ffn_input)) $\otimes$ up_proj(ffn_input)

### [`Qwen2Layer.cuh`](src/qwen2/Qwen2Layer.cuh)
A single layer in the decoder-based LLM.
- `bf16_add_in_place_kernel`. In place vector additions used for the residual connections.

### [`Qwen2Model.cuh`](src/qwen2/Qwen2Model.cuh)
No custom kernels in this file. It's mainly LLM orchestration code: load embedding, run embeddings through transformer layers, convert logits to token distribution, etc.



## Linux Setup

Install the NVIDIA driver, CUDA toolkit, CMake 3.27 or newer, and a C++20-capable compiler.

Place a Hugging Face snapshot of `Qwen/Qwen2.5-0.5B-Instruct` on the server. The runtime expects the model directory to contain:

```text
config.json
model.safetensors
tokenizer.json
```

If the model is not located at the default `/cs179/Qwen2.5-0.5B-Instruct`, set:

```bash
export TRANSFORMER_MODEL_DIR=/path/to/Qwen2.5-0.5B-Instruct
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target transformer
```

## Run

Generate the default 100-token autoregressive test output:

```bash
./build/transformer
```

Run the chatbot interface:

```bash
./build/transformer --interactive --max-seq-len 10000
```

Useful flags:

```text
--max-seq-len N       Maximum sequence length, default 100
--interactive         Read user prompts from stdin
--system-prompt TEXT  System prompt for interactive mode
```

The executable only supports Qwen2/Qwen2.5 0.5B-style BF16 safetensors with the expected tensor names and shapes.

## Test cases
Run all test cases with:
- `cd build`
- `cmake --build .`
- `ctest`

For tests that are failing, you can run them individually to see which elements were incorrect, for example:
- `cd build`
- `cmake --build .`
- `./silumulttest`

Failing output example:
```
difference at index 0: GPU calculated 10.875, CPU calculated 108.5
difference at index 1: GPU calculated -5.65625, CPU calculated 0.332031
difference at index 2: GPU calculated 7.3125, CPU calculated -76.5
...
```

## Acknowledgements

The non-CUDA scaffolding code, such as loading in the model weights and the tests, were provided by [Caltech's CS179's instructors](https://courses.cms.caltech.edu/cs179/).
