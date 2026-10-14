# Experimental Qwen3.6 on Apple Silicon

This fork adds a separate Metal decode engine for Qwen3.6-35B-A3B. Select it explicitly with `--family qwen36`;
Flash-Next keeps its existing engine and setup defaults. The server and browser chat are Strata's existing ones.

## Install and start

Apple Silicon, macOS 15 or newer, and Apple's Command Line Tools are required.

```sh
./setup.sh --family qwen36 --model UD-IQ2_XXS --context 4096 --vision no --yes --no-start
./run-qwen36-ud-iq2_xxs.sh
```

The IQ2_XXS download is 11.82 GB, pinned to Unsloth revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`
and checked against SHA-256 `627e2b05f83088448e27861ec392d56efc2b178783cae5adfa4ad2d988441203`.
This extra choice is Mac-only; the PC setup menu is unchanged. Q4 remains selectable (22.85 GB).
It needs no repacking, second experts copy or MTP download. Setup extracts only the tokenizer and builds
`build-metal/strata-q36-metal`. It needs CMake, Ninja and Apple clang, without metal-cpp or a CUDA compiler.
Open `http://127.0.0.1:8080` for chat. The OpenAI and Anthropic API routes are the same as other Strata models.

Q6 and Q8 are also selectable and have kernels, but only Q4 and IQ2_XXS are checked with real weights on this fork.
Use the internal SSD. The whole GGUF is mapped read-only; dense matrices are copied into shared Metal buffers,
and each layer copies only the selected experts into a bounded staging buffer. The rest are clean file pages
that macOS can reclaim. There is no full-expert host allocation or memory lock. SSD reads can dominate latency.

## Limits

- Experimental, text only. MTP, images and batched requests are not implemented.
- At most 4,096 tokens including prompt and answer. Prompt processing is one token at a time.
- GDN state width 128, attention head width 256, four convolution taps. Other layouts fail before inference.
- Tensor formats: F32, F16, BF16, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_S, IQ3_XXS and IQ4_XS.
  Other formats are rejected. Unsloth's IQ2_XXS file mixes these formats; it is not uniformly two-bit.
- Reference F32 activations. Answers can differ from the CUDA/HIP path's Q8 activation rounding.
- Expert selection happens on the CPU; projection matrices, GDN and attention run on Metal. This is a correctness
  baseline, not a port of the CUDA/HIP hybrid caching and prompt optimizations.
- Cancellation is checked between tokens, so a Stop may wait for the current token's layers to finish.

## Validation

On Apple M1 Pro (8 CPU cores, 14 GPU cores), 16 GB unified memory, macOS 27.0.1:

- Engine and all Metal kernels compile.
- 50 kernel checks pass: each weight format's matrix products, embedding rows, selected experts,
  three recurrent GDN steps with SiLU-gated norm, and grouped attention with RMS norm and RoPE.
- A synthetic four-layer model matches an independent NumPy forward implementation over three teacher-forced
  positions (relative logits L2 error `3.26e-7`). Prefix reuse, invalid token rejection, prompt cancellation and a
  subsequent fresh request pass through the actual engine pipe.
- 35 setup tests pass, including the existing Mac and Qwen3.6 PC setup tests.

The real Q4 GGUF's pinned SHA-256 passes. On the short Japanese prompt `2+2は？数字だけ答えて。`, the engine
answers `4` followed by the end-of-turn token. An independent CPU oracle linked against Strata's pinned
llama.cpp commit chooses the same next token at all eight teacher-forced positions (including positions after
end-of-turn). Mean KL across those positions is `0.00951`; the two response positions are below `5.4e-5`.
This is a short correctness check, not a broad quality evaluation.

In the first standalone run: loading took 7.0 s; 22 prompt tokens took 43.49 s (0.51 tok/s); eight fixed
CLI decode positions took 11.84 s (0.68 tok/s, seven recurrent forward steps after the first pick).
Metal buffers total 3.28 GiB including dense weights, selected-expert staging, state and a 4K KV cache.
Browser chat also passes on the real model: `2+2` gives `4`, a follow-up adding `3` gives `7`, and a new Japanese
chat returns `こんにちは。`. All three streaming requests return HTTP 200, end with `stop`, and produce no browser
JavaScript errors. Stopping a long prompt in the browser cancels it with zero generated tokens; a fresh
`1+1` request then returns `2`. With file pages warm, the 19-token greeting prompt takes 6.22 s and its three output tokens
0.93 s (the UI reports 3.2 tok/s; two recurrent forward steps). These short answers include the first pick from
prompt logits, so they are not a sustained-throughput benchmark.

That buffer figure excludes mapped file pages and other process/OS memory. The system used swap during the real-model
checks; this is not a claim that the model runs without swapping on a 16 GB desktop.

### IQ2_XXS real-model check

On the same 16 GB M1 Pro, the pinned 11.82 GB IQ2_XXS file passes SHA-256 verification and its 753-tensor layout check.
The arithmetic prompt above returns `4` and end-of-turn. All eight teacher-forced top-token choices match the
pinned llama.cpp CPU oracle, including six positions after end-of-turn. The two response positions have KL
`0.000101` and `0.000138`; mean KL over all eight positions is `0.05942` (the later after-end positions differ more).

The real browser chat also passes: `2+2` returns `4`, the follow-up adding `3` returns `7`, and a new greeting
returns `こんにちは。`. All three streaming requests return HTTP 200 with a normal stop, and there are no browser
JavaScript errors. The local deployment now uses IQ2_XXS; its Q4 weights, tokenizer and start configuration were
removed at the user's request (22.86 GB).

Standalone measurement: 2.9 s to load, 22 prompt tokens in 11.84 s (1.86 tok/s), eight fixed CLI decode positions
in 2.64 s (3.03 tok/s, seven recurrent steps). Metal buffers total 2.07 GiB. This is a short check, not a sustained
throughput or broad quality benchmark; file cache state matters, and mapped pages/OS memory are additional.

To repeat the checks after setup:

```sh
.venv/bin/ninja -C build-metal q36_parity
./build-metal/q36_parity
.venv/bin/python tools/test_q36_metal_model.py
.venv/bin/python -m unittest tools.test_setup_mac tools.test_setup_qwen36 tools.test_setup_mac_qwen36
```

The new CMake code is confined to `metal/`; CUDA, HIP and SYCL source files are unchanged. Those backends were not
built on this Mac.
