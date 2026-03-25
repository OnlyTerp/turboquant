# TurboQuant KV Cache Compression — Deep Research Report

_Compiled 2026-03-24 | 40+ searches across 6 categories_

## Executive Summary

TurboQuant is a Google Research project (arxiv 2504.19874, ICLR 2026) that combines **PolarQuant** (KV cache quantization via polar transformation, arxiv 2502.02617) with **QJL** (1-bit Quantized Johnson-Lindenstrauss transform for residual error) to achieve 3.5-bit KV cache quantization with **zero accuracy loss** on LongBench. **No open-source implementation of TurboQuant itself exists**, but QJL has a reference CUDA/PyTorch repo. We must build the Triton kernels ourselves.

---

## Category 1: TurboQuant / PolarQuant / QJL Specific

### 1.1 TurboQuant Paper (arxiv 2504.19874)
- **URL**: https://arxiv.org/html/2504.19874
- **Authors**: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research/DeepMind)
- **Key insight**: Two-stage quantization:
  1. **Stage 1 — PolarQuant (MSE-optimal)**: Random rotation of input vectors → each coordinate follows a Beta distribution → apply optimal Lloyd-Max scalar quantizer per coordinate. Most of the bit budget (e.g., 2.5-3.5 bits) goes here.
  2. **Stage 2 — QJL (1-bit residual)**: Apply 1-bit QJL transform to the residual from Stage 1 to produce an **unbiased inner product estimator**. Only 1 bit per coordinate for the residual.
- **Key algorithm details**:
  - Random rotation: multiply input vector by a random orthogonal matrix (can use randomized Hadamard for speed)
  - Post-rotation coordinates follow concentrated Beta distribution → coordinates are "near-independent" in high dimensions
  - Lloyd-Max quantizer codebooks for each coordinate are computed analytically for the Beta distribution — **NOT iterative/learned**
  - LUT (lookup table) for query-key product: replaces Q×K inner product with table lookup, avoiding RoPE recomputation
- **Performance claims**:
  - 3.5 bits/channel = absolute quality neutrality on LongBench
  - 2.5 bits/channel = marginal quality degradation
  - Up to 6x KV memory reduction on Llama-3.1-8B-Instruct
  - 4-bit TurboQuant = **8x performance over 32-bit** on H100 (attention logits computation)
  - Mixed-precision fused Triton kernel = 2-4x faster than FP32 GEMM
- **How it helps**: This is our target algorithm. The paper describes the mathematical framework but NOT implementation code.
- **⚠️ WARNING**: Paper describes custom Triton kernels but provides no source code. We must implement from scratch.

### 1.2 PolarQuant Paper (arxiv 2502.02617)
- **URL**: https://arxiv.org/html/2502.02617v1
- **Authors**: Insu Han (KAIST), Praneeth Kacham, Amin Karbasi, Vahab Mirrokni, Amir Zandieh
- **Key insight**: 
  - Divides key vectors into groups of 2D sub-vectors
  - Encodes as (radius, angle) in polar coordinates
  - Random preconditioning concentrates the angle distribution → no normalization needed
  - **Eliminates memory overhead** of scale/zero-point storage per block (saves 1-2 bits per value)
- **Recursive polar transformation**: Converts d-dimensional vector → polar coords recursively
  - For a 2D sub-vector (x₁, x₂): r = √(x₁² + x₂²), θ = atan2(x₂, x₁)
  - Recursive: pair up polar coords → further polar transform
  - Only angles are quantized; radii are reconstructed from quantized angles + preserved norm
- **Key practical implementation details** (Section 4.1):
  - Random preconditioning uses a random Hadamard matrix (not full random matrix) for efficiency
  - Angle quantization codebook is derived from the known concentrated distribution
  - Value cache quantized separately (typically at higher precision)
- **Performance**: 4.2x compression with best quality scores on long-context tasks
- **How it helps**: This is the "Stage 1" of TurboQuant. The polar transform + angle quantization replaces naive coordinate quantization.

### 1.3 QJL — 1-bit Quantized JL Transform (arxiv 2406.03482)
- **URL**: https://arxiv.org/abs/2406.03482
- **Code**: https://github.com/amirzandieh/QJL
- **Authors**: Amir Zandieh, Majid Daliri, Insu Han
- **Key insight**: 
  - Apply Johnson-Lindenstrauss transform as preconditioner → then quantize to single sign bit (+1/-1)
  - **Zero memory overhead** — no scales/zero-points needed
  - Produces **unbiased** inner product estimator
- **Reference implementation details**:
  - Has custom CUDA kernels in `qjl_kernel/` directory
  - `setup.py build_ext --inplace` to compile
  - Supports Llama 2/3 with configurable: `key_quantization_bits` (256 default), `value_quantization_bits` (2), `group_size` (32), `buffer_size` (128)
  - Key quantization uses more bits in initial layers: `key_quantization_bits_initial_layers` (512), `initial_layers_count` (15)
  - Outlier handling: `outlier_count_general` (8), `outlier_count_initial_layers` (8)
- **How it helps**: This is the "Stage 2" of TurboQuant. The reference CUDA code can serve as a starting point for understanding the algorithm, though we need Triton ports.
- **⚠️ WARNING**: The QJL repo is standalone — it patches HuggingFace transformers, NOT vLLM. We need vLLM integration.

### 1.4 Google Research Blog Post
- **URL**: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- **Key insight**: TurboQuant was tested on Gemma and Mistral models. Key claim: "4-bit TurboQuant achieves up to 8x performance increase over 32-bit unquantized keys on H100 accelerators when computing attention logits."
- **How it helps**: Confirms the LUT-based attention computation approach — the speedup comes from replacing FP32 Q×K matrix multiply with integer table lookups.

### 1.5 Open Source Status
- **Finding**: NO open-source TurboQuant implementation exists
- **QJL**: Has standalone CUDA implementation at https://github.com/amirzandieh/QJL
- **PolarQuant**: NO open-source implementation found
- **TurboQuant**: The GitHub repo "cg94301/turboquant" is an unrelated trading strategy project
- **⚠️ CRITICAL**: We are building this from scratch. This is a significant engineering effort.

---

## Category 2: KV Cache Quantization Implementations (What Exists)

### 2.1 vLLM FP8 KV Cache Quantization
- **URL**: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- **Source code**: `vllm/model_executor/layers/quantization/kv_cache.py`
- **Key insight**: 
  - Uses `BaseKVCacheMethod` class extending `QuantizeMethodBase`
  - Adds `_k_scale`, `_v_scale`, `_q_scale` as `torch.nn.Parameter` (initialized to -1.0)
  - KV cache dtype set via `kv_cache_dtype="fp8"` in `CacheConfig`
  - Three calibration modes: no calibration (scales=1.0), random token calibration (`calculate_kv_scales=True`), dataset calibration (via `llm-compressor`)
  - Dequantization happens on retrieval before attention computation
- **Key code**:
  ```python
  class BaseKVCacheMethod(QuantizeMethodBase):
      def create_weights(self, layer: torch.nn.Module):
          layer.q_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
          layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
          layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
  ```
- **How it helps**: Shows the exact vLLM extension point for quantized KV cache. We can extend `BaseKVCacheMethod` or create our own `QuantizeMethodBase` subclass for TurboQuant.
- **⚠️ WARNING**: vLLM's FP8 quantization is a simple scale-based approach. TurboQuant's polar transformation + QJL is fundamentally different and may require deeper integration than just extending `BaseKVCacheMethod`.

### 2.2 KIVI — 2-bit Asymmetric KV Cache Quantization (ICML 2024)
- **URL**: https://github.com/jy-yuan/KIVI
- **Key insight**: 
  - Per-channel quantization for keys, per-token for values (asymmetric approach)
  - 2/4 bit support with configurable group sizes
  - Has custom CUDA kernels in `quant/` directory
  - Includes `residual_length` parameter — keeps the most recent N tokens in FP16
  - Inspired HuggingFace Transformers KV cache quantization
  - Now supports GQA and transformers 4.43+
- **Architecture pattern**:
  - `models/llama_kivi.py` — monkey-patches the model
  - `quant/` — custom CUDA quantize/dequantize kernels
  - Config params: `k_bits`, `v_bits`, `group_size`, `residual_length`
- **How it helps**: Demonstrates the "recent tokens in FP16" pattern (useful for TurboQuant too). Shows how to integrate custom CUDA kernels with HuggingFace models. The asymmetric K/V approach is a useful reference.

### 2.3 KVQuant — Per-Channel Pre-RoPE Quantization
- **URL**: https://github.com/SqueezeAILab/KVQuant
- **Key insight**: 
  - Per-channel quantization for keys (before RoPE)
  - Non-uniform quantization (NUQ) for better representation
  - Dense-and-sparse quantization for outlier handling
- **How it helps**: Shows that pre-RoPE key quantization is important for quality. Relevant to our PolarQuant implementation since the polar transform should also be applied pre-RoPE.

### 2.4 llama.cpp KV Cache Quantization
- **URL**: https://github.com/ggml-org/llama.cpp/discussions/5932
- **Key insight**: 
  - Supports Q4_0 and Q4_1 KV cache formats
  - Per-block quantization with GGUF format
  - Quality: Q4 KV cache can be comparable to full precision in some cases
  - K-cache quantization may affect quality more than V-cache quantization
- **How it helps**: Validates that 4-bit KV cache is practical. The K vs V sensitivity difference aligns with TurboQuant's approach of using more bits for keys.

---

## Category 3: Triton Kernel Patterns (How to Build It)

### 3.1 Triton Fast Walsh-Hadamard Transform (FWHT)
- **URL**: https://github.com/arthurfeeney/fwht
- **Status**: WIP Triton implementation of FWHT for PyTorch
- **Key insight**:
  - Hadamard matrices expressed as Kronecker products: H_pq = kron(H_p, H_q)
  - Constructed via bit-wise dot product: (H_n)_{i,j} = (-1)^(i · j) where · is bitwise dot product
  - Requires input sizes factorable as 16^m * 2^n (or gets padded)
  - Max supported size: 16^3 * 2^3 (fits in shared memory)
  - Zero-pads inside kernel to next power of 2 (no extra global memory allocation)
- **Code structure**:
  ```python
  # Key pattern for Hadamard base case in Triton
  # (H_n)_{i,j} = (-1)^(i ⋅ j) — bitwise dot product
  # Triton should optimize most of this during compilation
  ```
- **How it helps**: Direct reference for implementing the random Hadamard preconditioning step in TurboQuant/PolarQuant. The Kronecker decomposition approach is the right one for Triton.
- **⚠️ WARNING**: This is WIP and has limitations. Max size 16^3 * 2^3 may not be sufficient for all head dimensions.

### 3.2 HadaCore — Tensor Core Accelerated Hadamard Transform
- **URL**: https://pytorch.org/blog/hadacore/ and https://github.com/pytorch-labs/applied-ai/tree/main/kernels/cuda/inference/hadamard_transform
- **Key insight**:
  - Hardware-aware work decomposition to leverage NVIDIA Tensor Cores
  - Applies 16×16 Hadamard transform to chunks → offloads to FP16 Tensor Core via `mma.m16n8k16` PTX instructions
  - 1.1–1.4x speedup on A100, 1.0–1.3x on H100 over Dao AI Lab's Fast Hadamard kernel
  - Peak gain: 3.5x on A100, 3.6x on H100
  - Processes 256-element fragments in parallel using warp-level Tensor Core operations
  - **Future work mentioned**: "plan to implement a Triton version of our kernel and experiment with kernel fusion to support fused Hadamard transform and quantization"
- **MMLU validation**: Using HadaCore with FP8 FlashAttention on Llama3.1-8B: 65.09 vs 64.40 without Hadamard (vs 65.38 FP16 baseline)
- **How it helps**: Proves that Hadamard transforms preserve quantization quality. The CUDA kernel reference can guide our Triton implementation. The "fused Hadamard + quantization" future work aligns exactly with TurboQuant's needs.

### 3.3 Triton Int4 Quantize/Dequantize Patterns
- **URL**: Multiple sources including PyTorch blog and community implementations
- **Key insight**:
  - Triton doesn't natively support int4 — must pack/unpack manually
  - Pack: two int4 values → one int8 byte: `(val2 << 4) | (val1 & 0xF)`
  - Unpack: `val1 = byte & 0xF`, `val2 = (byte >> 4) & 0xF`
  - Scale/zero-point: `(int_val - zero_point) * scale` for dequantization
  - `tl.where` for conditional selection during unpacking
  - BLOCK_SIZE must be even for pair packing
  - Grid adjustment: `grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * 2),)` for packing
- **How it helps**: The bit-packing patterns are essential for implementing TurboQuant's low-bit quantization in Triton. The code patterns for packing/unpacking are directly applicable.

### 3.4 Triton Philox PRNG
- **URL**: https://triton-lang.org/main/python-api/generated/triton.language.rand.html
- **Key insight**:
  - `tl.rand(seed, offset)` → uniform float32 in [0, 1)
  - `tl.randint(seed, offset)` → random int32
  - `tl.randint4x(seed, offsets)` → 4 blocks of random int32 (most efficient)
  - `tl.randn` → normal distribution
  - Counter-based PRNG: seed + offset → deterministic random numbers
  - Each thread can independently compute its random values using seed + global_offset
- **How it helps**: Essential for implementing the random preconditioning step. We can generate the random Hadamard sign pattern on-the-fly using `tl.rand` or precompute and cache it.
- **⚠️ WARNING**: For reproducibility across runs, use fixed seeds. The random preconditioning matrix must be consistent across quantize/dequantize operations.

### 3.5 Triton FlashAttention Reference
- **URL**: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- **Key insight**:
  - Standard Triton FlashAttention uses block-wise tiling of Q, K, V
  - Online softmax for numerical stability
  - Kernel fusion: matmul + softmax + matmul in single kernel
  - Streaming K and V blocks
- **How it helps**: Our TurboQuant attention kernel must integrate into this pattern. The key modification: instead of loading FP16 K blocks, we load quantized K blocks and dequantize on-the-fly before Q×K computation (or use the LUT approach).

### 3.6 Triton Fused Dequantization + GEMM
- **URL**: Multiple sources including lweitkamp.github.io/posts/qlora_dequantize/
- **Key insight**:
  - Fused dequant + GEMM avoids materializing intermediate dequantized weights
  - 65% speedup on A100, 124% on H100 for W4A16
  - Key pattern: load packed weights → unpack on-chip → immediately use in dot product
  - NF4 uses hardcoded 16-element lookup table with `tl.where` for efficient LUT
  - bitsandbytes `nf4_lut` function uses bitwise ops + conditional statements
- **How it helps**: The LUT pattern from NF4 dequantization is directly analogous to TurboQuant's LUT-based Q×K computation. We can adapt this pattern.

### 3.7 Triton Debugging & Pitfalls
- **URL**: https://triton-lang.org/main/programming-guide/chapter-3/debugging.html
- **Key pitfalls**:
  1. **Missing masks**: Always set `mask` and `other` parameters in `tl.load` for boundary conditions
  2. **Uncoalesced access**: Use `tl.arange` for contiguous offsets
  3. **Race conditions**: Multiple blocks writing to same memory — use `tl.atomic_*` for shared memory
  4. **1D blocks**: Triton blocks are 1D — reshape multi-dimensional problems
  5. **JIT compilation**: First call is slow (includes compilation) — always warm up
  6. **Power-of-2 blocks**: Triton requires block sizes to be powers of 2
- **Debugging tools**:
  - `TRITON_INTERPRET=1` — run on CPU with NumPy for debugging
  - `device_print` / `device_assert` — runtime debugging
  - `compute-sanitizer` — NVIDIA tool for checking data races
  - `TRITON_PRINT_AUTOTUNING=1` — see autotuner selections
- **How it helps**: Essential for avoiding bugs in our Triton kernels. The power-of-2 constraint affects how we handle non-power-of-2 head dimensions.

---

## Category 4: vLLM Internals (Where to Hook In)

### 4.1 vLLM Attention Backend Architecture
- **URL**: https://docs.vllm.ai/en/latest/design/attention_backends/
- **Source code**: `vllm/v1/attention/backends/flash_attn.py`
- **Key insight**:
  - "Switchboard" pattern: `ForwardContext` routes calls to correct backend based on `layer_name`
  - Attention backends are selected globally via `--attention-backend` or `AttentionConfig`
  - Cannot set different backends for prefill vs decode independently
  - Backend class must inherit from `AttentionBackend` (in `vllm.v1.attention.backend`)
- **Key source paths**:
  - `vllm/v1/attention/backends/flash_attn.py` — FlashAttentionBackend
  - `vllm/attention/layer.py` — core Attention class (stores K/V, performs attention)
  - `vllm/config/attention.py` — AttentionConfig class
  - `vllm/config/cache.py` — CacheConfig with `cache_dtype` parameter
- **How it helps**: Shows exactly where to plug in. We need to create a custom attention backend or modify the existing one to support TurboQuant quantized KV cache.

### 4.2 vLLM Plugin System (RECOMMENDED APPROACH)
- **URL**: https://docs.vllm.ai/en/latest/design/plugin_system/
- **Key insight**:
  - Uses Python `entry_points` mechanism
  - Plugin groups: `vllm.general_plugins`, `vllm.platform_plugins`, `vllm.io_processor_plugins`
  - **Platform plugins** are the right approach for custom attention backends
  - Entry point: define in `setup.py` under `vllm.platform_plugins`
  - Platform class inherits from `vllm.platforms.interface.Platform`
  - Must implement: `get_attn_backend_cls()`, `check_and_update_config()`, `device_type`
  - Worker class inherits from `WorkerBase`
  - Attention backend inherits from `AttentionBackend`
- **Plugin structure**:
  ```
  vllm_turboquant_plugin/
  ├── vllm_turboquant_plugin/
  │   ├── __init__.py
  │   ├── turboquant_platform.py
  │   ├── turboquant_worker.py
  │   ├── turboquant_attention.py  # Custom attention backend
  │   └── turboquant_ops.py
  └── setup.py
  ```
- **How it helps**: This is the OFFICIAL way to add custom attention backends to vLLM. Avoids monkey-patching issues.
- **⚠️ WARNING**: Plugin system may have limitations for complex modifications. The "CustomOp" system (`vllm.model_executor.custom_op.CustomOp`) provides another extension point.

### 4.3 Monkey Patching in vLLM (AVOID IF POSSIBLE)
- **URL**: https://discuss.vllm.ai/t/how-to-monkey-patch-vllm-correctly/1871
- **Key insight**:
  - vLLM uses multiprocessing → patches in main process don't propagate to workers
  - Must place patch code inside worker process (e.g., top of `vllm/v1/worker/gpu_worker.py`)
  - Maintenance nightmare: copying large source sections for minor changes
  - **vLLM officially discourages monkey patching** — use plugin system instead
- **How it helps**: Shows what NOT to do. Use the plugin system.

### 4.4 vLLM KV Cache Architecture (PagedAttention)
- **Key insight**:
  - KV cache divided into fixed-size blocks (typically 16 tokens)
  - Block table maps logical blocks → physical GPU memory locations
  - Non-contiguous physical storage → no memory fragmentation
  - Hash-based management supports prefix caching
  - `KVCacheManager` handles allocation, eviction, sharing
- **How it helps**: TurboQuant quantization must operate within the paged KV cache structure. Each page stores quantized KV data. The dequantization/LUT lookup happens at attention computation time, not at page load time.

### 4.5 vLLM KV Connector API (V1)
- **URL**: https://docs.vllm.ai/en/latest/api/vllm/distributed/kv_transfer/kv_connector/v1/example_connector/
- **Key methods**:
  - `build_connector_meta()` — track requests
  - `start_load_kv()` — initiate KV load
  - `wait_for_layer_load()` — check layer load
  - `save_kv_layer()` — save KV from paged buffer
  - `update_state_after_alloc()` — update after buffer alloc
- **How it helps**: If we want to save/load TurboQuant-quantized KV cache to disk or CPU memory, the KVConnector is the right API.

---

## Category 5: Related Compression Techniques

### 5.1 Lloyd-Max Quantizer
- **Key insight**:
  - Iterative algorithm: alternate between (1) update decision boundaries (midpoints) and (2) update reconstruction levels (centroids)
  - For TurboQuant: since post-rotation coordinates follow a Beta distribution, the Lloyd-Max codebook can be computed **analytically** (no iteration needed)
  - Codebook entries and boundaries are precomputed constants that can be stored in a lookup table
- **Python reference**: Basic implementation iterates until convergence with tolerance ~1e-6
- **How it helps**: Understanding the Lloyd-Max algorithm is essential for implementing the coordinate-wise quantizer in TurboQuant. The key optimization: since we know the distribution analytically (Beta), we precompute the codebook once offline and store it as a constant.

### 5.2 Randomized Hadamard Transform for Quantization
- **URL**: https://arxiv.org/html/2501.02625v1 and multiple related papers
- **Key insight**:
  - RHT "incoherentizes" matrices → makes entries more uniformly distributed (approx i.i.d. Gaussian)
  - Reduces outlier magnitudes → enables aggressive quantization
  - Projects like QuIP# use RHT for 2-4 bit quantization
  - Fast Walsh-Hadamard Transform (FWHT) minimizes runtime overhead of applying rotation
- **How it helps**: Confirms that randomized Hadamard is the right choice for TurboQuant's random rotation step (vs full random matrix which would be O(d²)).

### 5.3 QuaRot and SpinQuant
- **URL**: https://arxiv.org/html/2404.00456v1 (QuaRot), https://arxiv.org/abs/2405.16406 (SpinQuant)
- **Key insight**:
  - Four rotation types: R1, R2 (offline/fused into weights), R3 (online for KV cache), R4 (online for activations)
  - R3 rotation is directly relevant to our KV cache quantization
  - QuaRot uses fixed Hadamard matrices; SpinQuant learns optimized orthogonal matrices
  - Both require Fast Hadamard Transform kernel support
  - Can combine with GPTQ for weight quantization
- **How it helps**: The R3 rotation in QuaRot is conceptually similar to TurboQuant's random preconditioning for KV cache. Their implementation patterns (online rotation + quantization) are a direct reference.

### 5.4 REAP Expert Pruning (MoE)
- **URL**: https://github.com/CerebrasResearch/reap
- **Key insight**: Router-weighted expert activation pruning for MoE models. Complementary to KV cache compression — reduces static model size while KV cache compression reduces dynamic memory.
- **How it helps**: Not directly related to TurboQuant implementation, but shows the broader optimization landscape for MoE models.

---

## Category 6: Gotchas and Failure Modes

### 6.1 KV Cache Quantization Accuracy Loss
- **Key insight**:
  - **Key vectors are more sensitive** than Value vectors to precision loss
  - **Outlier tokens** produce Key vectors with extreme magnitudes → skew quantization ranges
  - **Attention sink tokens** (first few tokens) are critical — must preserve precision
  - **Pre-RoPE quantization** reduces accuracy impact
  - Mixed precision (K at higher precision than V) is effective
  - FP8 shows minimal accuracy impact; INT4/INT2 more aggressive
- **How it helps**: TurboQuant's polar transformation naturally handles outliers (they get absorbed into the radius, not the angle). The QJL residual further corrects bias. But we should still verify attention sink handling.

### 6.2 Triton FWHT Race Conditions
- **Key insight**:
  - FWHT butterfly pattern has data dependencies across stages
  - Triton handles intra-instance synchronization automatically
  - **Inter-instance synchronization** (across SMs) is NOT automatic — developer must manage it
  - If FWHT requires data exchange between instances → must launch multiple kernels or use global memory fences
  - Shared memory "bank conflicts" ≠ race conditions (bank conflicts are slow but correct; race conditions are wrong)
- **How it helps**: For implementing FWHT in Triton, keep the entire transform within one instance (thread block) to avoid inter-instance sync issues. The max size 16^3 * 2^3 from the fwht repo is specifically designed for this.

### 6.3 vLLM Attention Hook Problems
- **Key insight**:
  - Limited support for custom 4D attention masks
  - No independent prefill/decode backend selection
  - Evolving interfaces between vLLM versions
  - `torch.compile` conflicts with custom attention logic
  - Attention "switchboard" pattern makes direct manipulation difficult
- **How it helps**: We should use the plugin system, not try to hook into the attention layer directly. This avoids the multi-process and interface stability issues.

### 6.4 Quantization Numerical Stability
- **Key insight**:
  - Quantization errors propagate through attention scores
  - Small changes in K values can significantly alter attention patterns
  - Dequantization accuracy is critical — errors compound during attention
  - Group-wise scaling helps maintain stability
  - NVFP4 uses granular block scaling + high-precision FP8 scales for better accuracy
- **How it helps**: Our implementation must carefully handle the dequantization step. The TurboQuant approach (polar transform + analytic codebook + QJL residual) is designed to minimize these errors, but we need thorough numerical validation.

### 6.5 Triton-Specific Gotchas for Our Implementation
1. **Power-of-2 constraint**: Head dimensions (64, 128, 256) are all powers of 2 ✓
2. **Block size tuning**: Must autotune `BLOCK_SIZE`, `num_warps`, `num_stages`
3. **Mask for boundaries**: Always use masks for non-power-of-2 sequence lengths
4. **JIT warmup**: First kernel launch is slow — include warmup in benchmarks
5. **32-bit registers**: Triton operates on 32-bit registers — int4 unpacking requires bit ops on int32
6. **No native int4**: Must pack/unpack manually; can't use int4 in arithmetic directly
7. **Shared memory limits**: FWHT size limited by shared memory capacity — may need multiple kernel launches for large transforms

---

## Implementation Roadmap (Derived from Research)

### Phase 1: Core Kernels (Triton)
1. **Polar Transform Kernel**: Recursive 2D→polar conversion with Lloyd-Max codebook lookup
2. **Random Hadamard Preconditioning**: Triton FWHT kernel (adapt arthurfeeney/fwht)
3. **QJL Residual Quantizer**: 1-bit sign quantization of residual
4. **Fused Dequantize + Attention**: LUT-based Q×K product, dequantize V on-the-fly

### Phase 2: vLLM Integration
1. **Platform Plugin**: Create `vllm_turboquant_plugin` with custom attention backend
2. **Extend BaseKVCacheMethod**: Override quantize/dequantize for TurboQuant format
3. **Custom Attention Backend**: Modify FlashAttention to accept quantized KV blocks

### Phase 3: Validation
1. **Needle-in-haystack**: Reproduce paper's perfect accuracy claims
2. **LongBench**: Benchmark on standard tasks
3. **Performance profiling**: Verify 2-4x speedup claims on target hardware

### Key Dependencies
- **QJL reference code**: https://github.com/amirzandieh/QJL (CUDA kernels as reference)
- **FWHT Triton**: https://github.com/arthurfeeney/fwht (starting point for Hadamard)
- **HadaCore**: https://github.com/pytorch-labs/applied-ai/tree/main/kernels/cuda/inference/hadamard_transform (CUDA reference, future Triton version planned)
- **vLLM plugin docs**: https://docs.vllm.ai/en/latest/design/plugin_system/
- **KIVI**: https://github.com/jy-yuan/KIVI (reference for KV cache integration patterns)

### Critical Unknowns
1. **Lloyd-Max codebook for Beta distribution**: Must compute analytically from the paper's formulas — not available in any repo
2. **Optimal bit split**: Paper says 3.5 bits total, but the split between PolarQuant and QJL needs tuning
3. **RoPE interaction**: Must decide pre-RoPE vs post-RoPE quantization (PolarQuant paper suggests pre-RoPE)
4. **Grouped Query Attention**: Must handle GQA where K/V heads are shared across multiple Q heads
5. **Block size vs page size**: vLLM pages are 16 tokens — must handle quantized format within this structure
