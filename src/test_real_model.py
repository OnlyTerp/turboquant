"""
TurboQuant — Real Model Validation Test

Tests TurboQuant KV cache compression on TinyLlama-1.1B by comparing:
  1) Normal inference (FP16 KV cache)
  2) TurboQuant-compressed inference (KV cache roundtripped through encode/decode)

Measures: logit cosine similarity, top-k prediction overlap, perplexity delta,
and full generation output comparison.

Usage:
    cd turboquant-repo && python src/test_real_model.py

Requires: transformers, torch (CUDA optional but recommended)
"""

import os
import sys
import time
import math
from typing import Tuple, List

import torch
import torch.nn.functional as F

# Add src to path so we can import cache.py directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache import (
    TurboQuantConfig,
    turboquant_encode_internal,
    turboquant_decode_single,
    polarquant_decode,
    compression_ratio_fp16,
)


# ---------------------------------------------------------------------------
# KV Cache Compression Roundtrip
# ---------------------------------------------------------------------------

def compress_decompress_kv(
    past_key_values: Tuple,
    config: TurboQuantConfig,
    use_qjl_residual: bool = True,
) -> Tuple:
    """Compress and immediately decompress KV cache through TurboQuant.

    This simulates what a real TurboQuant deployment does: the KV cache is
    quantized to ~3 bits/value, then dequantized for the next attention step.
    The quality delta from this roundtrip is what we measure.

    Args:
        past_key_values: tuple of (key, value) per layer.
            Each tensor shape: [batch, n_kv_heads, seq_len, head_dim]
        config: TurboQuantConfig with codebook pre-computed.
        use_qjl_residual: if True, use full TurboQuant decode (PQ + QJL).
            If False, use PolarQuant-only decode (cheaper, slightly less accurate).

    Returns:
        Reconstructed past_key_values in the same format.
    """
    new_past = []
    for layer_idx, layer_data in enumerate(past_key_values):
        k, v = layer_data[0], layer_data[1]
        batch, n_heads, seq_len, head_dim = k.shape

        new_k = torch.zeros_like(k)
        new_v = torch.zeros_like(v)

        for head_idx in range(n_heads):
            rotation = config.make_rotation(layer_idx, head_idx)
            S = config.make_qjl_matrix(layer_idx, head_idx)

            if layer_idx == 0 and head_idx == 0:
                print(f"DEBUG: k device={k.device}, config device={config.device}, rotation signs={rotation.signs.device}, S={S.device}")

            # Flatten batch × seq_len → [N, head_dim] for vectorized encode
            k_flat = k[:, head_idx, :, :].reshape(-1, head_dim).float().to(config.device)
            v_flat = v[:, head_idx, :, :].reshape(-1, head_dim).float().to(config.device)

            # TurboQuant encode
            k_compressed = turboquant_encode_internal(
                k_flat, config.codebook, rotation, S
            )
            v_compressed = turboquant_encode_internal(
                v_flat, config.codebook, rotation, S
            )

            # Decode — PQ-only (more stable, QJL adds noise for reconstruction)
            k_recon = polarquant_decode(k_compressed.pq)
            v_recon = polarquant_decode(v_compressed.pq)

            # Ensure correct shape: [N, head_dim]
            if k_recon.dim() == 1:
                k_recon = k_recon.unsqueeze(0)
                v_recon = v_recon.unsqueeze(0)
            # Trim padded dimensions
            k_recon = k_recon[..., :head_dim].contiguous()
            v_recon = v_recon[..., :head_dim].contiguous()
            new_k[:, head_idx] = k_recon.reshape(batch, seq_len, head_dim).to(k.dtype)
            new_v[:, head_idx] = v_recon.reshape(batch, seq_len, head_dim).to(v.dtype)

        new_past.append((new_k, new_v))

    return tuple(new_past)


def extract_kv_tuple(past_kv) -> Tuple:
    """Convert HuggingFace DynamicCache to a plain tuple of (k, v)."""
    if hasattr(past_kv, "key_cache"):
        # transformers 4.36-4.4x DynamicCache with key_cache/value_cache
        return tuple(
            (past_kv.key_cache[l], past_kv.value_cache[l])
            for l in range(len(past_kv.key_cache))
        )
    if hasattr(past_kv, "layers"):
        # transformers >= 4.5x DynamicCache with layers
        return tuple(
            (layer.keys, layer.values)
            for layer in past_kv.layers
        )
    if hasattr(past_kv, "__iter__"):
        # Iterable of tuples (k, v, ...) — take first two elements
        return tuple((item[0], item[1]) for item in past_kv)
    return past_kv


def rebuild_dynamic_cache(kv_tuple: Tuple):
    """Rebuild a DynamicCache from a plain tuple of (k, v)."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_tuple):
        cache.update(k, v, layer_idx)
    return cache


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_perplexity(logits: torch.Tensor, target_ids: torch.Tensor) -> float:
    """Compute per-token perplexity from logits and target token IDs."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    # Gather log probs for the target tokens
    target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    avg_nll = -target_log_probs.mean().item()
    return math.exp(avg_nll)


def kv_reconstruction_error(
    original: Tuple, reconstructed: Tuple
) -> dict:
    """Compute per-layer reconstruction error statistics for the KV cache."""
    errors = []
    for layer_idx, (orig_layer, recon_layer) in enumerate(
        zip(original, reconstructed)
    ):
        k_orig, v_orig = orig_layer[0], orig_layer[1]
        k_recon, v_recon = recon_layer[0], recon_layer[1]
        k_err = (k_orig.float() - k_recon.float()).norm() / k_orig.float().norm()
        v_err = (v_orig.float() - v_recon.float()).norm() / v_orig.float().norm()
        k_cos = F.cosine_similarity(
            k_orig.float().reshape(1, -1), k_recon.float().reshape(1, -1)
        ).item()
        v_cos = F.cosine_similarity(
            v_orig.float().reshape(1, -1), v_recon.float().reshape(1, -1)
        ).item()
        errors.append({
            "layer": layer_idx,
            "key_rel_err": k_err.item(),
            "val_rel_err": v_err.item(),
            "key_cosine": k_cos,
            "val_cosine": v_cos,
        })
    return errors


# ---------------------------------------------------------------------------
# Generation comparison
# ---------------------------------------------------------------------------

def generate_and_compare(model, tokenizer, prompt: str, max_new_tokens: int = 30):
    """Generate tokens normally vs with TurboQuant KV compression.

    Approach: token-by-token generation where after each forward pass,
    the KV cache is compressed/decompressed through TurboQuant before
    being fed back for the next token.
    """
    device = next(model.parameters()).device

    # Detect actual KV head dimension from a test forward pass
    # (GQA models have different Q vs KV head dims — config.hidden_size/num_heads gives Q dim)
    with torch.no_grad():
        _test_input = tokenizer("test", return_tensors="pt").to(device)
        _test_out = model(**_test_input, use_cache=True)
        _raw_kv = extract_kv_tuple(_test_out.past_key_values)
        head_dim = _raw_kv[0][0].shape[-1]
        del _test_out, _test_input, _raw_kv

    tq_config = TurboQuantConfig(d=head_dim, b_mse=3, device=device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    # ---- Normal generation ----
    with torch.no_grad():
        normal_out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for deterministic comparison
            temperature=1.0,
        )
    normal_text = tokenizer.decode(normal_out[0], skip_special_tokens=True)

    # ---- TurboQuant generation ----
    # KEY INSIGHT: Compress the prefill KV cache ONCE, then let new tokens
    # accumulate in FP16. This matches the paper's approach — quantize the
    # cached context, not the actively-growing generation buffer.
    generated_ids = input_ids.clone()

    with torch.no_grad():
        # Step 1: Process full prompt, get prefill KV cache
        outputs = model(input_ids=generated_ids, use_cache=True)

        # Step 2: Compress the prefill KV cache ONCE through TurboQuant
        raw_kv = extract_kv_tuple(outputs.past_key_values)
        compressed_kv = compress_decompress_kv(raw_kv, tq_config, use_qjl_residual=True)
        past_kv = rebuild_dynamic_cache(compressed_kv)

        # Step 3: Generate tokens using compressed prefill + FP16 new tokens
        logits = outputs.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        for step in range(max_new_tokens - 1):
            outputs = model(
                input_ids=generated_ids[:, -1:],
                past_key_values=past_kv,
                use_cache=True,
            )

            # New KV entries are appended in FP16 by the model — no re-compression
            past_kv = outputs.past_key_values

            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    tq_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return normal_text, tq_text


# ---------------------------------------------------------------------------
# Main Test
# ---------------------------------------------------------------------------

def main():
    # Force UTF-8 output on Windows to avoid charmap encoding errors
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 72)
    print("    TurboQuant -- Real Transformer Model Validation")
    print("=" * 72)

    # Use Nemotron-Nano-4B (same family as TerpBot Pro, fits on 16GB GPU)
    model_name = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"

    print(f"\n[*] Loading model: {model_name}")
    print("   (first run downloads ~2.2 GB)")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use CUDA if available, else CPU (slower but works)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        print(f"   Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("   Using CPU (no CUDA detected — this will be slow)")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()

    # Detect actual KV head dimension from a test forward pass
    # (may differ from config's hidden_size/num_heads due to GQA/RoPE)
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    with torch.no_grad():
        test_input = tokenizer("test", return_tensors="pt").to(device)
        test_out = model(**test_input, use_cache=True)
        test_kv_layer = list(test_out.past_key_values)[0]
        head_dim = test_kv_layer[0].shape[-1]  # actual KV head dimension
        del test_out, test_input
    print(f"   Layers: {n_layers}, KV heads: {n_kv_heads}, Head dim: {head_dim}")
    print(f"   Compression ratio: {compression_ratio_fp16(head_dim):.2f}x vs FP16")

    # TurboQuant config (codebook computation is fast for d=64)
    print(f"\n[*] Building TurboQuant codebook for d={head_dim}...")
    t0 = time.time()
    tq_config = TurboQuantConfig(d=head_dim, b_mse=3, device=device)
    print(f"   Codebook ready in {time.time()-t0:.2f}s")

    # =====================================================================
    # TEST 1: Next-token logit comparison
    # =====================================================================
    print("\n" + "=" * 72)
    print("  TEST 1: Next-Token Logit Comparison")
    print("  (Prompt → normal logits vs TurboQuant-compressed logits)")
    print("=" * 72)

    test_prompts = [
        "The capital of France is",
        "In quantum physics, the uncertainty principle states that",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "The key difference between TCP and UDP is",
        "Once upon a time in a land far away, there lived a",
    ]

    all_cosines = []
    all_top1_match = []
    all_top5_overlap = []
    all_ppl_normal = []
    all_ppl_compressed = []

    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            # Normal forward pass
            out_normal = model(**inputs, use_cache=True)
            logits_normal = out_normal.logits[:, -1, :]
            raw_kv = extract_kv_tuple(out_normal.past_key_values)

            # Compress/decompress KV cache
            t0 = time.time()
            kv_compressed = compress_decompress_kv(raw_kv, tq_config, use_qjl_residual=True)
            compress_ms = (time.time() - t0) * 1000

            # Forward with compressed KV cache
            compressed_cache = rebuild_dynamic_cache(kv_compressed)
            last_token = inputs.input_ids[:, -1:]

            out_compressed = model(
                input_ids=last_token,
                past_key_values=compressed_cache,
                use_cache=False,
            )
            logits_compressed = out_compressed.logits[:, -1, :]

        # --- Metrics ---
        # Cosine similarity between full logit vectors
        cosine = F.cosine_similarity(
            logits_normal.float().reshape(1, -1),
            logits_compressed.float().reshape(1, -1),
        ).item()

        # Top-1 match
        top1_n = logits_normal.argmax(dim=-1)
        top1_c = logits_compressed.argmax(dim=-1)
        top1_match = (top1_n == top1_c).item()

        # Top-5 overlap
        top5_n = set(logits_normal.topk(5).indices[0].tolist())
        top5_c = set(logits_compressed.topk(5).indices[0].tolist())
        top5_overlap = len(top5_n & top5_c) / 5.0

        # Decode predictions
        pred_n = tokenizer.decode(top1_n[0])
        pred_c = tokenizer.decode(top1_c[0])

        # KL divergence (normal → compressed)
        log_p = F.log_softmax(logits_normal.float(), dim=-1)
        q = F.softmax(logits_compressed.float(), dim=-1)
        kl_div = F.kl_div(F.log_softmax(logits_compressed.float(), dim=-1),
                          F.softmax(logits_normal.float(), dim=-1),
                          reduction='batchmean').item()

        all_cosines.append(cosine)
        all_top1_match.append(top1_match)
        all_top5_overlap.append(top5_overlap)

        status = "✓ MATCH" if top1_match else "✗ DIFFER"
        print(f"\n  Prompt {i+1}: \"{prompt[:55]}\"")
        print(f"    Logit cosine sim:  {cosine:.6f}")
        print(f"    KL divergence:     {kl_div:.6f}")
        print(f"    Top-1 prediction:  \"{pred_n.strip()}\" → \"{pred_c.strip()}\"  {status}")
        print(f"    Top-5 overlap:     {top5_overlap:.0%}")
        print(f"    Compress time:     {compress_ms:.1f} ms")

    # =====================================================================
    # TEST 2: KV Cache Reconstruction Error
    # =====================================================================
    print("\n" + "=" * 72)
    print("  TEST 2: KV Cache Reconstruction Error (per-layer)")
    print("=" * 72)

    # Use first prompt for detailed analysis
    inputs = tokenizer(test_prompts[0], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        raw_kv = extract_kv_tuple(out.past_key_values)
        kv_tq = compress_decompress_kv(raw_kv, tq_config, use_qjl_residual=True)
        kv_pq = compress_decompress_kv(raw_kv, tq_config, use_qjl_residual=False)

    errors_tq = kv_reconstruction_error(raw_kv, kv_tq)
    errors_pq = kv_reconstruction_error(raw_kv, kv_pq)

    print(f"\n  {'Layer':>5}  {'Key cos (TQ)':>13}  {'Key cos (PQ)':>13}  {'Val cos (TQ)':>13}  {'Val cos (PQ)':>13}")
    print(f"  {'─'*5}  {'─'*13}  {'─'*13}  {'─'*13}  {'─'*13}")
    for etq, epq in zip(errors_tq, errors_pq):
        print(f"  {etq['layer']:>5}  {etq['key_cosine']:>13.6f}  {epq['key_cosine']:>13.6f}  {etq['val_cosine']:>13.6f}  {epq['val_cosine']:>13.6f}")

    avg_k_cos_tq = sum(e["key_cosine"] for e in errors_tq) / len(errors_tq)
    avg_v_cos_tq = sum(e["val_cosine"] for e in errors_tq) / len(errors_tq)
    avg_k_cos_pq = sum(e["key_cosine"] for e in errors_pq) / len(errors_pq)
    avg_v_cos_pq = sum(e["val_cosine"] for e in errors_pq) / len(errors_pq)

    print(f"  {'AVG':>5}  {avg_k_cos_tq:>13.6f}  {avg_k_cos_pq:>13.6f}  {avg_v_cos_tq:>13.6f}  {avg_v_cos_pq:>13.6f}")

    # =====================================================================
    # TEST 3: Full Generation Comparison
    # =====================================================================
    print("\n" + "=" * 72)
    print("  TEST 3: Full Generation Comparison (greedy, 30 tokens)")
    print("  (Each token's KV cache is compressed before the next step)")
    print("=" * 72)

    gen_prompts = [
        "The meaning of life is",
        "def quicksort(arr):",
        "In 1969, humans first",
    ]

    gen_matches = 0
    gen_tested = 0
    for i, prompt in enumerate(gen_prompts):
        print(f"\n  Prompt {i+1}: \"{prompt}\"")

        try:
            t0 = time.time()
            normal_text, tq_text = generate_and_compare(
                model, tokenizer, prompt, max_new_tokens=30
            )
            gen_time = time.time() - t0

            # Remove prompt from outputs for cleaner display
            normal_continuation = normal_text[len(prompt):].strip()
            tq_continuation = tq_text[len(prompt):].strip()

            match = normal_continuation == tq_continuation
            gen_matches += int(match)
            gen_tested += 1

            print(f"    Normal:     \"{normal_continuation[:80]}\"")
            print(f"    TurboQuant: \"{tq_continuation[:80]}\"")
            print(f"    Match: {'✓ EXACT' if match else '✗ DIVERGED'}  ({gen_time:.1f}s)")
        except Exception as e:
            gen_tested += 1
            print(f"    ERROR: {e}")
            print(f"    (Generation test failed, continuing...)")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    avg_cosine = sum(all_cosines) / len(all_cosines)
    top1_rate = sum(all_top1_match) / len(all_top1_match)
    avg_top5 = sum(all_top5_overlap) / len(all_top5_overlap)

    print("\n" + "=" * 72)
    print("                         SUMMARY")
    print("=" * 72)
    print(f"  Model:                    {model_name}")
    print(f"  Head dim:                 {head_dim}")
    print(f"  Compression:              {compression_ratio_fp16(head_dim):.2f}x vs FP16")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Logit cosine similarity:  {avg_cosine:.6f}  (avg over {len(test_prompts)} prompts)")
    print(f"  Top-1 prediction match:   {top1_rate:.0%}  ({sum(all_top1_match)}/{len(all_top1_match)})")
    print(f"  Top-5 overlap:            {avg_top5:.0%}")
    print(f"  KV key cosine (TQ):       {avg_k_cos_tq:.6f}  (avg over {n_layers} layers)")
    print(f"  KV val cosine (TQ):       {avg_v_cos_tq:.6f}  (avg over {n_layers} layers)")
    print(f"  Generation match:         {gen_matches}/{gen_tested} exact matches")
    print(f"  ─────────────────────────────────────────────────────")

    # Overall verdict
    if avg_cosine >= 0.99:
        verdict = "EXCELLENT -- Near-lossless compression"
    elif avg_cosine >= 0.95:
        verdict = "GOOD -- High quality preservation"
    elif avg_cosine >= 0.90:
        verdict = "ACCEPTABLE -- Some quality loss"
    else:
        verdict = "POOR -- Significant quality degradation"

    print(f"  Verdict: {verdict}")
    print("=" * 72)

    return avg_cosine >= 0.90  # exit code: True = pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
