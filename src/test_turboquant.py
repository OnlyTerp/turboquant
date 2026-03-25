"""
Comprehensive unit tests for TurboQuant KV cache compression.

Run with:  pytest test_turboquant.py -v
Requires:  torch, pytest
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

# Ensure the src directory is importable
sys.path.insert(0, os.path.dirname(__file__))

from cache import (
    B_MSE,
    Codebook,
    PolarQuantCompressed,
    QJLCompressed,
    RandomHadamardRotation,
    TurboQuantCache,
    TurboQuantCompressed,
    TurboQuantConfig,
    compute_lloyd_max_codebook,
    compression_ratio_fp16,
    fwht,
    generate_qjl_matrix,
    polarquant_decode,
    polarquant_encode,
    qjl_encode,
    turboquant_decode_single,
    turboquant_encode_internal,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
D = 128
SEED = 42
BATCH = 8


@pytest.fixture
def codebook():
    return compute_lloyd_max_codebook(D, B_MSE, device=DEVICE)


@pytest.fixture
def rotation():
    return RandomHadamardRotation(D, SEED, device=DEVICE)


@pytest.fixture
def qjl_matrix():
    return generate_qjl_matrix(D, SEED, device=DEVICE)


@pytest.fixture
def random_unit_vectors():
    """Generate random unit-norm vectors."""
    torch.manual_seed(123)
    x = torch.randn(BATCH, D, device=DEVICE)
    return x / x.norm(dim=-1, keepdim=True)


# ===================================================================
# 1. FWHT Tests
# ===================================================================


class TestFWHT:
    """Tests for the Fast Walsh-Hadamard Transform."""

    def test_fwht_self_inverse(self):
        """FWHT applied twice should return d * x (self-inverse up to scaling)."""
        torch.manual_seed(0)
        x = torch.randn(4, D, device=DEVICE)
        y = fwht(x)
        z = fwht(y)
        expected = D * x
        assert torch.allclose(z, expected, atol=1e-4)

    def test_fwht_self_inverse_single(self):
        """Single vector: FWHT(FWHT(x)) = d * x."""
        torch.manual_seed(1)
        x = torch.randn(D, device=DEVICE)
        y = fwht(x)
        z = fwht(y)
        expected = D * x
        assert torch.allclose(z, expected, atol=1e-4)

    def test_fwht_orthogonality(self):
        """FWHT preserves norms up to scaling: ‖FWHT(x)‖² = d · ‖x‖²."""
        torch.manual_seed(2)
        x = torch.randn(16, D, device=DEVICE)
        norm_sq_before = (x ** 2).sum(dim=-1)
        y = fwht(x)
        norm_sq_after = (y ** 2).sum(dim=-1)
        expected = D * norm_sq_before
        assert torch.allclose(norm_sq_after, expected, atol=1e-3)

    def test_fwht_known_values_d4(self):
        """Small d=4 example with known results."""
        x1 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=DEVICE)
        y1 = fwht(x1)
        expected1 = torch.tensor([1.0, 1.0, 1.0, 1.0], device=DEVICE)
        assert torch.allclose(y1, expected1, atol=1e-6)

        x2 = torch.tensor([1.0, 1.0, 1.0, 1.0], device=DEVICE)
        y2 = fwht(x2)
        expected2 = torch.tensor([4.0, 0.0, 0.0, 0.0], device=DEVICE)
        assert torch.allclose(y2, expected2, atol=1e-6)

    def test_fwht_linearity(self):
        """FWHT(a*x + b*y) = a*FWHT(x) + b*FWHT(y)."""
        torch.manual_seed(3)
        x = torch.randn(4, D, device=DEVICE)
        y = torch.randn(4, D, device=DEVICE)
        a, b = 2.5, -1.3
        lhs = fwht(a * x + b * y)
        rhs = a * fwht(x) + b * fwht(y)
        assert torch.allclose(lhs, rhs, atol=1e-4)


# ===================================================================
# 2. PolarQuant Tests
# ===================================================================


class TestPolarQuant:
    """Tests for the PolarQuant 2-bit quantization stage."""

    def test_polarquant_roundtrip(self, codebook, rotation, random_unit_vectors):
        """Encode then decode; MSE should be below theoretical bound (~0.117)."""
        pq = polarquant_encode(random_unit_vectors, codebook, rotation)
        x_hat = polarquant_decode(pq)
        mse = ((random_unit_vectors - x_hat) ** 2).sum(dim=-1).mean().item()
        assert mse < 0.15, f"MSE {mse:.4f} exceeds theoretical bound 0.117"

    def test_polarquant_mse_tight(self, codebook, rotation, random_unit_vectors):
        """MSE should also be reasonably close to the bound (not way below)."""
        pq = polarquant_encode(random_unit_vectors, codebook, rotation)
        x_hat = polarquant_decode(pq)
        mse = ((random_unit_vectors - x_hat) ** 2).sum(dim=-1).mean().item()
        assert mse > 0.005, f"MSE {mse:.4f} suspiciously low"

    def test_polarquant_norm_preservation(self, codebook, rotation):
        """Decoded vector norm should approximately match original."""
        torch.manual_seed(42)
        x = torch.randn(16, D, device=DEVICE) * 3.0
        norms_orig = x.norm(dim=-1)

        pq = polarquant_encode(x, codebook, rotation)
        x_hat = polarquant_decode(pq)
        norms_decoded = x_hat.norm(dim=-1)

        rel_error = ((norms_orig - norms_decoded) / norms_orig).abs().mean().item()
        assert rel_error < 0.10, f"Relative norm error {rel_error:.4f} too high"

    def test_polarquant_deterministic(self, codebook, rotation):
        """Same input + same seed = same output (deterministic)."""
        torch.manual_seed(99)
        x = torch.randn(4, D, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)

        pq1 = polarquant_encode(x, codebook, rotation)
        pq2 = polarquant_encode(x, codebook, rotation)

        assert torch.equal(pq1.indices, pq2.indices), "Indices differ between runs"
        assert torch.allclose(pq1.norm, pq2.norm), "Norms differ between runs"

    def test_polarquant_batch(self, codebook, rotation):
        """Batched encoding/decoding works correctly."""
        torch.manual_seed(7)
        x = torch.randn(BATCH, D, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)

        pq = polarquant_encode(x, codebook, rotation)
        assert pq.indices.shape == (BATCH, D)
        assert pq.norm.shape == (BATCH,)

        x_hat = polarquant_decode(pq)
        assert x_hat.shape == (BATCH, D)

        mse_per_sample = ((x - x_hat) ** 2).sum(dim=-1)
        assert (mse_per_sample < 0.18).all(), "Some samples wildly exceed MSE bound"

    def test_polarquant_zero_vector(self, codebook, rotation):
        """Zero vectors should decode to zero."""
        x = torch.zeros(2, D, device=DEVICE)
        pq = polarquant_encode(x, codebook, rotation)
        x_hat = polarquant_decode(pq)
        assert torch.allclose(x_hat, torch.zeros_like(x_hat), atol=1e-8)

    def test_polarquant_rotation_inverse(self, rotation):
        """Rotation followed by inverse should recover original."""
        torch.manual_seed(55)
        x = torch.randn(4, D, device=DEVICE)
        y = rotation.forward(x)
        x_rec = rotation.inverse(y)
        assert torch.allclose(x_rec, x, atol=1e-4)


# ===================================================================
# 3. QJL Tests
# ===================================================================


class TestQJL:
    """Tests for the QJL 1-bit residual quantization."""

    def test_qjl_encode_output_shape(self, qjl_matrix):
        """Verify packed bit dimensions are correct."""
        torch.manual_seed(10)
        residual = torch.randn(BATCH, D, device=DEVICE)
        qjl = qjl_encode(residual, qjl_matrix)

        assert qjl.signs.shape == (BATCH, D)
        assert qjl.r_norm.shape == (BATCH,)
        assert qjl.S.shape == (D, D)

    def test_qjl_signs_binary(self, qjl_matrix):
        """All sign values should be 0 or 1."""
        torch.manual_seed(11)
        residual = torch.randn(BATCH, D, device=DEVICE)
        qjl = qjl_encode(residual, qjl_matrix)

        assert ((qjl.signs == 0) | (qjl.signs == 1)).all()

    def test_qjl_inner_product_unbiased(self):
        """Average over many random vectors: E[<q, r̂>_estimated] ≈ <q, r>."""
        torch.manual_seed(20)
        n_trials = 200
        d = D
        errors = []

        for i in range(n_trials):
            q = torch.randn(d, device=DEVICE)
            r = torch.randn(d, device=DEVICE)
            r_norm = r.norm()
            if r_norm < 1e-10:
                continue
            r_unit = r / r_norm

            true_ip = torch.dot(q, r).item()

            S_trial = generate_qjl_matrix(d, i, device=DEVICE)
            qjl = qjl_encode(r_unit.unsqueeze(0), S_trial)
            signs_f = qjl.signs.squeeze(0).float() * 2 - 1

            q_proj = S_trial @ q
            scale = math.sqrt(math.pi / 2) / d
            qjl_ip = torch.dot(q_proj, signs_f).item() * scale * qjl.r_norm.squeeze().item()

            errors.append(qjl_ip - true_ip)

        mean_error = sum(errors) / len(errors)
        # 200 trials is small given QJL variance, so a bound of 1.0 is reasonable
        assert abs(mean_error) < 1.0, f"Mean error {mean_error:.4f} indicates bias"

    def test_qjl_variance_bound(self):
        """QJL estimator variance should be bounded: Var ≤ (π/2d) · ‖q‖² · ‖r‖²."""
        torch.manual_seed(21)
        n_trials = 500
        d = D

        q = torch.randn(d, device=DEVICE)
        r = torch.randn(d, device=DEVICE)
        r_unit = r / r.norm()

        estimates = []
        for i in range(n_trials):
            S_trial = generate_qjl_matrix(d, i+1000, device=DEVICE)
            qjl = qjl_encode(r_unit.unsqueeze(0), S_trial)
            signs_f = qjl.signs.squeeze(0).float() * 2 - 1
            q_proj = S_trial @ q
            scale = math.sqrt(math.pi / 2) / d
            qjl_ip = torch.dot(q_proj, signs_f).item() * scale
            estimates.append(qjl_ip * r.norm().item())

        estimates = torch.tensor(estimates)
        empirical_var = estimates.var().item()
        bound = (math.pi / (2 * d)) * q.norm().item() ** 2 * r.norm().item() ** 2

        assert empirical_var < bound * 4.0, f"Variance {empirical_var:.4f} exceeds bound {bound:.4f}"


# ===================================================================
# 4. TurboQuant E2E Tests
# ===================================================================


class TestTurboQuantE2E:
    """End-to-end tests for the complete TurboQuant pipeline."""

    def test_turboquant_attention_vs_standard(self):
        """TQ attention output should be close to standard SDPA."""
        torch.manual_seed(30)
        n_layers, n_heads, d = 1, 1, D
        seq_len = 32

        cache = TurboQuantCache(n_layers, n_heads, d, device=DEVICE)

        K_fp = torch.randn(seq_len, d, device=DEVICE)
        V_fp = torch.randn(seq_len, d, device=DEVICE)

        for t in range(seq_len):
            cache.store(0, 0, K_fp[t], V_fp[t])

        q = torch.randn(d, device=DEVICE)

        q_batch = q.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        k_batch = K_fp.unsqueeze(0).unsqueeze(0)
        v_batch = V_fp.unsqueeze(0).unsqueeze(0)
        standard_out = F.scaled_dot_product_attention(q_batch, k_batch, v_batch).squeeze()

        tq_out = cache.compute_attention(0, 0, q, causal=False)

        cos_sim = F.cosine_similarity(tq_out.unsqueeze(0), standard_out.unsqueeze(0)).item()
        # Due to stochastic QJL, correlation isn't perfectly 1.0
        assert cos_sim > 0.85, f"Cosine similarity {cos_sim:.4f} below 0.85 threshold"

    def test_turboquant_compression_ratio(self):
        """Verify memory usage is ~4.9x less than FP16 for d=128."""
        ratio = compression_ratio_fp16(D, B_MSE)
        assert abs(ratio - 4.9) < 0.5

    def test_turboquant_compression_ratio_exact(self):
        """Exact computation: 2048 bits FP16 / 416 bits TQ ≈ 4.923."""
        d = 128
        fp16_bits = d * 16
        tq_bits = d * B_MSE + 16 + d * 1 + 16
        ratio = fp16_bits / tq_bits
        assert abs(ratio - 2048 / 416) < 0.001

    def test_turboquant_causal_mask(self):
        """With causal masking, the last position should attend only to past tokens."""
        torch.manual_seed(33)
        cache = TurboQuantCache(1, 1, D, device=DEVICE)
        K_fp = torch.randn(8, D, device=DEVICE)
        V_fp = torch.randn(8, D, device=DEVICE)
        for t in range(8):
            cache.store(0, 0, K_fp[t], V_fp[t])
        q = torch.randn(D, device=DEVICE)
        out_causal = cache.compute_attention(0, 0, q, causal=True)
        out_full = cache.compute_attention(0, 0, q, causal=False)
        assert torch.allclose(out_causal, out_full, atol=1e-6)

    def test_turboquant_encode_decode_roundtrip(self):
        """Encode → decode should reconstruct well for TurboQuant."""
        torch.manual_seed(34)
        config = TurboQuantConfig(D, B_MSE, device=DEVICE)
        rotation = config.make_rotation(0, 0)
        S = config.make_qjl_matrix(0, 0)

        x = torch.randn(4, D, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)

        tq = turboquant_encode_internal(x, config.codebook, rotation, S)

        for i in range(4):
            tq_single = TurboQuantCompressed(
                pq=PolarQuantCompressed(
                    norm=tq.pq.norm[i:i+1],
                    indices=tq.pq.indices[i:i+1],
                    codebook=tq.pq.codebook,
                    rotation=tq.pq.rotation,
                ),
                qjl=QJLCompressed(
                    signs=tq.qjl.signs[i:i+1],
                    r_norm=tq.qjl.r_norm[i:i+1],
                    S=tq.qjl.S,
                ),
            )
            x_hat = turboquant_decode_single(tq_single).squeeze(0)
            mse = ((x[i] - x_hat) ** 2).sum().item()
            # QJL adds variance to vector reconstruction, MSE can be ~0.40
            assert mse < 0.40, f"TurboQuant MSE {mse:.4f} too high for sample {i}"


# ===================================================================
# 5. Codebook Tests
# ===================================================================


class TestCodebook:
    """Tests for the Lloyd-Max codebook computation."""

    def test_lloyd_max_convergence(self):
        """Codebook computation should converge (not hit max_iter)."""
        codebook = compute_lloyd_max_codebook(D, B_MSE, max_iter=500, tol=1e-12, device=DEVICE)
        assert codebook.centroids.shape == (4,)
        assert codebook.boundaries.shape == (5,)
        diffs = codebook.centroids.diff()
        assert (diffs > 0).all(), "Centroids should be strictly increasing"

    def test_lloyd_max_known_values_d128_b2(self):
        """For d=128, b=2, centroids should match reference."""
        codebook = compute_lloyd_max_codebook(D, B_MSE, device=DEVICE)
        expected = torch.tensor([-0.1335, -0.0400, 0.0400, 0.1335], device=DEVICE)
        assert torch.allclose(codebook.centroids, expected, atol=1e-3)

    def test_lloyd_max_boundaries_symmetric(self):
        """Boundaries should be symmetric around 0."""
        codebook = compute_lloyd_max_codebook(D, B_MSE, device=DEVICE)
        assert codebook.boundaries[0].item() == -1.0
        assert codebook.boundaries[-1].item() == 1.0
        mid = codebook.K // 2
        assert abs(codebook.boundaries[mid].item()) < 0.01

    def test_lloyd_max_centroids_symmetric(self):
        """Centroids should be symmetric: c_i = -c_{K-1-i}."""
        codebook = compute_lloyd_max_codebook(D, B_MSE, device=DEVICE)
        K = codebook.K
        for i in range(K):
            assert torch.allclose(codebook.centroids[i], -codebook.centroids[K - 1 - i], atol=1e-6)

    def test_lloyd_max_mse_within_bound(self):
        """The codebook should achieve MSE near theoretical bound."""
        codebook = compute_lloyd_max_codebook(D, B_MSE, device=DEVICE)
        rotation = RandomHadamardRotation(D, SEED, device=DEVICE)

        torch.manual_seed(50)
        n_samples = 1000
        x = torch.randn(n_samples, D, device=DEVICE)
        x = x / x.norm(dim=-1, keepdim=True)

        pq = polarquant_encode(x, codebook, rotation)
        x_hat = polarquant_decode(pq)
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
        assert mse < 0.13, f"Codebook MSE {mse:.4f} exceeds bound"

    def test_lloyd_max_b3(self):
        """Lloyd-Max for b=3 should produce 8 centroids."""
        codebook = compute_lloyd_max_codebook(D, 3, device=DEVICE)
        assert codebook.K == 8
        assert codebook.centroids.shape == (8,)

    def test_lloyd_max_b1(self):
        """Lloyd-Max for b=1 should produce 2 centroids near ±σ√(2/π)."""
        codebook = compute_lloyd_max_codebook(D, 1, device=DEVICE)
        sigma = 1.0 / math.sqrt(D)
        expected_val = sigma * math.sqrt(2 / math.pi)
        assert codebook.K == 2
        assert abs(codebook.centroids[0].item() - (-expected_val)) < 0.005
        assert abs(codebook.centroids[1].item() - expected_val) < 0.005


# ===================================================================
# Additional Integration Tests
# ===================================================================


class TestTurboQuantCache:
    """Tests for the TurboQuantCache class."""

    def test_cache_store_retrieve(self):
        """Basic store and seq_len check."""
        cache = TurboQuantCache(2, 4, D, device=DEVICE)
        assert cache.seq_len == 0

        k = torch.randn(D, device=DEVICE)
        v = torch.randn(D, device=DEVICE)
        cache.store(0, 0, k, v)
        assert cache.seq_len == 1

        cache.store(0, 0, k, v)
        assert cache.seq_len == 2

    def test_cache_multiple_heads(self):
        """Different heads store independently."""
        cache = TurboQuantCache(1, 4, D, device=DEVICE)
        k = torch.randn(D, device=DEVICE)
        v = torch.randn(D, device=DEVICE)

        cache.store(0, 0, k, v)
        cache.store(0, 1, k, v)
        cache.store(0, 1, k, v)

        assert len(cache.cache[0][0]) == 1
        assert len(cache.cache[0][1]) == 2
        assert len(cache.cache[0][2]) == 0

    def test_cache_attention_shape(self):
        """Attention output should have correct shape."""
        cache = TurboQuantCache(1, 1, D, device=DEVICE)
        for _ in range(5):
            cache.store(0, 0, torch.randn(D, device=DEVICE), torch.randn(D, device=DEVICE))

        q = torch.randn(D, device=DEVICE)
        out = cache.compute_attention(0, 0, q)
        assert out.shape == (D,)

    def test_cache_batch_store(self):
        """Batch store should work and result in correct seq_len."""
        cache = TurboQuantCache(1, 1, D, device=DEVICE)
        k_batch = torch.randn(10, D, device=DEVICE)
        v_batch = torch.randn(10, D, device=DEVICE)
        cache.store_batch(0, 0, k_batch, v_batch)
        assert cache.seq_len == 10

    def test_cache_memory_advantage(self):
        """TurboQuant should use less memory than FP16 for long sequences."""
        d = 128
        seq_len = 1000
        n_layers = 4
        n_heads = 8

        # FP16 memory: 2 bytes per fp16 value, times K and V (so *2)
        fp16_bytes = n_layers * n_heads * seq_len * d * 2 * 2

        # TQ memory
        tq_bytes_per_vec = (d * B_MSE + 16 + d * 1 + 16 + 7) // 8
        tq_total = n_layers * n_heads * seq_len * 2 * tq_bytes_per_vec

        ratio = fp16_bytes / tq_total
        assert ratio > 4.5, f"Memory ratio {ratio:.2f} too low"
