"""
TurboQuant KV Cache Compression — Pure PyTorch Implementation

Implements the TurboQuant algorithm (Algorithms 1 & 2) for near-optimal vector
quantization of transformer KV caches, combining:
  - TurboQuant_mse (random rotation + scalar Lloyd-Max quantization per coordinate)
  - QJL (1-bit residual correction for unbiased inner products)

Total: ~3 bits per coordinate → ~4.9× compression vs FP16.

Reference: https://arxiv.org/abs/2504.19874 (ICLR 2026)
Authors: Zandieh, Daliri, Hadian, Mirrokni (Google Research / NYU / Google DeepMind)
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B_MSE = 2          # bits per coordinate for TurboQuant_mse stage
B_QJL = 1          # bits per coordinate for QJL residual stage
B_TOTAL = B_MSE + B_QJL  # = 3 bits total per coordinate
EPS = 1e-10        # numerical stability threshold


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform (FWHT)
# ---------------------------------------------------------------------------

def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform.

    Self-inverse up to scaling: FWHT(FWHT(x)) = d * x.
    Preserves norms up to scaling: ‖FWHT(x)‖² = d · ‖x‖².
    Complexity: O(d log d).

    Args:
        x: [..., d] where d is a power of 2.

    Returns:
        Transformed tensor, same shape as input.
    """
    d = x.shape[-1]
    y = x.clone()
    h = 1
    while h < d:
        y_view = y.reshape(*y.shape[:-1], -1, 2 * h)
        a = y_view[..., :h].clone()
        b = y_view[..., h:].clone()
        y_view[..., :h] = a + b
        y_view[..., h:] = a - b
        y = y_view.reshape(*y.shape)
        h *= 2
    return y


def fwht_inplace(x: torch.Tensor) -> None:
    """In-place Fast Walsh-Hadamard Transform."""
    d = x.shape[-1]
    h = 1
    while h < d:
        y_view = x.reshape(*x.shape[:-1], -1, 2 * h)
        a = y_view[..., :h].clone()
        b = y_view[..., h:].clone()
        y_view[..., :h] = a + b
        y_view[..., h:] = a - b
        h *= 2


# ---------------------------------------------------------------------------
# Randomized Hadamard Transform (rotation for TurboQuant)
# ---------------------------------------------------------------------------

def _generate_signs(d: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate deterministic random ±1 signs from a seed."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randint(0, 2, (d,), generator=g, device=device, dtype=torch.float32) * 2 - 1


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


class RandomHadamardRotation:
    """Randomized Hadamard Transform: Π·x = (1/√d) · H · (D_signs ⊙ x).

    This implements the random rotation from Algorithm 1 of TurboQuant.
    After rotation, each coordinate of Π·x follows a Beta distribution
    that converges to N(0, 1/d) in high dimensions (Lemma 1 of the paper).

    Inverse: Π^T · y = D_signs ⊙ ((1/√d) · H · y).
    """

    def __init__(self, d: int, seed: int, device: torch.device = torch.device("cpu")):
        self.d = d
        self.seed = seed
        self.sqrt_d = math.sqrt(d)
        self.signs = _generate_signs(d, seed, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation: Π · x."""
        signs = self.signs.to(device=x.device, dtype=x.dtype)
        y = x * signs
        if self.d & (self.d - 1) == 0:
            fwht_inplace(y)
            y = y / self.sqrt_d
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation: Π^T · y = D ⊙ ((1/√d) · H · y)."""
        z = y.clone()
        if self.d & (self.d - 1) == 0:
            fwht_inplace(z)
            z = z / self.sqrt_d
        z = z * self.signs.to(device=z.device, dtype=z.dtype)
        return z


# ---------------------------------------------------------------------------
# Scalar Lloyd-Max Codebook (TurboQuant Algorithm 1)
# ---------------------------------------------------------------------------

def _beta_pdf(x: torch.Tensor, d: int) -> torch.Tensor:
    """PDF of a coordinate of a uniformly random point on S^{d-1}.

    f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2)
    for x ∈ [-1, 1].

    In high dimensions (d ≥ 64), this converges to N(0, 1/d).
    See Lemma 1 of the TurboQuant paper.
    """
    valid = (x > -1.0) & (x < 1.0)
    pdf = torch.zeros_like(x)
    if valid.any():
        x_valid = x[valid]
        # Use log-space for numerical stability
        log_coeff = (
            torch.lgamma(torch.tensor(d / 2.0, dtype=torch.float64))
            - 0.5 * math.log(math.pi)
            - torch.lgamma(torch.tensor((d - 1) / 2.0, dtype=torch.float64))
        )
        log_body = ((d - 3) / 2.0) * torch.log((1.0 - x_valid.double() ** 2).clamp(min=1e-30))
        pdf[valid] = torch.exp(log_coeff + log_body).float()
    return pdf


@dataclass
class Codebook:
    """Scalar Lloyd-Max codebook for TurboQuant coordinate quantization.

    After random rotation, each coordinate follows a Beta distribution
    (≈ N(0, 1/d) for large d). This codebook is optimal for that distribution.
    The SAME codebook is used for ALL coordinates — no per-level variation.
    """
    centroids: torch.Tensor    # [K] centroid values
    boundaries: torch.Tensor   # [K+1] decision boundaries
    d: int                     # dimension (for scaling)
    b: int                     # bits per coordinate
    K: int                     # number of centroids = 2^b

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map rotated coordinates to codebook indices.

        Args:
            x: [..., d] rotated coordinates (each in approx [-1, 1])

        Returns:
            indices: [..., d] uint8 indices in {0, ..., K-1}
        """
        boundaries = self.boundaries.to(device=x.device, dtype=x.dtype)
        idx = torch.searchsorted(boundaries, x.contiguous(), right=False) - 1
        return idx.clamp(0, self.K - 1).to(torch.uint8)

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        """Map codebook indices back to coordinate values.

        Args:
            idx: [..., d] uint8 indices

        Returns:
            x_hat: [..., d] reconstructed rotated coordinates
        """
        centroids = self.centroids.to(device=idx.device)
        return centroids[idx.long()]


def compute_lloyd_max_codebook(
    d: int,
    b: int,
    max_iter: int = 500,
    tol: float = 1e-12,
    device: torch.device = torch.device("cpu"),
) -> Codebook:
    """Compute scalar Lloyd-Max codebook for TurboQuant.

    Solves the continuous 1D k-means problem for the Beta distribution
    of coordinates after random rotation (Eq. 3 in the paper).

    For d ≥ 64 this is approximately N(0, 1/d), with optimal centroids:
      b=1: {±√(2/(πd))}
      b=2: {±0.453/√d, ±1.51/√d}

    Args:
        d: vector dimension (determines the Beta distribution shape)
        b: bits per coordinate
        max_iter: Lloyd-Max iterations
        tol: convergence tolerance
        device: torch device

    Returns:
        Codebook with centroids and boundaries
    """
    K = 2 ** b

    # Support range: coordinates of unit vectors lie in [-1, 1]
    # but concentrate around 0 with std ≈ 1/√d
    # Use a practical support of [-4/√d, 4/√d] for numerical stability
    sigma = 1.0 / math.sqrt(d)
    lo = max(-1.0, -6.0 * sigma)
    hi = min(1.0, 6.0 * sigma)

    grid_size = 16385
    grid = torch.linspace(lo, hi, grid_size, device=device, dtype=torch.float64)

    # Compute PDF on grid
    if d >= 64:
        # Use Gaussian approximation N(0, 1/d) for numerical stability
        pdf = torch.exp(-0.5 * d * grid ** 2) * math.sqrt(d / (2.0 * math.pi))
    else:
        pdf = _beta_pdf(grid.float(), d).double()

    pdf = pdf.clamp_min(0)
    mass = torch.trapz(pdf, grid)
    if mass.item() <= EPS:
        raise ValueError(f"Degenerate density for d={d}")
    pdf = pdf / mass

    # Initialize centroids uniformly
    centroids = torch.linspace(lo, hi, K + 2, device=device, dtype=torch.float64)[1:-1]
    boundaries = torch.empty(K + 1, device=device, dtype=torch.float64)
    boundaries[0] = lo
    boundaries[-1] = hi

    # Lloyd-Max iteration
    for _ in range(max_iter):
        # Update boundaries (midpoints between centroids)
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        old_centroids = centroids.clone()

        # Update centroids (conditional means within each interval)
        for i in range(K):
            mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            grid_slice = grid[mask]
            pdf_slice = pdf[mask]
            if grid_slice.numel() < 2:
                centroids[i] = 0.5 * (boundaries[i] + boundaries[i + 1])
                continue

            interval_mass = torch.trapz(pdf_slice, grid_slice)
            if interval_mass.item() <= EPS:
                centroids[i] = 0.5 * (boundaries[i] + boundaries[i + 1])
            else:
                interval_moment = torch.trapz(pdf_slice * grid_slice, grid_slice)
                centroids[i] = interval_moment / interval_mass

        if (centroids - old_centroids).abs().max().item() < tol:
            break

    # Final boundary update
    boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

    return Codebook(
        centroids=centroids.float(),
        boundaries=boundaries.float(),
        d=d,
        b=b,
        K=K,
    )


# ---------------------------------------------------------------------------
# QJL Random Matrix
# ---------------------------------------------------------------------------

def generate_qjl_matrix(d: int, seed: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Generate the QJL random matrix S ∈ R^{d×d} with i.i.d. N(0,1) entries.

    The paper specifies Gaussian entries (Definition 1): S_{i,j} ~ N(0, 1).
    This provides unbiased inner product estimation via QJL (Lemma 4).

    Note: Previous versions used Rademacher (±1). The paper uses Gaussian,
    but Rademacher also satisfies the JL property. We use Rademacher for
    efficiency (no floating point storage needed for the matrix entries).
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randint(0, 2, (d, d), generator=g, device=device).float() * 2 - 1


# ---------------------------------------------------------------------------
# TurboQuant MSE: Scalar per-coordinate quantization (Algorithm 1)
# ---------------------------------------------------------------------------

@dataclass
class PolarQuantCompressed:
    """Compressed representation from TurboQuant_mse stage.

    Named PolarQuantCompressed for backward compatibility, but this now
    implements scalar per-coordinate quantization (Algorithm 1), NOT
    recursive polar transform.
    """
    norm: torch.Tensor            # [batch] L2 norms
    indices: torch.Tensor         # [batch, d] uint8 indices in {0..K-1}
    codebook: Codebook
    rotation: RandomHadamardRotation

    @property
    def d(self) -> int:
        return self.codebook.d


def polarquant_encode(x: torch.Tensor, codebook: Codebook, rotation: RandomHadamardRotation) -> PolarQuantCompressed:
    """TurboQuant_mse encode (Algorithm 1, lines 4-7):
    1. Rotate: y = Π · x
    2. Quantize each coordinate independently using Lloyd-Max codebook
    3. Store norm separately in FP16
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    x = x.detach().float()
    norm = x.norm(dim=-1).to(torch.float16)
    zero_mask = norm < EPS

    # Normalize to unit sphere (paper assumes ||x|| = 1)
    safe_norm = norm.float().clamp(min=EPS)
    x_unit = x / safe_norm.unsqueeze(-1)

    # Pad to power-of-2 if needed
    d_actual = x.shape[-1]
    d_padded = _next_power_of_two(d_actual)
    if d_padded != d_actual:
        padding = torch.zeros(*x_unit.shape[:-1], d_padded - d_actual,
                              device=x.device, dtype=x.dtype)
        x_unit = torch.cat([x_unit, padding], dim=-1)

    # Step 1: Random rotation (induces Beta ≈ N(0, 1/d) on each coordinate)
    y = rotation.forward(x_unit)

    # Step 2: Scalar quantization per coordinate
    indices = codebook.quantize(y)

    if zero_mask.any():
        indices[zero_mask] = 0

    return PolarQuantCompressed(norm=norm, indices=indices, codebook=codebook, rotation=rotation)


def polarquant_decode(c: PolarQuantCompressed) -> torch.Tensor:
    """TurboQuant_mse decode (Algorithm 1, lines 8-11):
    1. Look up centroids for each coordinate
    2. Inverse rotation: x_hat = Π^T · y_hat
    3. Scale by stored norm
    """
    # Step 1: Dequantize (centroid lookup)
    y_hat = c.codebook.dequantize(c.indices)

    # Step 2: Inverse rotation
    x_hat = c.rotation.inverse(y_hat)

    # Unpad if needed
    d_actual = c.codebook.d
    if x_hat.shape[-1] > d_actual:
        x_hat = x_hat[..., :d_actual]

    # Step 3: Scale by norm
    x_hat = x_hat * c.norm.float().unsqueeze(-1)

    zero_mask = c.norm < EPS
    if zero_mask.any():
        x_hat[zero_mask] = 0.0

    return x_hat


# ---------------------------------------------------------------------------
# QJL: 1-bit residual quantization (part of Algorithm 2)
# ---------------------------------------------------------------------------

@dataclass
class QJLCompressed:
    """Compressed representation from QJL (1-bit per coord + residual norm)."""
    signs: torch.Tensor       # [batch, d] in {0, 1}
    r_norm: torch.Tensor      # [batch] residual norm
    S: torch.Tensor           # [d, d] random matrix

    @property
    def d(self) -> int:
        return self.signs.shape[-1]


def qjl_encode(residual: torch.Tensor, S: torch.Tensor) -> QJLCompressed:
    """QJL encode (Algorithm 2, line 7): sign(S · r).

    The QJL provides unbiased inner product estimation on the residual.
    """
    if residual.dim() == 1:
        residual = residual.unsqueeze(0)

    r_norm = residual.norm(dim=-1)
    safe_norm = r_norm.clamp(min=EPS)
    r_unit = residual / safe_norm.unsqueeze(-1)

    projected = r_unit @ S.T  # [batch, d]
    signs = (projected >= 0).long()

    return QJLCompressed(signs=signs, r_norm=r_norm, S=S)


# ---------------------------------------------------------------------------
# TurboQuant: Complete pipeline (Algorithm 2)
# ---------------------------------------------------------------------------

@dataclass
class TurboQuantCompressed:
    """Complete TurboQuant compressed representation (b+1 bits per coord).

    Combines TurboQuant_mse (b bits) + QJL (1 bit) for unbiased inner products.
    """
    pq: PolarQuantCompressed  # MSE-optimal quantization
    qjl: QJLCompressed        # residual correction

    @property
    def d(self) -> int:
        return self.pq.d


class TurboQuantConfig:
    """Configuration for a TurboQuant cache."""

    def __init__(self, d: int = 128, b_mse: int = B_MSE, device: torch.device = torch.device("cpu")):
        self.d = d
        self.d_padded = _next_power_of_two(d)
        self.b_mse = b_mse
        self.device = device
        # Compute scalar codebook for the padded dimension
        # (rotation operates on padded dim, so codebook must match)
        self.codebook = compute_lloyd_max_codebook(self.d_padded, b_mse, device=device)
        self.codebook.d = d  # track actual dimension for unpadding

    def make_rotation(self, layer_idx: int, head_idx: int) -> RandomHadamardRotation:
        seed = ((layer_idx * 1000003) ^ (head_idx * 999979) ^ 0xA5A5A5A5) & 0xFFFFFFFF
        return RandomHadamardRotation(self.d_padded, seed, self.device)

    def make_qjl_matrix(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        seed = ((layer_idx * 1000003) ^ (head_idx * 999979) ^ 0x5A5A5A5A) & 0xFFFFFFFF
        return generate_qjl_matrix(self.d, seed, self.device)


def turboquant_encode_internal(
    x: torch.Tensor,
    codebook: Codebook,
    rotation: RandomHadamardRotation,
    S: torch.Tensor,
) -> TurboQuantCompressed:
    """Full TurboQuant encode (Algorithm 2, lines 4-8):
    1. MSE-optimal quantization via scalar Lloyd-Max
    2. Compute residual
    3. QJL 1-bit quantization of residual
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    pq = polarquant_encode(x, codebook, rotation)
    x_hat = polarquant_decode(pq).float()

    # Compute residual in original space
    x_for_residual = x.detach().float()
    if x_for_residual.shape[-1] != x_hat.shape[-1]:
        x_for_residual = x_for_residual[..., :x_hat.shape[-1]]
    residual = x_for_residual - x_hat
    qjl = qjl_encode(residual, S)

    return TurboQuantCompressed(pq=pq, qjl=qjl)


def turboquant_decode_single(c: TurboQuantCompressed) -> torch.Tensor:
    """Full TurboQuant decode (Algorithm 2, lines 9-12):
    x_hat = DeQuant_mse(idx) + √(π/2)/d · ‖r‖ · S^T · qjl_signs
    """
    k_hat = polarquant_decode(c.pq)  # MSE reconstruction

    # QJL residual correction
    signs_f = c.qjl.signs.float() * 2 - 1  # {-1, +1}
    d = c.d
    scale = math.sqrt(math.pi / 2) / d
    r_hat = (signs_f @ c.qjl.S) * scale
    r_hat = r_hat * c.qjl.r_norm.unsqueeze(-1)

    return k_hat + r_hat


# ---------------------------------------------------------------------------
# TurboQuant Cache
# ---------------------------------------------------------------------------

class TurboQuantCache:
    """TurboQuant-compressed KV cache for transformer attention."""

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d: int = 128,
        b_mse: int = B_MSE,
        device: torch.device = torch.device("cpu"),
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d = d
        self.device = device
        self.config = TurboQuantConfig(d, b_mse, device=device)

        self.rotations: List[List[RandomHadamardRotation]] = []
        self.qjl_matrices: List[List[torch.Tensor]] = []
        for l in range(n_layers):
            self.rotations.append([])
            self.qjl_matrices.append([])
            for h in range(n_heads):
                self.rotations[l].append(self.config.make_rotation(l, h))
                self.qjl_matrices[l].append(self.config.make_qjl_matrix(l, h))

        self.cache: List[List[List[Tuple[TurboQuantCompressed, TurboQuantCompressed]]]] = []
        for l in range(n_layers):
            self.cache.append([])
            for h in range(n_heads):
                self.cache[l].append([])

    @property
    def seq_len(self) -> int:
        if self.n_layers == 0 or self.n_heads == 0:
            return 0
        return len(self.cache[0][0])

    def store(self, layer_idx: int, head_idx: int, k_vec: torch.Tensor, v_vec: torch.Tensor):
        rotation = self.rotations[layer_idx][head_idx]
        S = self.qjl_matrices[layer_idx][head_idx]
        k_c = turboquant_encode_internal(k_vec, self.config.codebook, rotation, S)
        v_c = turboquant_encode_internal(v_vec, self.config.codebook, rotation, S)
        self.cache[layer_idx][head_idx].append((k_c, v_c))

    def store_batch(self, layer_idx: int, head_idx: int, k_vecs: torch.Tensor, v_vecs: torch.Tensor):
        rotation = self.rotations[layer_idx][head_idx]
        S = self.qjl_matrices[layer_idx][head_idx]
        k_all = turboquant_encode_internal(k_vecs, self.config.codebook, rotation, S)
        v_all = turboquant_encode_internal(v_vecs, self.config.codebook, rotation, S)

        for i in range(k_vecs.shape[0]):
            k_single = TurboQuantCompressed(
                pq=PolarQuantCompressed(
                    norm=k_all.pq.norm[i:i+1], indices=k_all.pq.indices[i:i+1],
                    codebook=k_all.pq.codebook, rotation=k_all.pq.rotation,
                ),
                qjl=QJLCompressed(
                    signs=k_all.qjl.signs[i:i+1], r_norm=k_all.qjl.r_norm[i:i+1], S=S,
                ),
            )
            v_single = TurboQuantCompressed(
                pq=PolarQuantCompressed(
                    norm=v_all.pq.norm[i:i+1], indices=v_all.pq.indices[i:i+1],
                    codebook=v_all.pq.codebook, rotation=v_all.pq.rotation,
                ),
                qjl=QJLCompressed(
                    signs=v_all.qjl.signs[i:i+1], r_norm=v_all.qjl.r_norm[i:i+1], S=S,
                ),
            )
            self.cache[layer_idx][head_idx].append((k_single, v_single))

    def compute_attention(
        self, layer_idx: int, head_idx: int, q_vec: torch.Tensor, causal: bool = True,
        qjl_score_weight: float = 0.5,
    ) -> torch.Tensor:
        """Compute attention output using compressed KV cache."""
        d = self.d
        seq_len = len(self.cache[layer_idx][head_idx])
        if seq_len == 0:
            return torch.zeros(d, device=self.device)

        q_vec = q_vec.float()
        S = self.qjl_matrices[layer_idx][head_idx]
        qjl_scale = math.sqrt(math.pi / 2) / d

        q_proj = S @ q_vec  # pre-project query for QJL scoring

        # Batch-decode all PQ keys
        pq_norms = torch.stack([
            self.cache[layer_idx][head_idx][t][0].pq.norm.squeeze(0)
            for t in range(seq_len)
        ])
        pq_indices = torch.cat([
            self.cache[layer_idx][head_idx][t][0].pq.indices
            for t in range(seq_len)
        ], dim=0)

        rotation = self.rotations[layer_idx][head_idx]
        pq_batch = PolarQuantCompressed(
            norm=pq_norms,
            indices=pq_indices,
            codebook=self.config.codebook,
            rotation=rotation,
        )
        k_hat_all = polarquant_decode(pq_batch)
        score_pq_all = (k_hat_all @ q_vec) / math.sqrt(d)

        if qjl_score_weight > 0.0:
            signs_pm_all = torch.cat([
                self.cache[layer_idx][head_idx][t][0].qjl.signs
                for t in range(seq_len)
            ], dim=0).float() * 2 - 1

            r_norms_all = torch.stack([
                self.cache[layer_idx][head_idx][t][0].qjl.r_norm.squeeze(0)
                for t in range(seq_len)
            ]).float()

            qjl_ips = (signs_pm_all @ q_proj)
            score_qjl_all = qjl_ips * qjl_scale * r_norms_all / math.sqrt(d)
            scores = score_pq_all + qjl_score_weight * score_qjl_all
        else:
            scores = score_pq_all

        attn_weights = F.softmax(scores, dim=0)

        # Batch-decode all PQ values
        v_pq_norms = torch.stack([
            self.cache[layer_idx][head_idx][t][1].pq.norm.squeeze(0)
            for t in range(seq_len)
        ])
        v_pq_indices = torch.cat([
            self.cache[layer_idx][head_idx][t][1].pq.indices
            for t in range(seq_len)
        ], dim=0)

        v_pq_batch = PolarQuantCompressed(
            norm=v_pq_norms,
            indices=v_pq_indices,
            codebook=self.config.codebook,
            rotation=rotation,
        )
        v_hat_all = polarquant_decode(v_pq_batch)

        output = (attn_weights.unsqueeze(-1) * v_hat_all.float()).sum(0)
        return output


# ---------------------------------------------------------------------------
# Utility: Compression ratio analysis
# ---------------------------------------------------------------------------

def compression_ratio_fp16(d: int, b_mse: int = B_MSE) -> float:
    """Compute compression ratio vs FP16."""
    fp16_bits = d * 16
    # TurboQuant: b_mse bits per coordinate (MSE) + 1 bit per coordinate (QJL)
    # + FP16 norm + FP16 residual norm
    tq_bits = d * b_mse + 16 + d * 1 + 16
    return fp16_bits / tq_bits


def memory_bytes_per_vector(d: int, b_mse: int = B_MSE) -> Tuple[int, int]:
    """Returns (tq_bytes, fp16_bytes) per vector."""
    tq_bits = d * b_mse + 16 + d * 1 + 16
    tq_bytes = (tq_bits + 7) // 8
    fp16_bytes = d * 2
    return tq_bytes, fp16_bytes
