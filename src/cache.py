"""
TurboQuant KV Cache Compression — Pure PyTorch Implementation

Complete implementation of the TurboQuant algorithm (Algorithms 1 & 2) from:
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
  Zandieh, Daliri, Hadian, Mirrokni — ICLR 2026
  https://arxiv.org/abs/2504.19874

Features implemented:
  - TurboQuant_mse (random rotation + scalar Lloyd-Max quantization)
  - TurboQuant_prod (MSE quantization + QJL residual for unbiased inner products)
  - Mixed-precision outlier channel handling (Section 2.3)
  - Online codebook construction from actual data (Section 4.1)
  - Gaussian QJL random matrix (Definition 1)

Reference: https://arxiv.org/abs/2504.19874
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B_MSE = 3          # default MSE bits per coordinate (paper "3.5-bit" mixed-precision mode)
B_QJL = 1          # bits per coordinate for QJL residual stage
B_TOTAL = B_MSE + B_QJL  # base total bits/coordinate before mixed-precision averaging
EPS = 1e-10        # numerical stability threshold

# Outlier channel defaults (Section 2.3)
N_OUTLIER_CHANNELS = 32    # number of outlier channels (out of 128)
OUTLIER_EXTRA_BITS = 1     # extra bits for outlier channels


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform (FWHT)
# ---------------------------------------------------------------------------

def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform.

    Self-inverse up to scaling: FWHT(FWHT(x)) = d * x.
    Preserves norms up to scaling: ‖FWHT(x)‖² = d · ‖x‖².
    Complexity: O(d log d).
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
    """In-place Fast Walsh-Hadamard Transform.

    WARNING: x must be contiguous. Non-contiguous tensors will produce
    incorrect results because reshape may return a copy.
    """
    if not x.is_contiguous():
        raise ValueError("fwht_inplace requires contiguous input. Call .contiguous() first.")
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


class RandomOrthogonalRotation:
    """Dense random orthogonal rotation via QR decomposition.

    Generates a full random orthogonal matrix P (P^T P = I) from a
    Gaussian random matrix using QR factorization. This is the
    theoretically exact approach described in the paper.

    O(d²) storage and O(d²) per vector — use RandomHadamardRotation
    for the O(d log d) practical variant.
    """

    def __init__(self, d: int, seed: int, device: torch.device = torch.device("cpu")):
        self.d = d
        self.seed = seed
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)
        A = torch.randn(d, d, generator=g)
        Q, R = torch.linalg.qr(A)
        # Ensure uniform Haar measure: fix sign ambiguity of QR
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        Q = Q * diag_sign.unsqueeze(0)
        self.P = Q.to(device)  # [d, d] orthogonal matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation: P · x."""
        P = self.P.to(device=x.device, dtype=x.dtype)
        return x @ P.T

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation: P^T · y  (P is orthogonal so P^{-1} = P^T)."""
        P = self.P.to(device=y.device, dtype=y.dtype)
        return y @ P


class RandomHadamardRotation:
    """Randomized Hadamard Transform: Π·x = (1/√d) · H · (D_signs ⊙ x).

    O(d log d) practical variant of random orthogonal rotation.
    After rotation, each coordinate follows Beta ≈ N(0, 1/d) (Lemma 1).

    The paper notes: "For our implementation, we use random rotation matrices
    (square matrices P satisfying P^T P = I), which preserve the norms and
    inner products exactly" (Section 3.2 footnote).
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
        """Apply inverse rotation: Π^T · y."""
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
    for x ∈ [-1, 1]. Converges to N(0, 1/d) for d ≥ 64.
    """
    valid = (x > -1.0) & (x < 1.0)
    pdf = torch.zeros_like(x)
    if valid.any():
        x_valid = x[valid]
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
    """
    centroids: torch.Tensor    # [K] centroid values
    boundaries: torch.Tensor   # [K+1] decision boundaries
    d: int                     # dimension (for tracking)
    b: int                     # bits per coordinate
    K: int                     # number of centroids = 2^b

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Map rotated coordinates to codebook indices."""
        boundaries = self.boundaries.to(device=x.device, dtype=x.dtype)
        idx = torch.searchsorted(boundaries, x.contiguous(), right=False) - 1
        return idx.clamp(0, self.K - 1).to(torch.uint8)

    def dequantize(self, idx: torch.Tensor) -> torch.Tensor:
        """Map codebook indices back to coordinate values."""
        centroids = self.centroids.to(device=idx.device)
        return centroids[idx.long()]


def _solve_lloyd_max(
    pdf: torch.Tensor,
    grid: torch.Tensor,
    K: int,
    max_iter: int = 500,
    tol: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve the Lloyd-Max quantization problem on a 1D density."""
    device = grid.device
    lo, hi = grid[0].item(), grid[-1].item()

    centroids = torch.linspace(lo, hi, K + 2, device=device, dtype=torch.float64)[1:-1]
    boundaries = torch.empty(K + 1, device=device, dtype=torch.float64)
    boundaries[0] = lo
    boundaries[-1] = hi

    for _ in range(max_iter):
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        old_centroids = centroids.clone()

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
                centroids[i] = torch.trapz(pdf_slice * grid_slice, grid_slice) / interval_mass

        if (centroids - old_centroids).abs().max().item() < tol:
            break

    boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
    return centroids.float(), boundaries.float()


def compute_lloyd_max_codebook(
    d: int,
    b: int,
    max_iter: int = 500,
    tol: float = 1e-12,
    device: torch.device = torch.device("cpu"),
) -> Codebook:
    """Compute scalar Lloyd-Max codebook for TurboQuant (Eq. 3).

    Solves the continuous 1D k-means for the Beta distribution of
    coordinates after random rotation.
    """
    K = 2 ** b
    sigma = 1.0 / math.sqrt(d)
    lo = max(-1.0, -6.0 * sigma)
    hi = min(1.0, 6.0 * sigma)

    grid_size = 16385
    grid = torch.linspace(lo, hi, grid_size, device=device, dtype=torch.float64)

    if d >= 64:
        pdf = torch.exp(-0.5 * d * grid ** 2) * math.sqrt(d / (2.0 * math.pi))
    else:
        pdf = _beta_pdf(grid.float(), d).double()

    pdf = pdf.clamp_min(0)
    mass = torch.trapz(pdf, grid)
    if mass.item() <= EPS:
        raise ValueError(f"Degenerate density for d={d}")
    pdf = pdf / mass

    centroids, boundaries = _solve_lloyd_max(pdf, grid, K, max_iter, tol)
    return Codebook(centroids=centroids, boundaries=boundaries, d=d, b=b, K=K)


def compute_online_codebook(
    data: torch.Tensor,
    b: int,
    max_iter: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Codebook:
    """Compute codebook from actual rotated data using 1D k-means (Section 4.1).

    The paper notes: "online approach requires additional clustering computation
    during every prefill stage, this one-time cost is offset by improved
    performance compared to the offline approach."

    Args:
        data: [N, d] rotated coordinate values (flattened to 1D for k-means)
        b: bits per coordinate
        max_iter: k-means iterations
    """
    K = 2 ** b
    # Flatten all coordinates into a single 1D distribution
    flat = data.reshape(-1).float().to(device)

    # Build empirical PDF via histogram
    n_bins = 16385
    lo = flat.min().item() - 1e-6
    hi = flat.max().item() + 1e-6
    grid = torch.linspace(lo, hi, n_bins, device=device, dtype=torch.float64)
    hist = torch.histogram(flat.cpu().double(), bins=n_bins, range=(lo, hi))
    pdf = hist.hist.to(device).double()
    pdf = pdf / (pdf.sum() * (hi - lo) / n_bins)

    # Grid for PDF (bin centers)
    bin_edges = hist.bin_edges.to(device).double()
    grid = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    centroids, boundaries = _solve_lloyd_max(pdf, grid, K, max_iter)
    return Codebook(centroids=centroids, boundaries=boundaries, d=data.shape[-1], b=b, K=K)


# ---------------------------------------------------------------------------
# QJL Random Matrix (Definition 1)
# ---------------------------------------------------------------------------

def generate_qjl_matrix(d: int, seed: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Generate the QJL random matrix S ∈ R^{d×d} with i.i.d. N(0, 1) entries.

    Definition 1 of the paper: S_{i,j} ~ N(0, 1).
    Gaussian entries provide unbiased inner product estimation (Lemma 4).
    """
    g = torch.Generator(device='cpu')  # Generate on CPU for consistency
    g.manual_seed(seed)
    S = torch.randn(d, d, generator=g)
    return S.to(device)


# ---------------------------------------------------------------------------
# Mixed-precision outlier channel support (Section 2.3)
# ---------------------------------------------------------------------------

@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed-precision outlier channel quantization.

    From Section 2.3: "splitting channels into outlier and non-outlier sets,
    and applying two independent instances of TurboQuant to each, allocating
    higher bit precision to outliers."

    Example: 2.5-bit mode = 32 outlier channels at 3 bits + 96 regular at 2 bits
             = (32×3 + 96×2)/128 = 2.5 effective bits
    """
    n_outlier: int = N_OUTLIER_CHANNELS
    b_regular: int = 3
    b_outlier: int = 4
    outlier_indices: Optional[torch.Tensor] = None  # [n_outlier] channel indices
    regular_indices: Optional[torch.Tensor] = None  # [d - n_outlier] channel indices
    codebook_regular: Optional[Codebook] = None
    codebook_outlier: Optional[Codebook] = None
    rotation_regular: Optional[RandomHadamardRotation] = None
    rotation_outlier: Optional[RandomHadamardRotation] = None

    @property
    def effective_bits(self) -> float:
        if self.outlier_indices is None or self.regular_indices is None:
            return float(self.b_regular)
        d = len(self.outlier_indices) + len(self.regular_indices)
        return (len(self.outlier_indices) * self.b_outlier +
                len(self.regular_indices) * self.b_regular) / d


def detect_outlier_channels(
    x_calibration: torch.Tensor,
    n_outlier: int = N_OUTLIER_CHANNELS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identify outlier channels from original-space calibration statistics.

    Args:
        x_calibration: [N, d] calibration vectors in the original coordinate space
        n_outlier: number of channels to mark as outliers

    Returns:
        outlier_indices: [n_outlier] channel indices with highest variance
        regular_indices: [d - n_outlier] remaining channel indices
    """
    if x_calibration.dim() == 1:
        x_calibration = x_calibration.unsqueeze(0)

    d = x_calibration.shape[-1]
    if d <= 1:
        return torch.zeros(0, dtype=torch.long), torch.arange(d, dtype=torch.long)

    n_outlier = min(n_outlier, d - 1)

    if x_calibration.shape[0] > 1:
        channel_var = x_calibration.float().var(dim=0, unbiased=False)
    else:
        channel_var = x_calibration.float().pow(2).mean(dim=0)

    _, sorted_idx = channel_var.sort(descending=True)
    outlier_indices = sorted_idx[:n_outlier].sort().values
    regular_indices = sorted_idx[n_outlier:].sort().values

    return outlier_indices, regular_indices


# ---------------------------------------------------------------------------
# TurboQuant MSE: Scalar per-coordinate quantization (Algorithm 1)
# ---------------------------------------------------------------------------

@dataclass
class PolarQuantCompressed:
    """Compressed representation from TurboQuant_mse stage.

    Supports both uniform and mixed-precision quantization.
    Named PolarQuantCompressed for backward compatibility.
    """
    norm: Optional[torch.Tensor] = None                     # [batch] L2 norms for uniform mode
    indices: Optional[torch.Tensor] = None                  # [batch, d] uint8 indices for uniform mode
    codebook: Optional[Codebook] = None                     # primary codebook for uniform mode
    rotation: Optional[RandomHadamardRotation] = None
    original_dim: int = 0
    regular_norm: Optional[torch.Tensor] = None
    outlier_norm: Optional[torch.Tensor] = None
    regular_indices: Optional[torch.Tensor] = None
    outlier_indices: Optional[torch.Tensor] = None
    regular_quantized_indices: Optional[torch.Tensor] = None
    outlier_quantized_indices: Optional[torch.Tensor] = None
    codebook_regular: Optional[Codebook] = None
    codebook_outlier: Optional[Codebook] = None
    rotation_regular: Optional[RandomHadamardRotation] = None
    rotation_outlier: Optional[RandomHadamardRotation] = None

    @property
    def d(self) -> int:
        if self.original_dim:
            return self.original_dim
        if self.codebook is not None:
            return self.codebook.d
        return 0

    @property
    def is_mixed_precision(self) -> bool:
        return self.regular_quantized_indices is not None or self.outlier_quantized_indices is not None


def polarquant_encode(
    x: torch.Tensor,
    codebook: Codebook,
    rotation: RandomHadamardRotation,
    mixed: Optional[MixedPrecisionConfig] = None,
) -> PolarQuantCompressed:
    """TurboQuant_mse encode with optional two-instance mixed precision."""
    if x.dim() == 1:
        x = x.unsqueeze(0)

    x = x.detach().float()
    d_actual = x.shape[-1]

    if (
        mixed is not None and
        mixed.outlier_indices is not None and
        mixed.regular_indices is not None and
        mixed.codebook_regular is not None and
        mixed.codebook_outlier is not None and
        mixed.rotation_regular is not None and
        mixed.rotation_outlier is not None
    ):
        regular_idx = mixed.regular_indices.to(device=x.device)
        outlier_idx = mixed.outlier_indices.to(device=x.device)

        x_regular = x[..., regular_idx]
        x_outlier = x[..., outlier_idx]

        regular_norm = x_regular.norm(dim=-1).to(torch.float16)
        outlier_norm = x_outlier.norm(dim=-1).to(torch.float16)

        safe_regular_norm = regular_norm.float().clamp(min=EPS)
        safe_outlier_norm = outlier_norm.float().clamp(min=EPS)
        x_regular_unit = x_regular / safe_regular_norm.unsqueeze(-1)
        x_outlier_unit = x_outlier / safe_outlier_norm.unsqueeze(-1)

        if mixed.rotation_regular.d != x_regular_unit.shape[-1]:
            x_regular_unit = F.pad(x_regular_unit, (0, mixed.rotation_regular.d - x_regular_unit.shape[-1]))
        if mixed.rotation_outlier.d != x_outlier_unit.shape[-1]:
            x_outlier_unit = F.pad(x_outlier_unit, (0, mixed.rotation_outlier.d - x_outlier_unit.shape[-1]))

        y_regular = mixed.rotation_regular.forward(x_regular_unit)
        y_outlier = mixed.rotation_outlier.forward(x_outlier_unit)

        regular_quantized_indices = mixed.codebook_regular.quantize(y_regular)
        outlier_quantized_indices = mixed.codebook_outlier.quantize(y_outlier)

        regular_zero_mask = regular_norm < EPS
        outlier_zero_mask = outlier_norm < EPS
        if regular_zero_mask.any():
            regular_quantized_indices[regular_zero_mask] = 0
        if outlier_zero_mask.any():
            outlier_quantized_indices[outlier_zero_mask] = 0

        return PolarQuantCompressed(
            original_dim=d_actual,
            regular_norm=regular_norm,
            outlier_norm=outlier_norm,
            regular_indices=regular_idx.detach().cpu(),
            outlier_indices=outlier_idx.detach().cpu(),
            regular_quantized_indices=regular_quantized_indices,
            outlier_quantized_indices=outlier_quantized_indices,
            codebook_regular=mixed.codebook_regular,
            codebook_outlier=mixed.codebook_outlier,
            rotation_regular=mixed.rotation_regular,
            rotation_outlier=mixed.rotation_outlier,
        )

    norm = x.norm(dim=-1).to(torch.float16)
    zero_mask = norm < EPS
    safe_norm = norm.float().clamp(min=EPS)
    x_unit = x / safe_norm.unsqueeze(-1)

    d_padded = _next_power_of_two(d_actual)
    if d_padded != d_actual:
        x_unit = F.pad(x_unit, (0, d_padded - d_actual))

    y = rotation.forward(x_unit)
    indices = codebook.quantize(y)
    if zero_mask.any():
        indices[zero_mask] = 0
    return PolarQuantCompressed(
        norm=norm,
        indices=indices,
        codebook=codebook,
        rotation=rotation,
        original_dim=d_actual,
    )


def polarquant_decode(c: PolarQuantCompressed) -> torch.Tensor:
    """TurboQuant_mse decode with mixed-precision support.

    Algorithm 1, lines 8-11:
    1. Dequantize using appropriate codebook per channel
    2. Inverse rotation
    3. Scale by stored norm
    """
    if c.is_mixed_precision:
        batch = 0
        device = None
        if c.regular_quantized_indices is not None:
            batch = c.regular_quantized_indices.shape[0]
            device = c.regular_quantized_indices.device
        elif c.outlier_quantized_indices is not None:
            batch = c.outlier_quantized_indices.shape[0]
            device = c.outlier_quantized_indices.device
        x_hat = torch.zeros(batch, c.d, device=device, dtype=torch.float32)

        if c.regular_quantized_indices is not None and c.regular_indices is not None:
            y_regular = c.codebook_regular.dequantize(c.regular_quantized_indices)
            x_regular = c.rotation_regular.inverse(y_regular)[..., :len(c.regular_indices)]
            x_regular = x_regular * c.regular_norm.float().unsqueeze(-1)
            regular_zero_mask = c.regular_norm < EPS
            if regular_zero_mask.any():
                x_regular[regular_zero_mask] = 0.0
            regular_idx = c.regular_indices.to(device=device)
            x_hat[..., regular_idx] = x_regular

        if c.outlier_quantized_indices is not None and c.outlier_indices is not None:
            y_outlier = c.codebook_outlier.dequantize(c.outlier_quantized_indices)
            x_outlier = c.rotation_outlier.inverse(y_outlier)[..., :len(c.outlier_indices)]
            x_outlier = x_outlier * c.outlier_norm.float().unsqueeze(-1)
            outlier_zero_mask = c.outlier_norm < EPS
            if outlier_zero_mask.any():
                x_outlier[outlier_zero_mask] = 0.0
            outlier_idx = c.outlier_indices.to(device=device)
            x_hat[..., outlier_idx] = x_outlier

        return x_hat

    y_hat = c.codebook.dequantize(c.indices)
    x_hat = c.rotation.inverse(y_hat)
    if x_hat.shape[-1] > c.d:
        x_hat = x_hat[..., :c.d]
    x_hat = x_hat * c.norm.float().unsqueeze(-1)
    zero_mask = c.norm < EPS
    if zero_mask.any():
        x_hat[zero_mask] = 0.0
    return x_hat


# ---------------------------------------------------------------------------
# QJL: 1-bit residual quantization (Algorithm 2, lines 6-7)
# ---------------------------------------------------------------------------

@dataclass
class QJLCompressed:
    """Compressed representation from QJL (1-bit per coord + residual norm)."""
    signs: torch.Tensor       # [batch, d] in {0, 1}
    r_norm: torch.Tensor      # [batch] residual norm
    S: torch.Tensor           # [d, d] random Gaussian matrix

    @property
    def d(self) -> int:
        return self.signs.shape[-1]


def qjl_encode(residual: torch.Tensor, S: torch.Tensor) -> QJLCompressed:
    """QJL encode (Algorithm 2, line 7): qjl = sign(S · r)."""
    if residual.dim() == 1:
        residual = residual.unsqueeze(0)

    r_norm = residual.norm(dim=-1)
    safe_norm = r_norm.clamp(min=EPS)
    r_unit = residual / safe_norm.unsqueeze(-1)

    projected = r_unit @ S.T
    signs = (projected >= 0).long()

    return QJLCompressed(signs=signs, r_norm=r_norm, S=S)


# ---------------------------------------------------------------------------
# TurboQuant: Complete pipeline (Algorithm 2)
# ---------------------------------------------------------------------------

@dataclass
class TurboQuantCompressed:
    """Complete TurboQuant compressed representation.

    Combines TurboQuant_mse + QJL for unbiased inner products.
    """
    pq: PolarQuantCompressed
    qjl: QJLCompressed

    @property
    def d(self) -> int:
        return self.pq.d


class TurboQuantConfig:
    """Configuration for a TurboQuant cache.

    Supports three modes:
    - Uniform: all channels at b_mse bits (default)
    - Mixed 2.5-bit: 32 outlier channels at 3 bits, 96 regular at 2 bits
    - Mixed 3.5-bit: 32 outlier channels at 4 bits, 96 regular at 3 bits
    """

    def __init__(
        self,
        d: int = 128,
        b_mse: int = B_MSE,
        device: torch.device = torch.device("cpu"),
        mixed_precision: bool = True,
        n_outlier: int = N_OUTLIER_CHANNELS,
        b_outlier: Optional[int] = None,
        use_online_codebook: bool = False,
        rotation_mode: str = "hadamard",
    ):
        """
        Args:
            rotation_mode: "hadamard" (default, O(d log d)) or "dense" (full
                random orthogonal via QR, O(d²)). Both satisfy P^T P = I.
        """
        self.d = d
        self.d_padded = _next_power_of_two(d)
        self.b_mse = b_mse
        self.device = device
        self.rotation_mode = rotation_mode
        self.mixed_precision = mixed_precision
        self.n_outlier = n_outlier
        self.b_outlier = b_outlier if b_outlier is not None else b_mse + 1
        self.use_online_codebook = use_online_codebook

        self.codebook = compute_lloyd_max_codebook(self.d_padded, b_mse, device=device)
        self.codebook.d = d

        self._mixed_configs: dict = {}

    def _make_rotation_impl(self, d: int, seed: int):
        """Create a rotation using the configured mode."""
        if self.rotation_mode == "dense":
            return RandomOrthogonalRotation(d, seed, self.device)
        return RandomHadamardRotation(d, seed, self.device)

    def make_rotation(self, layer_idx: int, head_idx: int):
        seed = ((layer_idx * 1000003) ^ (head_idx * 999979) ^ 0xA5A5A5A5) & 0xFFFFFFFF
        return self._make_rotation_impl(self.d_padded, seed)

    def make_subset_rotation(self, layer_idx: int, head_idx: int, subset: str, subset_dim: int):
        if self.rotation_mode == "dense":
            # Dense mode doesn't need power-of-2 padding
            salt = 0x13572468 if subset == "regular" else 0x24681357
            seed = ((layer_idx * 1000003) ^ (head_idx * 999979) ^ salt ^ subset_dim) & 0xFFFFFFFF
            return RandomOrthogonalRotation(subset_dim, seed, self.device)
        subset_padded = _next_power_of_two(max(subset_dim, 1))
        salt = 0x13572468 if subset == "regular" else 0x24681357
        seed = ((layer_idx * 1000003) ^ (head_idx * 999979) ^ salt ^ subset_padded) & 0xFFFFFFFF
        return RandomHadamardRotation(subset_padded, seed, self.device)

    def make_qjl_matrix(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        seed = ((layer_idx * 1000003) ^ (head_idx * 999979) ^ 0x5A5A5A5A) & 0xFFFFFFFF
        return generate_qjl_matrix(self.d, seed, self.device)

    def get_mixed_config(
        self,
        layer_idx: int,
        head_idx: int,
        calibration_vectors: Optional[torch.Tensor] = None,
    ) -> Optional[MixedPrecisionConfig]:
        if not self.mixed_precision:
            return None

        key = (layer_idx, head_idx)
        if key not in self._mixed_configs:
            if calibration_vectors is None:
                return None

            if calibration_vectors.dim() == 1:
                calibration_vectors = calibration_vectors.unsqueeze(0)

            outlier_idx, regular_idx = detect_outlier_channels(calibration_vectors, self.n_outlier)
            regular_dim = int(regular_idx.numel())
            outlier_dim = int(outlier_idx.numel())
            rotation_regular = self.make_subset_rotation(layer_idx, head_idx, "regular", regular_dim)
            rotation_outlier = self.make_subset_rotation(layer_idx, head_idx, "outlier", outlier_dim)

            if self.use_online_codebook:
                regular_data = calibration_vectors[..., regular_idx].float()
                outlier_data = calibration_vectors[..., outlier_idx].float()

                regular_norm = regular_data.norm(dim=-1, keepdim=True).clamp(min=EPS)
                outlier_norm = outlier_data.norm(dim=-1, keepdim=True).clamp(min=EPS)
                regular_unit = regular_data / regular_norm
                outlier_unit = outlier_data / outlier_norm

                if rotation_regular.d != regular_unit.shape[-1]:
                    regular_unit = F.pad(regular_unit, (0, rotation_regular.d - regular_unit.shape[-1]))
                if rotation_outlier.d != outlier_unit.shape[-1]:
                    outlier_unit = F.pad(outlier_unit, (0, rotation_outlier.d - outlier_unit.shape[-1]))

                regular_rotated = rotation_regular.forward(regular_unit)
                outlier_rotated = rotation_outlier.forward(outlier_unit)

                codebook_regular = compute_online_codebook(
                    regular_rotated, self.b_mse, device=self.device
                )
                codebook_regular.d = regular_dim
                codebook_outlier = compute_online_codebook(
                    outlier_rotated, self.b_outlier, device=self.device
                )
                codebook_outlier.d = outlier_dim
            else:
                codebook_regular = compute_lloyd_max_codebook(
                    rotation_regular.d, self.b_mse, device=self.device
                )
                codebook_regular.d = regular_dim
                codebook_outlier = compute_lloyd_max_codebook(
                    rotation_outlier.d, self.b_outlier, device=self.device
                )
                codebook_outlier.d = outlier_dim

            self._mixed_configs[key] = MixedPrecisionConfig(
                n_outlier=self.n_outlier,
                b_regular=self.b_mse,
                b_outlier=self.b_outlier,
                outlier_indices=outlier_idx,
                regular_indices=regular_idx,
                codebook_regular=codebook_regular,
                codebook_outlier=codebook_outlier,
                rotation_regular=rotation_regular,
                rotation_outlier=rotation_outlier,
            )

        return self._mixed_configs.get(key)


def turboquant_encode_internal(
    x: torch.Tensor,
    codebook: Codebook,
    rotation: RandomHadamardRotation,
    S: torch.Tensor,
    mixed: Optional[MixedPrecisionConfig] = None,
) -> TurboQuantCompressed:
    """Full TurboQuant encode (Algorithm 2):
    1. MSE-optimal quantization (with optional mixed precision)
    2. Compute residual
    3. QJL 1-bit quantization of residual
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    pq = polarquant_encode(x, codebook, rotation, mixed=mixed)
    x_hat = polarquant_decode(pq).float()

    # Residual in original space
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
    k_hat = polarquant_decode(c.pq)

    signs_f = c.qjl.signs.float() * 2 - 1
    d = c.d
    scale = math.sqrt(math.pi / 2) / d
    r_hat = (signs_f @ c.qjl.S) * scale
    r_hat = r_hat * c.qjl.r_norm.unsqueeze(-1)

    return k_hat + r_hat


# ---------------------------------------------------------------------------
# TurboQuant Cache
# ---------------------------------------------------------------------------

class TurboQuantCache:
    """TurboQuant-compressed KV cache for transformer attention.

    Supports both uniform and mixed-precision quantization.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d: int = 128,
        b_mse: int = B_MSE,
        device: torch.device = torch.device("cpu"),
        mixed_precision: bool = True,
        n_outlier: int = N_OUTLIER_CHANNELS,
        b_outlier: Optional[int] = None,
        use_online_codebook: bool = False,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d = d
        self.device = device
        self.config = TurboQuantConfig(
            d, b_mse, device=device,
            mixed_precision=mixed_precision,
            n_outlier=n_outlier,
            b_outlier=b_outlier,
            use_online_codebook=use_online_codebook,
        )

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

    def _get_mixed_config(
        self, layer_idx: int, head_idx: int, calibration_vectors: Optional[torch.Tensor] = None
    ) -> Optional[MixedPrecisionConfig]:
        return self.config.get_mixed_config(layer_idx, head_idx, calibration_vectors)

    def store(self, layer_idx: int, head_idx: int, k_vec: torch.Tensor, v_vec: torch.Tensor):
        rotation = self.rotations[layer_idx][head_idx]
        S = self.qjl_matrices[layer_idx][head_idx]

        # For mixed precision, detect outliers from the first store
        mixed = None
        if self.config.mixed_precision:
            mixed = self._get_mixed_config(layer_idx, head_idx, k_vec)

        k_c = turboquant_encode_internal(k_vec, self.config.codebook, rotation, S, mixed=mixed)
        v_c = turboquant_encode_internal(v_vec, self.config.codebook, rotation, S, mixed=mixed)
        self.cache[layer_idx][head_idx].append((k_c, v_c))

    def store_batch(self, layer_idx: int, head_idx: int, k_vecs: torch.Tensor, v_vecs: torch.Tensor):
        rotation = self.rotations[layer_idx][head_idx]
        S = self.qjl_matrices[layer_idx][head_idx]

        # Detect outliers from the batch
        mixed = None
        if self.config.mixed_precision:
            mixed = self._get_mixed_config(layer_idx, head_idx, k_vecs)

        k_all = turboquant_encode_internal(k_vecs, self.config.codebook, rotation, S, mixed=mixed)
        v_all = turboquant_encode_internal(v_vecs, self.config.codebook, rotation, S, mixed=mixed)

        for i in range(k_vecs.shape[0]):
            k_single = TurboQuantCompressed(
                pq=PolarQuantCompressed(
                    norm=None if k_all.pq.norm is None else k_all.pq.norm[i:i+1],
                    indices=None if k_all.pq.indices is None else k_all.pq.indices[i:i+1],
                    codebook=k_all.pq.codebook,
                    rotation=k_all.pq.rotation,
                    original_dim=k_all.pq.original_dim,
                    regular_norm=None if k_all.pq.regular_norm is None else k_all.pq.regular_norm[i:i+1],
                    outlier_norm=None if k_all.pq.outlier_norm is None else k_all.pq.outlier_norm[i:i+1],
                    regular_indices=k_all.pq.regular_indices,
                    outlier_indices=k_all.pq.outlier_indices,
                    regular_quantized_indices=None if k_all.pq.regular_quantized_indices is None else k_all.pq.regular_quantized_indices[i:i+1],
                    outlier_quantized_indices=None if k_all.pq.outlier_quantized_indices is None else k_all.pq.outlier_quantized_indices[i:i+1],
                    codebook_regular=k_all.pq.codebook_regular,
                    codebook_outlier=k_all.pq.codebook_outlier,
                    rotation_regular=k_all.pq.rotation_regular,
                    rotation_outlier=k_all.pq.rotation_outlier,
                ),
                qjl=QJLCompressed(
                    signs=k_all.qjl.signs[i:i+1], r_norm=k_all.qjl.r_norm[i:i+1], S=S,
                ),
            )
            v_single = TurboQuantCompressed(
                pq=PolarQuantCompressed(
                    norm=None if v_all.pq.norm is None else v_all.pq.norm[i:i+1],
                    indices=None if v_all.pq.indices is None else v_all.pq.indices[i:i+1],
                    codebook=v_all.pq.codebook,
                    rotation=v_all.pq.rotation,
                    original_dim=v_all.pq.original_dim,
                    regular_norm=None if v_all.pq.regular_norm is None else v_all.pq.regular_norm[i:i+1],
                    outlier_norm=None if v_all.pq.outlier_norm is None else v_all.pq.outlier_norm[i:i+1],
                    regular_indices=v_all.pq.regular_indices,
                    outlier_indices=v_all.pq.outlier_indices,
                    regular_quantized_indices=None if v_all.pq.regular_quantized_indices is None else v_all.pq.regular_quantized_indices[i:i+1],
                    outlier_quantized_indices=None if v_all.pq.outlier_quantized_indices is None else v_all.pq.outlier_quantized_indices[i:i+1],
                    codebook_regular=v_all.pq.codebook_regular,
                    codebook_outlier=v_all.pq.codebook_outlier,
                    rotation_regular=v_all.pq.rotation_regular,
                    rotation_outlier=v_all.pq.rotation_outlier,
                ),
                qjl=QJLCompressed(
                    signs=v_all.qjl.signs[i:i+1], r_norm=v_all.qjl.r_norm[i:i+1], S=S,
                ),
            )
            self.cache[layer_idx][head_idx].append((k_single, v_single))

    def compute_attention(
        self, layer_idx: int, head_idx: int, q_vec: torch.Tensor,
        qjl_score_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute attention output using compressed KV cache.

        Args:
            qjl_score_weight: Weight for QJL inner product correction.
                1.0 = paper-correct unbiased estimator (Theorem 2).
                <1.0 = bias-variance tradeoff (lower variance, introduces bias).
                0.0 = PolarQuant-only scoring (no QJL correction).

        Note: Causal masking is not yet implemented. All stored KV tokens
        are attended to. For autoregressive generation, this is correct
        as long as tokens are stored in order.
        """
        d = self.d
        seq_len = len(self.cache[layer_idx][head_idx])
        if seq_len == 0:
            return torch.zeros(d, device=self.device)

        q_vec = q_vec.float()
        S = self.qjl_matrices[layer_idx][head_idx]
        qjl_scale = math.sqrt(math.pi / 2) / d

        q_proj = S @ q_vec

        # Use first token's mixed-precision info for batch decode
        first_pq = self.cache[layer_idx][head_idx][0][0].pq
        pq_batch = PolarQuantCompressed(
            norm=None if first_pq.norm is None else torch.stack([
                self.cache[layer_idx][head_idx][t][0].pq.norm.squeeze(0)
                for t in range(seq_len)
            ]),
            indices=None if first_pq.indices is None else torch.cat([
                self.cache[layer_idx][head_idx][t][0].pq.indices
                for t in range(seq_len)
            ], dim=0),
            codebook=first_pq.codebook,
            rotation=first_pq.rotation,
            original_dim=first_pq.original_dim,
            regular_norm=None if first_pq.regular_norm is None else torch.stack([
                self.cache[layer_idx][head_idx][t][0].pq.regular_norm.squeeze(0)
                for t in range(seq_len)
            ]),
            outlier_norm=None if first_pq.outlier_norm is None else torch.stack([
                self.cache[layer_idx][head_idx][t][0].pq.outlier_norm.squeeze(0)
                for t in range(seq_len)
            ]),
            regular_indices=first_pq.regular_indices,
            outlier_indices=first_pq.outlier_indices,
            regular_quantized_indices=None if first_pq.regular_quantized_indices is None else torch.cat([
                self.cache[layer_idx][head_idx][t][0].pq.regular_quantized_indices
                for t in range(seq_len)
            ], dim=0),
            outlier_quantized_indices=None if first_pq.outlier_quantized_indices is None else torch.cat([
                self.cache[layer_idx][head_idx][t][0].pq.outlier_quantized_indices
                for t in range(seq_len)
            ], dim=0),
            codebook_regular=first_pq.codebook_regular,
            codebook_outlier=first_pq.codebook_outlier,
            rotation_regular=first_pq.rotation_regular,
            rotation_outlier=first_pq.rotation_outlier,
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

        first_v_pq = self.cache[layer_idx][head_idx][0][1].pq
        v_pq_batch = PolarQuantCompressed(
            norm=None if first_v_pq.norm is None else torch.stack([
                self.cache[layer_idx][head_idx][t][1].pq.norm.squeeze(0)
                for t in range(seq_len)
            ]),
            indices=None if first_v_pq.indices is None else torch.cat([
                self.cache[layer_idx][head_idx][t][1].pq.indices
                for t in range(seq_len)
            ], dim=0),
            codebook=first_v_pq.codebook,
            rotation=first_v_pq.rotation,
            original_dim=first_v_pq.original_dim,
            regular_norm=None if first_v_pq.regular_norm is None else torch.stack([
                self.cache[layer_idx][head_idx][t][1].pq.regular_norm.squeeze(0)
                for t in range(seq_len)
            ]),
            outlier_norm=None if first_v_pq.outlier_norm is None else torch.stack([
                self.cache[layer_idx][head_idx][t][1].pq.outlier_norm.squeeze(0)
                for t in range(seq_len)
            ]),
            regular_indices=first_v_pq.regular_indices,
            outlier_indices=first_v_pq.outlier_indices,
            regular_quantized_indices=None if first_v_pq.regular_quantized_indices is None else torch.cat([
                self.cache[layer_idx][head_idx][t][1].pq.regular_quantized_indices
                for t in range(seq_len)
            ], dim=0),
            outlier_quantized_indices=None if first_v_pq.outlier_quantized_indices is None else torch.cat([
                self.cache[layer_idx][head_idx][t][1].pq.outlier_quantized_indices
                for t in range(seq_len)
            ], dim=0),
            codebook_regular=first_v_pq.codebook_regular,
            codebook_outlier=first_v_pq.codebook_outlier,
            rotation_regular=first_v_pq.rotation_regular,
            rotation_outlier=first_v_pq.rotation_outlier,
        )
        v_hat_all = polarquant_decode(v_pq_batch)

        output = (attn_weights.unsqueeze(-1) * v_hat_all.float()).sum(0)
        return output


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compression_ratio_fp16(
    d: int,
    b_mse: int = B_MSE,
    mixed_precision: bool = True,
    n_outlier: int = N_OUTLIER_CHANNELS,
    b_outlier: Optional[int] = None,
) -> float:
    """Compute paper-style headline compression ratio vs FP16."""
    fp16_bits = d * 16
    if mixed_precision:
        b_outlier = b_outlier if b_outlier is not None else b_mse + OUTLIER_EXTRA_BITS
        n_outlier = min(n_outlier, d)
        mse_bits = n_outlier * b_outlier + (d - n_outlier) * b_mse
    else:
        mse_bits = d * b_mse
    tq_bits = mse_bits
    return fp16_bits / tq_bits


def memory_bytes_per_vector(
    d: int,
    b_mse: int = B_MSE,
    mixed_precision: bool = True,
    n_outlier: int = N_OUTLIER_CHANNELS,
    b_outlier: Optional[int] = None,
) -> Tuple[int, int]:
    """Returns (TurboQuant bytes, FP16 bytes) per vector including norms and QJL."""
    if mixed_precision:
        b_outlier = b_outlier if b_outlier is not None else b_mse + OUTLIER_EXTRA_BITS
        n_outlier = min(n_outlier, d)
        mse_bits = n_outlier * b_outlier + (d - n_outlier) * b_mse
        pq_norm_bits = 32
    else:
        mse_bits = d * b_mse
        pq_norm_bits = 16
    tq_bits = mse_bits + pq_norm_bits + d * B_QJL + 16
    tq_bytes = (tq_bits + 7) // 8
    fp16_bytes = d * 2
    return tq_bytes, fp16_bytes


TurboQuantMSECompressed = PolarQuantCompressed
turboquant_mse_encode = polarquant_encode
turboquant_mse_decode = polarquant_decode
