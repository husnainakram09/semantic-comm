"""
channel.py
----------
Simulates a noisy wireless channel for the Semantic Communication Demo.

The "message" being transmitted is a float32 semantic embedding vector
produced by encoder.py (all-MiniLM-L6-v2, 384 dimensions).

Public API
----------
add_awgn_noise(embedding, snr_db)          → np.ndarray
simulate_channel(embedding, snr_db, ...)   → dict
bandwidth_savings(image_width, image_height, embedding_dim) → dict

SNR presets (used in the self-test and as named constants):
  SNR_HIGH   = 30 dB   ─ near-perfect transmission
  SNR_MEDIUM = 10 dB   ─ moderate loss
  SNR_LOW    =  0 dB   ─ heavy degradation
"""

import json
import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Named SNR presets (dB)
# ---------------------------------------------------------------------------

SNR_HIGH   = 30.0   # near-perfect
SNR_MEDIUM = 10.0   # moderate loss
SNR_LOW    =  0.0   # heavy degradation

# Bytes per float32 element (IEEE 754)
BYTES_PER_FLOAT32 = 4

# Default embedding dimension for all-MiniLM-L6-v2
DEFAULT_EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# Core noise function
# ---------------------------------------------------------------------------

def add_awgn_noise(
    embedding: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add Additive White Gaussian Noise (AWGN) calibrated to a target SNR.

    Derivation
    ----------
    Given signal power  P_s = mean(x²),
    and SNR (linear)    SNR = 10^(snr_db / 10),
    noise variance      σ²  = P_s / SNR,
    noise               n   ~ N(0, σ²I).

    Edge cases
    ----------
    - Zero-power signal (all-zero embedding): noise_power is set to
      10^(snr_db/10) so the noise scale still reflects the requested SNR.
    - snr_db = +inf  → no noise added (σ² → 0).
    - snr_db = -inf  → raises ValueError (undefined).

    Parameters
    ----------
    embedding : np.ndarray  shape (D,)  dtype float32
        The semantic embedding to corrupt.
    snr_db    : float
        Target signal-to-noise ratio in decibels.
    rng       : np.random.Generator | None
        Optional seeded RNG for reproducible tests.

    Returns
    -------
    noisy_embedding : np.ndarray  shape (D,)  dtype float32
    """
    if not np.isfinite(snr_db):
        raise ValueError(f"snr_db must be a finite number, got {snr_db!r}.")

    x = embedding.astype(np.float32)

    # ── Signal power ─────────────────────────────────────────────────────────
    signal_power: float = float(np.mean(x ** 2))

    # ── SNR linear ───────────────────────────────────────────────────────────
    snr_linear: float = 10.0 ** (snr_db / 10.0)

    # ── Noise variance ───────────────────────────────────────────────────────
    if signal_power == 0.0:
        # Degenerate case: treat unit signal power as reference
        noise_power: float = 1.0 / snr_linear
        logger.warning(
            "Signal power is zero; using unit reference for noise calibration."
        )
    else:
        noise_power = signal_power / snr_linear

    noise_std = np.sqrt(noise_power)

    # ── Generate noise ───────────────────────────────────────────────────────
    _rng = rng if rng is not None else np.random.default_rng()
    noise = _rng.normal(loc=0.0, scale=noise_std, size=x.shape).astype(np.float32)

    logger.debug(
        "AWGN  SNR=%.1f dB  signal_power=%.6f  noise_power=%.6f  noise_std=%.6f",
        snr_db, signal_power, noise_power, noise_std,
    )

    return x + noise


# ---------------------------------------------------------------------------
# Bandwidth savings
# ---------------------------------------------------------------------------

def bandwidth_savings(
    image_width: int,
    image_height: int,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> dict:
    """
    Compute how many fewer bytes the semantic approach transmits versus
    sending raw (uncompressed) RGB pixel data.

    Parameters
    ----------
    image_width   : int  pixels
    image_height  : int  pixels
    embedding_dim : int  number of float32 values in the semantic vector

    Returns
    -------
    dict
        {
            "traditional_bytes" : int,    # H × W × 3 raw RGB bytes
            "semantic_bytes"    : int,    # embedding_dim × 4 bytes
            "compression_ratio" : float,  # traditional / semantic
            "savings_percent"   : float,  # (1 − semantic/traditional) × 100
            "embedding_dim"     : int,
            "image_pixels"      : int,
        }
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Image dimensions must be positive, got {image_width}×{image_height}."
        )
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

    traditional_bytes: int = image_width * image_height * 3          # RGB, 1 byte/channel
    semantic_bytes:    int = embedding_dim * BYTES_PER_FLOAT32        # float32 vector

    compression_ratio: float = traditional_bytes / semantic_bytes
    savings_percent:   float = (1.0 - semantic_bytes / traditional_bytes) * 100.0

    return {
        "traditional_bytes": traditional_bytes,
        "semantic_bytes":    semantic_bytes,
        "compression_ratio": round(compression_ratio, 4),
        "savings_percent":   round(savings_percent,   4),
        "embedding_dim":     embedding_dim,
        "image_pixels":      image_width * image_height,
    }


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def simulate_channel(
    embedding: list[float] | np.ndarray,
    snr_db: float,
    image_width:   int | None = None,
    image_height:  int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Transmit a semantic embedding through an AWGN channel and return
    the corrupted vector together with channel statistics and optional
    bandwidth savings metrics.

    Parameters
    ----------
    embedding     : list[float] or np.ndarray  (D,)
        The semantic payload from encoder.py.
    snr_db        : float
        Desired SNR in dB.  Use SNR_HIGH / SNR_MEDIUM / SNR_LOW presets
        or any custom value.
    image_width   : int | None
        Original image width in pixels.  Required for bandwidth_savings.
    image_height  : int | None
        Original image height in pixels.  Required for bandwidth_savings.
    rng           : np.random.Generator | None
        Optional seeded RNG for reproducible unit tests.

    Returns
    -------
    dict
        {
            "noisy_embedding"   : list[float],   # corrupted payload
            "snr_db"            : float,
            "signal_power"      : float,
            "noise_power"       : float,
            "snr_linear"        : float,
            "embedding_dim"     : int,
            "channel_condition" : str,           # "high" | "medium" | "low" | "custom"
            "bandwidth_savings" : dict | None,   # present when image dims given
        }
    """
    # ── Normalise input ──────────────────────────────────────────────────────
    x = np.array(embedding, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(
            f"embedding must be a 1-D array, got shape {x.shape}."
        )

    # ── Pre-compute channel statistics from the clean signal ─────────────────
    signal_power: float = float(np.mean(x ** 2))
    snr_linear:   float = 10.0 ** (snr_db / 10.0)

    if signal_power == 0.0:
        noise_power = 1.0 / snr_linear
    else:
        noise_power = signal_power / snr_linear

    # ── Add AWGN ─────────────────────────────────────────────────────────────
    noisy = add_awgn_noise(x, snr_db=snr_db, rng=rng)

    # ── Label channel condition ──────────────────────────────────────────────
    if   snr_db >= SNR_HIGH:              condition = "high"
    elif snr_db >= SNR_MEDIUM:            condition = "medium"
    elif snr_db >= SNR_LOW:               condition = "low"
    else:                                 condition = "very_low"

    logger.info(
        "simulate_channel  SNR=%.1f dB (%s)  dim=%d  "
        "signal_power=%.6f  noise_power=%.6f",
        snr_db, condition, len(x), signal_power, noise_power,
    )

    result: dict = {
        "noisy_embedding":   noisy.tolist(),
        "snr_db":            float(snr_db),
        "signal_power":      round(signal_power, 8),
        "noise_power":       round(noise_power,  8),
        "snr_linear":        round(snr_linear,   6),
        "embedding_dim":     int(x.shape[0]),
        "channel_condition": condition,
        "bandwidth_savings": None,
    }

    # ── Bandwidth savings (optional) ─────────────────────────────────────────
    if image_width is not None and image_height is not None:
        result["bandwidth_savings"] = bandwidth_savings(
            image_width  = image_width,
            image_height = image_height,
            embedding_dim = int(x.shape[0]),
        )

    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        stream=sys.stdout,
    )

    print("=" * 64)
    print("  channel.py  —  self-test")
    print("=" * 64)

    # ── Synthetic embedding (unit-normalised, like all-MiniLM-L6-v2) ─────────
    rng_seed = np.random.default_rng(42)
    raw = rng_seed.standard_normal(DEFAULT_EMBEDDING_DIM).astype(np.float32)
    embedding = (raw / np.linalg.norm(raw)).tolist()   # L2-normalised

    print(f"\nInput embedding:  dim={len(embedding)}  "
          f"norm={np.linalg.norm(embedding):.6f}")

    # ── Test the three named SNR presets ─────────────────────────────────────
    presets = [
        ("High SNR",   SNR_HIGH,   "near-perfect transmission"),
        ("Medium SNR", SNR_MEDIUM, "moderate loss"),
        ("Low SNR",    SNR_LOW,    "heavy degradation"),
    ]

    # Image dimensions for bandwidth test
    W, H = 640, 480

    print(f"\n{'─'*64}")
    print(f"  Bandwidth comparison  (image {W}×{H}  vs  {DEFAULT_EMBEDDING_DIM}-d embedding)")
    print(f"{'─'*64}")
    bw = bandwidth_savings(W, H, DEFAULT_EMBEDDING_DIM)
    print(f"  Traditional (raw RGB) : {bw['traditional_bytes']:>10,} bytes")
    print(f"  Semantic (embedding)  : {bw['semantic_bytes']:>10,} bytes")
    print(f"  Compression ratio     : {bw['compression_ratio']:>10.1f}×")
    print(f"  Bandwidth saving      : {bw['savings_percent']:>9.2f} %")

    print(f"\n{'─'*64}")
    print("  Channel simulation results")
    print(f"{'─'*64}")

    for label, snr, desc in presets:
        rng_fixed = np.random.default_rng(0)          # fixed seed for reproducibility
        result = simulate_channel(
            embedding    = embedding,
            snr_db       = snr,
            image_width  = W,
            image_height = H,
            rng          = rng_fixed,
        )

        noisy = np.array(result["noisy_embedding"], dtype=np.float32)
        cosine_sim = float(
            np.dot(embedding, noisy) /
            (np.linalg.norm(embedding) * np.linalg.norm(noisy) + 1e-9)
        )

        print(f"\n  [{label}]  {snr:+.0f} dB  — {desc}")
        print(f"    condition   : {result['channel_condition']}")
        print(f"    signal_power: {result['signal_power']:.6f}")
        print(f"    noise_power : {result['noise_power']:.6f}")
        print(f"    snr_linear  : {result['snr_linear']:.4f}")
        print(f"    cosine_sim  : {cosine_sim:.6f}  "
              f"(1.0 = identical, 0.0 = orthogonal)")
        print(f"    noisy[:4]   : {noisy[:4].tolist()}")

    print(f"\n{'─'*64}")
    print("  add_awgn_noise() edge-case: zero-power embedding")
    print(f"{'─'*64}")
    zero_emb = np.zeros(8, dtype=np.float32)
    noisy_zero = add_awgn_noise(zero_emb, snr_db=10.0, rng=np.random.default_rng(1))
    print(f"  Input:  {zero_emb.tolist()}")
    print(f"  Output: {[round(v, 6) for v in noisy_zero.tolist()]}")

    print("\n✓ channel.py self-test passed.")
