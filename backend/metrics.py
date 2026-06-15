"""
metrics.py
----------
Evaluates Semantic Communication quality against traditional pixel
transmission across four complementary dimensions:

  SSIM  – Structural Similarity Index     [0, 1]        (higher = better)
  PSNR  – Peak Signal-to-Noise Ratio      [0, ∞) dB     (higher = better)
  BLEU  – Caption n-gram overlap          [0, 1]        (higher = better)
  BW    – Bandwidth savings %             [0, 100)      (higher = better)

Image metrics (SSIM, PSNR)
---------------------------
Both images are decoded from their source format, converted to uint8 RGB,
and resized to a common resolution (the original's dimensions) before any
pixel-level comparison. This is necessary because the SD-generated
reconstruction is always 512×512 while the original can be any size.

Text metric (BLEU)
------------------
BLEU is computed between the encoder's original caption (reference) and
the decoder's recovered/truncated caption (hypothesis) using NLTK's
sentence_bleu with uniform 1–4-gram weights and smoothing to handle short
hypotheses that would otherwise score zero due to missing n-grams.

Bandwidth savings
-----------------
  traditional_bytes = H × W × 3        (raw uncompressed RGB)
  semantic_bytes    = embedding_dim × 4  (384 float32 values = 1,536 bytes)
  savings           = (1 − semantic / traditional) × 100

Public API
----------
compute_ssim(original_img_bytes, reconstructed_img_bytes)           → float
compute_psnr(original_img_bytes, reconstructed_img_bytes)           → float
compute_bleu(reference_caption, hypothesis_caption)                 → float
compute_all_metrics(original_img_bytes, recon_img_bytes,
                    orig_caption, recov_caption,
                    compression_ratio)                               → dict
"""

import base64
import io
import logging
import re
import sys

import numpy as np
from PIL import Image
from skimage.metrics import (
    peak_signal_noise_ratio as _skimage_psnr,
    structural_similarity   as _skimage_ssim,
)
from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NLTK data is not downloaded at runtime on Spaces.
# ---------------------------------------------------------------------------

# Avoid runtime downloads on Hugging Face Spaces cold starts. BLEU uses a
# lightweight regex tokenizer below, so punkt data is not required.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bytes_to_array(img_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes (JPEG, PNG, WebP, …) to a uint8 NumPy array
    with shape (H, W, 3) in RGB colour order.

    Raises
    ------
    ValueError  if the bytes cannot be decoded as an image.
    """
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Cannot decode image bytes: {exc}") from exc


def _b64_uri_to_array(data_uri: str) -> np.ndarray:
    """
    Decode a base64 data URI (``data:image/…;base64,…``) to a uint8
    NumPy array with shape (H, W, 3).

    Raises
    ------
    ValueError  if the URI is malformed or the payload cannot be decoded.
    """
    try:
        _header, encoded = data_uri.split(",", 1)
        img_bytes = base64.b64decode(encoded)
    except Exception as exc:
        raise ValueError(
            f"Malformed base64 data URI — could not extract payload: {exc}"
        ) from exc
    return _bytes_to_array(img_bytes)


def _align_arrays(
    orig: np.ndarray,
    recon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize *recon* to match *orig*'s (H, W) if the shapes differ.

    The original's resolution is used as the reference so that PSNR / SSIM
    are always measured in the original image's pixel space.

    Uses bilinear interpolation — a neutral choice that neither sharpens
    nor excessively blurs the resized image.
    """
    if orig.shape == recon.shape:
        return orig, recon

    h, w = orig.shape[:2]
    logger.debug(
        "Resizing reconstruction from %s to (%d, %d) for pixel comparison.",
        recon.shape[:2], h, w,
    )
    recon_pil = Image.fromarray(recon).resize((w, h), Image.BILINEAR)
    return orig, np.array(recon_pil, dtype=np.uint8)


def _tokenise(text: str) -> list[str]:
    """
    Tokenise *text* into a list of lowercase word tokens.

    Tries the NLTK word tokeniser first (handles punctuation properly);
    falls back to a simple ``str.split()`` if NLTK data is unavailable
    (e.g. in a fully offline environment).
    """
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------------
# 1. SSIM
# ---------------------------------------------------------------------------

def compute_ssim(
    original_img_bytes: bytes,
    reconstructed_img_bytes: bytes,
) -> float:
    """
    Compute the Structural Similarity Index between two images.

    SSIM measures perceptual similarity across luminance, contrast, and
    structure — it correlates better with human visual judgment than MSE.

    Parameters
    ----------
    original_img_bytes        : raw bytes of the reference image.
    reconstructed_img_bytes   : raw bytes of the reconstructed image.

    Returns
    -------
    float in [−1, 1]; practically [0, 1] for natural images.
    1.0 = identical, 0.0 = no structural similarity.
    """
    orig  = _bytes_to_array(original_img_bytes)
    recon = _bytes_to_array(reconstructed_img_bytes)
    orig, recon = _align_arrays(orig, recon)

    score = float(
        _skimage_ssim(orig, recon, channel_axis=2, data_range=255)
    )
    logger.debug("SSIM = %.6f", score)
    return score


# ---------------------------------------------------------------------------
# 2. PSNR
# ---------------------------------------------------------------------------

def compute_psnr(
    original_img_bytes: bytes,
    reconstructed_img_bytes: bytes,
) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio between two images.

    PSNR = 10 · log₁₀(MAX² / MSE)

    Interpretation (rough guide for 8-bit images):
        ≥ 40 dB  — excellent, visually lossless
        30–40 dB — good quality
        20–30 dB — noticeable artefacts
        < 20 dB  — poor / heavy degradation

    Parameters
    ----------
    original_img_bytes        : raw bytes of the reference image.
    reconstructed_img_bytes   : raw bytes of the reconstructed image.

    Returns
    -------
    float in dB.  Returns ``inf`` when MSE = 0 (identical images).
    """
    orig  = _bytes_to_array(original_img_bytes)
    recon = _bytes_to_array(reconstructed_img_bytes)
    orig, recon = _align_arrays(orig, recon)

    score = float(
        _skimage_psnr(orig, recon, data_range=255)
    )
    logger.debug("PSNR = %.4f dB", score)
    return score


# ---------------------------------------------------------------------------
# 3. BLEU
# ---------------------------------------------------------------------------

def compute_bleu(
    reference_caption: str,
    hypothesis_caption: str,
) -> float:
    """
    Compute the BLEU score between a reference and hypothesis caption.

    Uses uniform 1–4-gram weights (BLEU-4) with NLTK's method1 smoothing
    so that short hypotheses (which lack 3-/4-gram matches) don't
    automatically receive a score of zero.

    Parameters
    ----------
    reference_caption  : the original caption produced by the encoder.
    hypothesis_caption : the recovered (possibly truncated) caption from
                         the decoder.

    Returns
    -------
    float in [0, 1].
    1.0 = perfect match, 0.0 = no n-gram overlap.
    """
    ref_tokens  = _tokenise(reference_caption.strip())
    hyp_tokens  = _tokenise(hypothesis_caption.strip())

    if not ref_tokens or not hyp_tokens:
        logger.warning(
            "compute_bleu: empty token list — ref=%r  hyp=%r",
            ref_tokens, hyp_tokens,
        )
        return 0.0

    # Uniform 1-to-4-gram weights; method1 adds epsilon counts to avoid
    # log(0) when a higher-order gram has zero matches.
    smoothing = SmoothingFunction().method1
    score = float(
        sentence_bleu(
            references  = [ref_tokens],
            hypothesis  = hyp_tokens,
            weights     = (0.25, 0.25, 0.25, 0.25),
            smoothing_function = smoothing,
        )
    )
    logger.debug(
        "BLEU = %.6f  |  ref_tokens=%d  hyp_tokens=%d",
        score, len(ref_tokens), len(hyp_tokens),
    )
    return score


# ---------------------------------------------------------------------------
# 4. Bandwidth savings (helper, also exposed publicly)
# ---------------------------------------------------------------------------

def _compute_bandwidth_savings(
    original_img_bytes:  bytes,
    compression_ratio:   float,
) -> dict:
    """
    Compute bandwidth savings of semantic vs traditional transmission.

    Parameters
    ----------
    original_img_bytes : raw bytes of the uploaded original image.
    compression_ratio  : traditional_bytes / semantic_bytes
                         (passed in from channel.bandwidth_savings()).

    Returns
    -------
    dict
        {
            "traditional_bytes"   : int,
            "semantic_bytes"      : int,
            "compression_ratio"   : float,
            "savings_percent"     : float,
            "image_resolution"    : [int, int],   # [W, H]
        }
    """
    img  = Image.open(io.BytesIO(original_img_bytes)).convert("RGB")
    w, h = img.size
    traditional_bytes = w * h * 3               # raw uncompressed RGB

    # Back-compute semantic_bytes from the ratio so we stay consistent
    # with the figure already sent by channel.py.
    semantic_bytes = int(round(traditional_bytes / compression_ratio)) \
        if compression_ratio > 0 else 0

    savings_percent = (
        (1.0 - semantic_bytes / traditional_bytes) * 100.0
        if traditional_bytes > 0 else 0.0
    )

    return {
        "traditional_bytes": traditional_bytes,
        "semantic_bytes":    semantic_bytes,
        "compression_ratio": round(compression_ratio, 4),
        "savings_percent":   round(savings_percent,   4),
        "image_resolution":  [w, h],
    }


# ---------------------------------------------------------------------------
# 5. Master function
# ---------------------------------------------------------------------------

def compute_all_metrics(
    original_img_bytes:      bytes,
    recon_img_bytes:         bytes,
    orig_caption:            str,
    recov_caption:           str,
    compression_ratio:       float,
) -> dict:
    """
    Compute all four quality metrics in a single call.

    Parameters
    ----------
    original_img_bytes   : raw bytes of the reference image (upload).
    recon_img_bytes      : raw bytes of the reconstructed image.
                           May be decoded from a base64 URI by the caller.
    orig_caption         : caption generated by the encoder.
    recov_caption        : caption used by the decoder (possibly truncated).
    compression_ratio    : traditional_bytes / semantic_bytes
                           (from channel.simulate_channel → bandwidth_savings).

    Returns
    -------
    dict
        {
            "ssim": float,             [0, 1]
            "psnr": float,             dB
            "bleu": float,             [0, 1]
            "bandwidth": {
                "traditional_bytes"  : int,
                "semantic_bytes"     : int,
                "compression_ratio"  : float,
                "savings_percent"    : float,
                "image_resolution"   : [int, int],
            },
            "interpretation": {
                "ssim_label"   : str,  "excellent" | "good" | "fair" | "poor"
                "psnr_label"   : str,  "excellent" | "good" | "fair" | "poor"
                "bleu_label"   : str,  "high" | "moderate" | "low"
                "overall"      : str,  human-readable summary sentence
            },
            "captions": {
                "original"   : str,
                "recovered"  : str,
            },
        }
    """
    # ── Compute the four metrics ─────────────────────────────────────────────
    ssim_val = compute_ssim(original_img_bytes, recon_img_bytes)
    psnr_val = compute_psnr(original_img_bytes, recon_img_bytes)
    bleu_val = compute_bleu(orig_caption, recov_caption)
    bw_dict  = _compute_bandwidth_savings(original_img_bytes, compression_ratio)

    # ── Interpretation labels ────────────────────────────────────────────────
    ssim_label = (
        "excellent" if ssim_val >= 0.90 else
        "good"      if ssim_val >= 0.70 else
        "fair"      if ssim_val >= 0.40 else
        "poor"
    )
    psnr_label = (
        "excellent" if psnr_val >= 40.0 else
        "good"      if psnr_val >= 30.0 else
        "fair"      if psnr_val >= 20.0 else
        "poor"
    )
    bleu_label = (
        "high"     if bleu_val >= 0.60 else
        "moderate" if bleu_val >= 0.30 else
        "low"
    )

    overall = (
        f"Semantic transmission achieved {bw_dict['savings_percent']:.1f}% bandwidth "
        f"reduction with {ssim_label} structural similarity "
        f"(SSIM={ssim_val:.3f}, PSNR={psnr_val:.1f} dB) and "
        f"{bleu_label} semantic fidelity (BLEU={bleu_val:.3f})."
    )

    logger.info(
        "Metrics: SSIM=%.4f (%s)  PSNR=%.2f dB (%s)  "
        "BLEU=%.4f (%s)  BW-saving=%.2f%%",
        ssim_val, ssim_label,
        psnr_val, psnr_label,
        bleu_val, bleu_label,
        bw_dict["savings_percent"],
    )

    return {
        "ssim":   round(ssim_val, 6),
        "psnr":   round(psnr_val, 4),
        "bleu":   round(bleu_val, 6),
        "bandwidth":     bw_dict,
        "interpretation": {
            "ssim_label": ssim_label,
            "psnr_label": psnr_label,
            "bleu_label": bleu_label,
            "overall":    overall,
        },
        "captions": {
            "original":  orig_caption,
            "recovered": recov_caption,
        },
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        stream=sys.stdout,
    )

    print("=" * 64)
    print("  metrics.py  —  self-test")
    print("=" * 64)

    # ── Build synthetic test images ──────────────────────────────────────────
    rng = np.random.default_rng(42)

    def _make_img_bytes(arr: np.ndarray, fmt="PNG") -> bytes:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format=fmt)
        return buf.getvalue()

    # Reference: random 256×256 RGB image
    orig_arr  = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)

    # Near-identical: tiny Gaussian noise on top
    near_arr  = np.clip(
        orig_arr.astype(np.int16) + rng.integers(-5, 6, orig_arr.shape),
        0, 255
    ).astype(np.uint8)

    # Heavily degraded: large noise
    noisy_arr = np.clip(
        orig_arr.astype(np.int16) + rng.integers(-80, 81, orig_arr.shape),
        0, 255
    ).astype(np.uint8)

    # Different size: simulate SD 512×512 output
    sd_arr    = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)

    orig_bytes  = _make_img_bytes(orig_arr)
    near_bytes  = _make_img_bytes(near_arr)
    noisy_bytes = _make_img_bytes(noisy_arr)
    sd_bytes    = _make_img_bytes(sd_arr)

    print(f"\nOriginal image   : 256×256  ({len(orig_bytes):,} bytes PNG)")
    print(f"Near-identical   : 256×256  ({len(near_bytes):,} bytes PNG)")
    print(f"Heavily degraded : 256×256  ({len(noisy_bytes):,} bytes PNG)")
    print(f"SD output (diff size) : 512×512  ({len(sd_bytes):,} bytes PNG)")

    # ── 1. compute_ssim ──────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  1. compute_ssim()")
    print("─" * 64)
    ssim_near  = compute_ssim(orig_bytes, near_bytes)
    ssim_noisy = compute_ssim(orig_bytes, noisy_bytes)
    ssim_sd    = compute_ssim(orig_bytes, sd_bytes)          # cross-size
    print(f"  near-identical  : {ssim_near:.6f}   (expect close to 1.0)")
    print(f"  heavily noisy   : {ssim_noisy:.6f}  (expect lower)")
    print(f"  512×512 SD img  : {ssim_sd:.6f}  (after resize)")
    assert ssim_near > ssim_noisy, "SSIM ordering violated!"
    print("  ✓ SSIM ordering correct")

    # ── 2. compute_psnr ──────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  2. compute_psnr()")
    print("─" * 64)
    psnr_near  = compute_psnr(orig_bytes, near_bytes)
    psnr_noisy = compute_psnr(orig_bytes, noisy_bytes)
    psnr_sd    = compute_psnr(orig_bytes, sd_bytes)
    print(f"  near-identical  : {psnr_near:.4f} dB  (expect high, ~30+ dB)")
    print(f"  heavily noisy   : {psnr_noisy:.4f} dB  (expect lower)")
    print(f"  512×512 SD img  : {psnr_sd:.4f} dB  (after resize)")
    assert psnr_near > psnr_noisy, "PSNR ordering violated!"
    print("  ✓ PSNR ordering correct")

    # ── 3. compute_bleu ──────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  3. compute_bleu()")
    print("─" * 64)
    caption_full    = "a fluffy orange cat sitting on a wooden windowsill looking outside"
    caption_partial = "a fluffy orange cat sitting on a"     # 60% — medium fidelity
    caption_short   = "a fluffy orange cat"                  # 30% — low fidelity
    caption_diff    = "a sports car driving on a highway at sunset"

    cases = [
        ("identical",    caption_full,    caption_full),
        ("60% retained", caption_full,    caption_partial),
        ("30% retained", caption_full,    caption_short),
        ("unrelated",    caption_full,    caption_diff),
    ]
    for label, ref, hyp in cases:
        score = compute_bleu(ref, hyp)
        print(f"  {label:<16} BLEU={score:.6f}  ref={len(ref.split())}w  hyp={len(hyp.split())}w")

    bleu_full    = compute_bleu(caption_full, caption_full)
    bleu_partial = compute_bleu(caption_full, caption_partial)
    bleu_short   = compute_bleu(caption_full, caption_short)
    assert bleu_full >= bleu_partial >= bleu_short, "BLEU ordering violated!"
    print("  ✓ BLEU ordering correct")

    # ── 4. Edge cases ────────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  4. Edge cases")
    print("─" * 64)
    bleu_empty = compute_bleu("hello world", "")
    print(f"  empty hypothesis   BLEU={bleu_empty:.6f}  (expect 0.0)")
    assert bleu_empty == 0.0, "Empty hypothesis should give 0.0"
    print("  ✓ empty hypothesis → 0.0")

    # ── 5. compute_all_metrics ───────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  5. compute_all_metrics()")
    print("─" * 64)
    result = compute_all_metrics(
        original_img_bytes = orig_bytes,
        recon_img_bytes    = noisy_bytes,
        orig_caption       = caption_full,
        recov_caption      = caption_partial,
        compression_ratio  = 600.0,             # 256×256×3 / (384×4) ≈ 512
    )

    print(json.dumps({
        "ssim":     result["ssim"],
        "psnr":     result["psnr"],
        "bleu":     result["bleu"],
        "bandwidth": {
            k: result["bandwidth"][k]
            for k in ("traditional_bytes","semantic_bytes",
                      "compression_ratio","savings_percent")
        },
        "interpretation": result["interpretation"],
    }, indent=2))

    assert "ssim"            in result
    assert "psnr"            in result
    assert "bleu"            in result
    assert "bandwidth"       in result
    assert "interpretation"  in result
    assert "captions"        in result
    assert result["captions"]["original"]  == caption_full
    assert result["captions"]["recovered"] == caption_partial
    print("\n  ✓ All keys present and captions round-trip correctly")

    print("\n✓ metrics.py self-test passed.")
