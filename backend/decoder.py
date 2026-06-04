"""
decoder.py
----------
Receiver-side decoder for the Semantic Communication Demo.

Optimized for CPU-only environments (HF Spaces free tier).

Two-step reconstruction pipeline
---------------------------------
1.  Semantic error correction  (caption recovery)
    ─────────────────────────────────────────────
    The channel SNR determines how much of the original caption survived:

        SNR > 20 dB  →  full caption                       ("high"   fidelity)
        5 < SNR ≤ 20 →  first 60 % of words               ("medium" fidelity)
        SNR ≤ 5      →  first 30 % of words               ("low"    fidelity)

    Alongside the threshold rule we compute the cosine similarity between
    the clean caption embedding and the received noisy embedding.  This
    similarity is logged and included in the response as a diagnostic —
    it confirms that the SNR threshold and the actual embedding degradation
    agree.

2.  Text-to-image synthesis  (Stable Diffusion v1.5)
    ──────────────────────────────────────────────────
    The recovered caption is fed to runwayml/stable-diffusion-v1-5
    (via HuggingFace diffusers) to regenerate a 512 × 512 image.

Model loading strategy
----------------------
Both the SD pipeline and the sentence-transformer are module-level
singletons loaded at app startup to prevent timeout on first request.

Public API
----------
decode_embedding(noisy_embedding, original_caption, snr_db, seed) → dict
image_to_base64(pil_image, fmt) → str
ensure_decoder_models_loaded() → None
"""

import base64
import io
import logging
import math
import os
import sys
import time
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Force CPU-only
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set Hugging Face cache directory
HF_HOME = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ['HF_HOME'] = HF_HOME


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SD_MODEL_ID     = "runwayml/stable-diffusion-v1-5"
IMAGE_SIZE      = 512         # SD v1.5 native resolution
INFERENCE_STEPS = 15          # Reduced for CPU (was 20, now 15 for faster inference)
GUIDANCE_SCALE  = 7.5         # classifier-free guidance strength

# SNR decision thresholds (dB)
SNR_HIGH_THRESH   = 20.0      # above → full caption
SNR_MEDIUM_THRESH =  5.0      # above → 60 %; at/below → 30 %

# Caption word-retention fractions per fidelity level
FRAC_HIGH   = 1.00
FRAC_MEDIUM = 0.60
FRAC_LOW    = 0.30

# Torch device: CPU-only for HF Spaces
DEVICE: str = "cpu"


# ---------------------------------------------------------------------------
# Module-level model cache  (load once at startup, reuse across every request)
# ---------------------------------------------------------------------------

_sd_pipe:   object | None = None   # StableDiffusionPipeline
_st_model:  object | None = None   # SentenceTransformer  (cosine-sim diagnostic)
_models_loaded: bool = False        # Flag to track initialization


def _load_stable_diffusion() -> None:
    """
    Load the Stable Diffusion v1.5 pipeline into the module-level cache.
    Called once at app startup via ensure_decoder_models_loaded().
    """
    global _sd_pipe

    if _sd_pipe is not None:
        return                          # already loaded

    logger.info(
        "Loading Stable Diffusion pipeline '%s' on device='%s' …",
        SD_MODEL_ID, DEVICE,
    )
    t0 = time.perf_counter()

    try:
        import torch
        from diffusers import StableDiffusionPipeline

        # CPU uses float32 (no float16 support like CUDA)
        dtype = torch.float32

        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,            # disabled for research demo
            requires_safety_checker=False,
            low_cpu_mem_usage=True,         # Enable memory-efficient loading
        ).to(DEVICE)

        # CPU memory optimization: enable attention slicing
        pipe.enable_attention_slicing()
        logger.info("Attention slicing enabled for CPU memory efficiency.")

        _sd_pipe = pipe
        logger.info(
            "✓ Stable Diffusion ready  (%.1f s, dtype=%s, device=%s)",
            time.perf_counter() - t0, dtype, DEVICE,
        )

    except Exception as exc:
        logger.exception("Failed to load Stable Diffusion pipeline.")
        raise RuntimeError(f"Stable Diffusion load failed: {exc}") from exc


def _load_sentence_transformer() -> None:
    """
    Load all-MiniLM-L6-v2 for computing cosine-similarity diagnostics.
    Non-fatal: if the model cannot be loaded the cosine-sim field is
    simply omitted from the response rather than crashing the pipeline.
    """
    global _st_model

    if _st_model is not None:
        return

    logger.info("Loading sentence-transformer (all-MiniLM-L6-v2) for decoder diagnostics…")
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        _st_model.to(DEVICE)
        logger.info("✓ Sentence-transformer loaded (decoder)")
    except Exception as exc:
        logger.warning(
            "Could not load sentence-transformer (%s). "
            "Cosine-sim diagnostics will be unavailable.", exc,
        )


def ensure_decoder_models_loaded() -> None:
    """
    Public function to ensure all decoder models are loaded.
    Call this once on app startup to prevent timeout on first request.
    """
    global _models_loaded
    
    if _models_loaded:
        return
    
    logger.info("Loading decoder models…")
    _load_stable_diffusion()
    _load_sentence_transformer()
    _models_loaded = True
    logger.info("✓ All decoder models loaded successfully")


# ---------------------------------------------------------------------------
# Step 1 — Caption recovery (semantic error correction)
# ---------------------------------------------------------------------------

def _recover_caption(
    original_caption: str,
    snr_db:           float,
    noisy_embedding:  np.ndarray,
) -> tuple[str, str, float | None]:
    """
    Decide how much of the original caption to retain based on channel SNR.

    The logic mirrors how a real semantic receiver would degrade gracefully:
    at high SNR the full meaning survived; at low SNR only the most
    prominent semantic tokens are recoverable.

    Additionally, if the sentence-transformer is available we compute the
    cosine similarity between the clean caption embedding and the received
    noisy embedding — this is a ground-truth measure of semantic preservation
    and is returned as a diagnostic alongside the fidelity label.

    Parameters
    ----------
    original_caption : str
        Caption produced by the encoder (transmitted as metadata or
        pre-shared between sender and receiver — standard in SemComm).
    snr_db           : float
        Channel SNR in dB at the receiver.
    noisy_embedding  : np.ndarray  shape (D,)
        The received (noisy) embedding vector.

    Returns
    -------
    (recovered_caption, fidelity_level, cosine_similarity)
        fidelity_level    : "high" | "medium" | "low"
        cosine_similarity : float in [-1, 1] or None if ST unavailable
    """
    words = original_caption.split()
    n     = len(words)

    # ── SNR-driven truncation ─────────────────────────────────────────────────
    if snr_db > SNR_HIGH_THRESH:
        keep     = n                                 # retain everything
        fidelity = "high"
    elif snr_db > SNR_MEDIUM_THRESH:
        keep     = max(1, math.ceil(n * FRAC_MEDIUM))
        fidelity = "medium"
    else:
        keep     = max(1, math.ceil(n * FRAC_LOW))
        fidelity = "low"

    recovered = " ".join(words[:keep])

    logger.info(
        "Caption recovery  SNR=%.1f dB → fidelity=%s  "
        "words retained: %d / %d  recovered=%r",
        snr_db, fidelity, keep, n, recovered,
    )

    # ── Cosine-similarity diagnostic ──────────────────────────────────────────
    cosine_sim: float | None = None
    _load_sentence_transformer()

    if _st_model is not None:
        try:
            clean_emb: np.ndarray = _st_model.encode(
                original_caption,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

            # Normalise the noisy embedding to the unit sphere before comparison
            noisy_norm = noisy_embedding / (np.linalg.norm(noisy_embedding) + 1e-9)
            cosine_sim = float(np.clip(np.dot(clean_emb, noisy_norm), -1.0, 1.0))

            logger.debug(
                "Cosine similarity (clean caption emb vs noisy rx emb): %.4f",
                cosine_sim,
            )
        except Exception as diag_exc:
            logger.debug("Cosine-sim diagnostic failed (non-fatal): %s", diag_exc)

    return recovered, fidelity, cosine_sim


# ---------------------------------------------------------------------------
# Step 2 — Image synthesis (Stable Diffusion v1.5)
# ---------------------------------------------------------------------------

def _synthesise_image(
    caption: str,
    seed:    int | None = None,
) -> Image.Image:
    """
    Generate a 512 × 512 RGB image from a text prompt using SD v1.5.

    Parameters
    ----------
    caption : str
        The (possibly truncated) recovered caption used as the prompt.
    seed    : int | None
        Optional RNG seed for reproducible outputs.

    Returns
    -------
    PIL.Image.Image  (RGB, 512 × 512)
    """
    _load_stable_diffusion()          # no-op after first call

    import torch

    generator = (
        torch.Generator(device=DEVICE).manual_seed(seed)
        if seed is not None
        else None
    )

    logger.info(
        "SD inference  prompt=%r  steps=%d  guidance=%.1f  device=%s",
        caption, INFERENCE_STEPS, GUIDANCE_SCALE, DEVICE,
    )
    t0 = time.perf_counter()

    with torch.inference_mode():
        output = _sd_pipe(
            prompt              = caption,
            height              = IMAGE_SIZE,
            width               = IMAGE_SIZE,
            num_inference_steps = INFERENCE_STEPS,
            guidance_scale      = GUIDANCE_SCALE,
            generator           = generator,
        )

    img: Image.Image = output.images[0]
    logger.info(
        "Image generated in %.1f s  size=%s  mode=%s",
        time.perf_counter() - t0, img.size, img.mode,
    )
    return img


# ---------------------------------------------------------------------------
# Public helper — image serialisation
# ---------------------------------------------------------------------------

def image_to_base64(pil_image: Image.Image, fmt: str = "PNG") -> str:
    """
    Encode a PIL image as a base64 data URI ready for JSON transport.

    Parameters
    ----------
    pil_image : PIL.Image.Image
    fmt       : "PNG"  — lossless, recommended for research output
                "JPEG" — smaller, acceptable for previews

    Returns
    -------
    str  e.g.  "data:image/png;base64,iVBORw0KGgo…"
    """
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format=fmt.upper())
    b64  = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def decode_embedding(
    noisy_embedding:  list[float] | np.ndarray,
    original_caption: str,
    snr_db:           float,
    seed:             int | None = None,
) -> dict:
    """
    Full receiver pipeline: noisy embedding → recovered caption → image.

    Parameters
    ----------
    noisy_embedding  : list[float] or np.ndarray  shape (D,)
        The semantic vector after passing through the noisy channel
        (output of channel.simulate_channel).
    original_caption : str
        Caption generated by the encoder.  In a real system this is
        transmitted as compact side-information (or pre-agreed at both
        ends) — a standard assumption in semantic communication research.
    snr_db           : float
        The channel SNR in dB used during transmission.
    seed             : int | None
        Optional RNG seed for reproducible image generation.

    Returns
    -------
    dict
        {
            "reconstructed_image_b64" : str,          # PNG data URI
            "recovered_caption"       : str,          # prompt used for SD
            "fidelity_level"          : str,          # "high"|"medium"|"low"
            "snr_db"                  : float,
            "cosine_similarity"       : float | None, # clean vs noisy emb
            "image_size"              : [int, int],   # [W, H] = [512, 512]
            "inference_steps"         : int,
        }

    Raises
    ------
    ValueError
        Bad input shapes or empty/blank caption.
    RuntimeError
        Model load failure or SD inference crash.
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    emb = np.array(noisy_embedding, dtype=np.float32)
    if emb.ndim != 1 or emb.size == 0:
        raise ValueError(
            f"noisy_embedding must be a non-empty 1-D array, got shape {emb.shape}."
        )

    original_caption = original_caption.strip()
    if not original_caption:
        raise ValueError("original_caption must not be empty or whitespace.")

    if not math.isfinite(snr_db):
        raise ValueError(f"snr_db must be a finite number, got {snr_db!r}.")

    # ── Step 1: semantic error correction ─────────────────────────────────────
    recovered_caption, fidelity_level, cosine_sim = _recover_caption(
        original_caption = original_caption,
        snr_db           = snr_db,
        noisy_embedding  = emb,
    )

    # ── Step 2: text-to-image synthesis ───────────────────────────────────────
    pil_image = _synthesise_image(recovered_caption, seed=seed)

    # ── Step 3: serialise output ──────────────────────────────────────────────
    b64_uri = image_to_base64(pil_image, fmt="PNG")

    return {
        "reconstructed_image_b64": b64_uri,
        "recovered_caption":       recovered_caption,
        "fidelity_level":          fidelity_level,
        "snr_db":                  float(snr_db),
        "cosine_similarity":       round(cosine_sim, 6) if cosine_sim is not None else None,
        "image_size":              list(pil_image.size),   # [W, H]
        "inference_steps":         INFERENCE_STEPS,
    }


# ---------------------------------------------------------------------------
# Self-test  (python decoder.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        stream=sys.stdout,
    )

    print("=" * 64)
    print("  decoder.py  —  self-test")
    print("=" * 64)

    # ── Build a synthetic noisy embedding (unit-sphere, MiniLM shape) ─────────
    _rng  = np.random.default_rng(42)
    _raw  = _rng.standard_normal(384).astype(np.float32)
    noisy = (_raw / np.linalg.norm(_raw)).tolist()

    SAMPLE_CAPTION = (
        "a fluffy orange cat sitting on a wooden windowsill looking outside"
    )

    print(f"\nOriginal caption : {SAMPLE_CAPTION!r}")
    print(f"Embedding dim    : {len(noisy)}")

    # ── 1. Caption recovery across all three fidelity levels ─────────────────
    print()
    print("─" * 64)
    print("  Step 1 — Caption recovery (fidelity logic, no SD required)")
    print("─" * 64)

    emb_arr = np.array(noisy, dtype=np.float32)
    test_cases = [
        (25.0, "high"),
        (12.0, "medium"),
        (2.0,  "low"),
    ]

    all_ok = True
    for snr_test, expected in test_cases:
        recovered, fidelity, cosine = _recover_caption(
            SAMPLE_CAPTION, snr_test, emb_arr
        )
        ok   = fidelity == expected
        mark = "✓" if ok else "✗"
        if not ok:
            all_ok = False
        cos_str = f"  cosine_sim={cosine:.4f}" if cosine is not None else ""
        print(
            f"  {mark} SNR={snr_test:>5.1f} dB  fidelity={fidelity:<8}"
            f"  words={len(recovered.split())}  {cos_str}"
        )
        print(f"    recovered: {recovered!r}")

    assert all_ok, "One or more fidelity assertions failed!"

    # ── 2. image_to_base64 smoke test ─────────────────────────────────────────
    print()
    print("─" * 64)
    print("  Step 2 — image_to_base64() smoke test")
    print("─" * 64)

    dummy_img = Image.new("RGB", (512, 512), color=(34, 197, 94))  # signal green
    for fmt_test in ("PNG", "JPEG"):
        uri = image_to_base64(dummy_img, fmt=fmt_test)
        expected_prefix = f"data:image/{fmt_test.lower()};base64,"
        assert uri.startswith(expected_prefix), f"Wrong prefix for {fmt_test}"
        approx_kb = len(uri) * 3 / 4 / 1024
        print(f"  ✓ {fmt_test:<5}  URI length={len(uri):,} chars  (~{approx_kb:.1f} KB)")

    # ── 3. Full pipeline test (only runs if diffusers is installed) ───────────
    print()
    print("─" * 64)
    print("  Step 3 — Full decode_embedding() pipeline")
    print("─" * 64)

    try:
        import diffusers        # noqa: F401 — just checking availability
        import torch            # noqa: F401

        print("  diffusers + torch found — running SD inference (may take a while)…")
        result = decode_embedding(
            noisy_embedding  = noisy,
            original_caption = SAMPLE_CAPTION,
            snr_db           = 12.0,        # medium fidelity
            seed             = 0,
        )

        assert result["fidelity_level"]    == "medium"
        assert result["image_size"]        == [512, 512]
        assert result["inference_steps"]   == INFERENCE_STEPS
        assert result["reconstructed_image_b64"].startswith("data:image/png;base64,")

        print(f"  ✓ fidelity_level    : {result['fidelity_level']}")
        print(f"  ✓ recovered_caption : {result['recovered_caption']!r}")
        print(f"  ✓ image_size        : {result['image_size']}")
        print(f"  ✓ cosine_similarity : {result['cosine_similarity']}")
        print(f"  ✓ b64 prefix        : {result['reconstructed_image_b64'][:48]}…")

    except ImportError as imp_exc:
        print(f"  ⚠  Skipping SD test — missing dependency: {imp_exc}")
        print("     Install with:  pip install diffusers transformers accelerate")

    # ── 4. Input validation ───────────────────────────────────────────────────
    print()
    print("─" * 64)
    print("  Step 4 — Input validation guards")
    print("─" * 64)

    guards = [
        ("empty embedding",  lambda: decode_embedding([], SAMPLE_CAPTION, 10.0)),
        ("blank caption",    lambda: decode_embedding(noisy, "   ",        10.0)),
        ("inf snr_db",       lambda: decode_embedding(noisy, SAMPLE_CAPTION, float("inf"))),
    ]

    for label, fn in guards:
        try:
            fn()
            print(f"  ✗ {label}: expected ValueError but none raised!")
            all_ok = False
        except ValueError as ve:
            print(f"  ✓ {label}: raised ValueError — {ve}")

    print()
    print("✓ decoder.py self-test passed.")
