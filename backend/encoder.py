"""
encoder.py
----------
Semantic image encoder for the Semantic Communication Demo.

Pipeline
--------
1. BLIP (Salesforce/blip-image-captioning-base)
   Generates a natural-language caption that captures the *meaning* of
   the image — this is the semantic payload we want to transmit.

2. Sentence-Transformers (all-MiniLM-L6-v2)
   Encodes the caption into a dense float32 embedding vector (384-dim).
   This compact representation is what gets transmitted over the noisy
   channel (see channel.py).

Public API
----------
encode_image(image_bytes: bytes) -> dict
    {
        "caption":     str,          # human-readable semantic description
        "embedding":   list[float],  # 384-d semantic vector
        "token_count": int,          # number of tokens in the caption
    }
"""

import io
import logging
import os
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Device selection ────────────────────────────────────────────────────────
DEVICE: str = os.environ.get("DEVICE", "cpu")


# ── Lazy-loaded model holders ───────────────────────────────────────────────
_blip_processor  = None
_blip_model      = None
_sentence_model  = None


# ---------------------------------------------------------------------------
# Private loaders (called once, then cached in module globals)
# ---------------------------------------------------------------------------

def _load_blip() -> None:
    """Load BLIP captioning processor + model on first use."""
    global _blip_processor, _blip_model

    if _blip_processor is not None:
        return  # already loaded

    logger.info("Loading BLIP captioning model (Salesforce/blip-image-captioning-base)…")
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration

        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(DEVICE).eval()

        logger.info("BLIP model loaded on device='%s'.", DEVICE)

    except Exception as exc:
        logger.exception("Failed to load BLIP model.")
        raise RuntimeError(f"BLIP load error: {exc}") from exc


def _load_sentence_transformer() -> None:
    """Load the sentence-transformer embedding model on first use."""
    global _sentence_model

    if _sentence_model is not None:
        return  # already loaded

    logger.info("Loading sentence-transformer (all-MiniLM-L6-v2)…")
    try:
        from sentence_transformers import SentenceTransformer

        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Sentence-transformer loaded.")

    except Exception as exc:
        logger.exception("Failed to load sentence-transformer model.")
        raise RuntimeError(f"SentenceTransformer load error: {exc}") from exc


# ---------------------------------------------------------------------------
# Caption generation
# ---------------------------------------------------------------------------

def _generate_caption(image: Image.Image) -> str:
    """
    Run BLIP on a PIL RGB image and return the generated caption string.

    Parameters
    ----------
    image : PIL.Image.Image  (RGB)

    Returns
    -------
    caption : str
        e.g. "a dog sitting on a wooden floor next to a window"
    """
    _load_blip()

    import torch

    inputs = _blip_processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = _blip_model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4,
            early_stopping=True,
        )

    caption: str = _blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def _embed_caption(caption: str) -> np.ndarray:
    """
    Encode a caption string into a 384-dimensional float32 numpy vector
    using the all-MiniLM-L6-v2 sentence-transformer.

    Parameters
    ----------
    caption : str

    Returns
    -------
    embedding : np.ndarray  shape (384,)  dtype float32
    """
    _load_sentence_transformer()

    embedding: np.ndarray = _sentence_model.encode(
        caption,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit-sphere → cosine similarity = dot product
    ).astype(np.float32)

    return embedding


# ---------------------------------------------------------------------------
# Token count helper
# ---------------------------------------------------------------------------

def _count_tokens(caption: str) -> int:
    """
    Count word-level tokens in the caption.
    Uses the BLIP tokenizer if available, otherwise falls back to
    a simple whitespace split.
    """
    if _blip_processor is not None:
        try:
            token_ids = _blip_processor.tokenizer.encode(
                caption, add_special_tokens=False
            )
            return len(token_ids)
        except Exception:
            pass  # fall through to whitespace fallback

    return len(caption.split())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_image(image_bytes: bytes) -> dict:
    """
    Full semantic encoding pipeline: image → caption → embedding.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of an uploaded image file (JPEG, PNG, WebP, …).

    Returns
    -------
    result : dict
        {
            "caption":     str,          # BLIP-generated semantic caption
            "embedding":   list[float],  # 384-d float32 semantic vector
            "token_count": int,          # subword token count of caption
        }

    Raises
    ------
    ValueError
        If image_bytes cannot be decoded as an image.
    RuntimeError
        If a model fails to load or inference fails.
    """
    # ── 1. Decode image ─────────────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Cannot decode image bytes: {exc}") from exc

    logger.debug("Image decoded: size=%s, mode=%s", image.size, image.mode)

    # ── 2. Generate caption ─────────────────────────────────────────────────
    caption = _generate_caption(image)
    logger.info("Caption generated: %r", caption)

    # ── 3. Embed caption ────────────────────────────────────────────────────
    embedding: np.ndarray = _embed_caption(caption)
    logger.debug(
        "Embedding shape=%s  norm=%.4f",
        embedding.shape,
        float(np.linalg.norm(embedding)),
    )

    # ── 4. Count tokens ─────────────────────────────────────────────────────
    token_count = _count_tokens(caption)

    return {
        "caption":     caption,
        "embedding":   embedding.tolist(),   # JSON-serialisable list[float]
        "token_count": token_count,
    }


# ---------------------------------------------------------------------------
# Initialization for startup (HF Spaces optimization)
# ---------------------------------------------------------------------------

def initialize_encoder_models():
    """
    Pre-load all encoder models at app startup.
    This avoids cold-start delays on the first /encode request.
    """
    logger.info("Pre-loading BLIP and SentenceTransformer models…")
    _load_blip()
    _load_sentence_transformer()
    logger.info("✓ Encoder models loaded successfully")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json
    import urllib.request

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    # ── Load a sample image ──────────────────────────────────────────────────
    # Use a local path if provided, otherwise fetch a small public test image.
    if len(sys.argv) > 1:
        sample_path = sys.argv[1]
        print(f"Loading local image: {sample_path}")
        with open(sample_path, "rb") as f:
            img_bytes = f.read()
    else:
        SAMPLE_URL = (
            "https://huggingface.co/datasets/huggingface/documentation-images"
            "/resolve/main/pipeline-cat-chonk.jpeg"
        )
        print(f"Fetching sample image from:\n  {SAMPLE_URL}\n")
        with urllib.request.urlopen(SAMPLE_URL) as resp:   # noqa: S310
            img_bytes = resp.read()

    print(f"Image size: {len(img_bytes):,} bytes\n")
    print("Running encode_image()…\n")

    # ── Run the encoder ──────────────────────────────────────────────────────
    result = encode_image(img_bytes)

    # ── Pretty-print summary ─────────────────────────────────────────────────
    emb = result["embedding"]
    summary = {
        "caption":          result["caption"],
        "token_count":      result["token_count"],
        "embedding_dim":    len(emb),
        "embedding_dtype":  "float32",
        "embedding_norm":   round(float(np.linalg.norm(emb)), 6),
        "embedding_preview": [round(v, 6) for v in emb[:8]] + ["…"],
    }

    print(json.dumps(summary, indent=2))
    print("\n✓ encoder.py self-test passed.")
