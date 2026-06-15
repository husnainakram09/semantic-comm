"""
Receiver-side decoder for the Semantic Communication Demo.

The public API remains compatible with the frontend:
    decode_embedding(...) returns a PNG data URI plus caption/fidelity metadata.
"""

import base64
import io
import logging
import math
import os
import random
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

HF_HOME = os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
)
DIFFUSERS_CACHE = os.environ.setdefault(
    "DIFFUSERS_CACHE", os.path.join(HF_HOME, "diffusers")
)
SENTENCE_TRANSFORMERS_HOME = os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME", os.path.join(HF_HOME, "sentence-transformers")
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("CPU_NUM_THREADS", "2"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("CPU_NUM_THREADS", "2"))

DEVICE = "cpu"
SD_MODEL_ID = os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
SENTENCE_MODEL_ID = os.environ.get(
    "SENTENCE_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
)

IMAGE_SIZE = int(os.environ.get("DECODE_IMAGE_SIZE", "384"))
INFERENCE_STEPS = int(os.environ.get("SD_INFERENCE_STEPS", "8"))
GUIDANCE_SCALE = float(os.environ.get("SD_GUIDANCE_SCALE", "5.0"))
ENABLE_DIFFUSION = os.environ.get("ENABLE_DIFFUSION", "0").lower() in {
    "1",
    "true",
    "yes",
}

SNR_HIGH_THRESH = 20.0
SNR_MEDIUM_THRESH = 5.0
FRAC_MEDIUM = 0.60
FRAC_LOW = 0.30

_sd_pipe = None
_st_model = None
_sd_failed = False
_sd_lock = threading.Lock()
_st_lock = threading.Lock()


def _offline_mode() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes"}


def _load_sentence_transformer() -> None:
    global _st_model

    if _st_model is not None:
        return

    with _st_lock:
        if _st_model is not None:
            return

        logger.info("Loading decoder sentence-transformer '%s' on CPU.", SENTENCE_MODEL_ID)
        try:
            from sentence_transformers import SentenceTransformer

            _st_model = SentenceTransformer(
                SENTENCE_MODEL_ID,
                cache_folder=SENTENCE_TRANSFORMERS_HOME,
                device=DEVICE,
            )
            logger.info("Decoder sentence-transformer loaded from cache.")
        except Exception as exc:
            _st_model = None
            logger.warning("Cosine similarity disabled; model load failed: %s", exc)


def _load_stable_diffusion() -> bool:
    global _sd_pipe, _sd_failed

    if not ENABLE_DIFFUSION:
        return False
    if _sd_pipe is not None:
        return True
    if _sd_failed:
        return False

    with _sd_lock:
        if _sd_pipe is not None:
            return True
        if _sd_failed:
            return False

        logger.info("Loading diffusers pipeline '%s' on CPU.", SD_MODEL_ID)
        t0 = time.perf_counter()
        try:
            import torch
            from diffusers import StableDiffusionPipeline

            torch.set_num_threads(int(os.environ.get("CPU_NUM_THREADS", "2")))
            pipe = StableDiffusionPipeline.from_pretrained(
                SD_MODEL_ID,
                cache_dir=DIFFUSERS_CACHE,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True,
                local_files_only=_offline_mode(),
            )
            pipe.set_progress_bar_config(disable=True)
            pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            _sd_pipe = pipe.to(DEVICE)
            logger.info(
                "Diffusers pipeline ready in %.1f s on CPU.",
                time.perf_counter() - t0,
            )
            return True
        except Exception as exc:
            _sd_pipe = None
            _sd_failed = True
            logger.warning(
                "Diffusers pipeline unavailable; using lightweight CPU renderer: %s",
                exc,
            )
            return False


def _recover_caption(
    original_caption: str,
    snr_db: float,
    noisy_embedding: np.ndarray,
) -> tuple[str, str, float | None]:
    words = original_caption.split()
    n_words = len(words)

    if snr_db > SNR_HIGH_THRESH:
        keep = n_words
        fidelity = "high"
    elif snr_db > SNR_MEDIUM_THRESH:
        keep = max(1, math.ceil(n_words * FRAC_MEDIUM))
        fidelity = "medium"
    else:
        keep = max(1, math.ceil(n_words * FRAC_LOW))
        fidelity = "low"

    recovered = " ".join(words[:keep])
    cosine_sim = None
    _load_sentence_transformer()

    if _st_model is not None:
        try:
            clean_emb = _st_model.encode(
                original_caption,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            noisy_norm = noisy_embedding / (np.linalg.norm(noisy_embedding) + 1e-9)
            cosine_sim = float(np.clip(np.dot(clean_emb, noisy_norm), -1.0, 1.0))
        except Exception:
            logger.debug("Cosine-sim diagnostic failed.", exc_info=True)

    return recovered, fidelity, cosine_sim


def _fallback_image(caption: str, fidelity: str, seed: int | None) -> Image.Image:
    rng = random.Random(seed if seed is not None else hash(caption) & 0xFFFFFFFF)
    palettes = {
        "high": ((31, 122, 140), (249, 199, 79), (255, 255, 248)),
        "medium": ((87, 117, 144), (67, 170, 139), (250, 250, 246)),
        "low": ((73, 80, 87), (248, 150, 30), (246, 246, 240)),
    }
    bg, accent, paper = palettes.get(fidelity, palettes["medium"])
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), bg)
    draw = ImageDraw.Draw(img, "RGBA")

    for _ in range(18):
        x0 = rng.randint(-IMAGE_SIZE // 4, IMAGE_SIZE)
        y0 = rng.randint(-IMAGE_SIZE // 4, IMAGE_SIZE)
        radius = rng.randint(28, max(32, IMAGE_SIZE // 4))
        color = accent + (rng.randint(35, 95),)
        draw.ellipse((x0, y0, x0 + radius, y0 + radius), fill=color)

    margin = max(18, IMAGE_SIZE // 14)
    draw.rounded_rectangle(
        (margin, IMAGE_SIZE - margin * 5, IMAGE_SIZE - margin, IMAGE_SIZE - margin),
        radius=8,
        fill=paper + (230,),
    )

    font = ImageFont.load_default()
    words = caption.split()
    lines = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > 42:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)

    y = IMAGE_SIZE - margin * 4 + 4
    for text in lines[:4]:
        draw.text((margin + 10, y), text, fill=(30, 33, 36), font=font)
        y += 16

    return img


def _synthesise_image(
    caption: str,
    fidelity: str,
    seed: int | None = None,
) -> Image.Image:
    if not _load_stable_diffusion():
        return _fallback_image(caption, fidelity, seed)

    import torch

    generator = torch.Generator(device=DEVICE)
    if seed is not None:
        generator.manual_seed(seed)

    logger.info(
        "Diffusers inference prompt=%r steps=%d guidance=%.1f size=%d device=%s",
        caption,
        INFERENCE_STEPS,
        GUIDANCE_SCALE,
        IMAGE_SIZE,
        DEVICE,
    )
    with torch.inference_mode():
        output = _sd_pipe(
            prompt=caption,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator if seed is not None else None,
        )
    return output.images[0].convert("RGB")


def image_to_base64(pil_image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format=fmt.upper(), optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def decode_embedding(
    noisy_embedding: list[float] | np.ndarray,
    original_caption: str,
    snr_db: float,
    seed: int | None = None,
) -> dict:
    emb = np.asarray(noisy_embedding, dtype=np.float32)
    if emb.ndim != 1 or emb.size == 0:
        raise ValueError(
            f"noisy_embedding must be a non-empty 1-D array, got shape {emb.shape}."
        )

    original_caption = original_caption.strip()
    if not original_caption:
        raise ValueError("original_caption must not be empty or whitespace.")
    if not math.isfinite(snr_db):
        raise ValueError(f"snr_db must be a finite number, got {snr_db!r}.")

    recovered_caption, fidelity_level, cosine_sim = _recover_caption(
        original_caption=original_caption,
        snr_db=snr_db,
        noisy_embedding=emb,
    )
    pil_image = _synthesise_image(recovered_caption, fidelity_level, seed=seed)

    return {
        "reconstructed_image_b64": image_to_base64(pil_image, fmt="PNG"),
        "recovered_caption": recovered_caption,
        "fidelity_level": fidelity_level,
        "snr_db": float(snr_db),
        "cosine_similarity": round(cosine_sim, 6) if cosine_sim is not None else None,
        "image_size": list(pil_image.size),
        "inference_steps": INFERENCE_STEPS if ENABLE_DIFFUSION and _sd_pipe is not None else 0,
    }
