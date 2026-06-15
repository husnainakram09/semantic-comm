"""
Semantic image encoder for the Semantic Communication Demo.

Public API:
    encode_image(image_bytes) -> {
        "caption": str,
        "embedding": list[float],
        "token_count": int,
    }
"""

import io
import logging
import os
import threading

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

HF_HOME = os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
)
TRANSFORMERS_CACHE = os.environ.setdefault(
    "TRANSFORMERS_CACHE", os.path.join(HF_HOME, "transformers")
)
SENTENCE_TRANSFORMERS_HOME = os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME", os.path.join(HF_HOME, "sentence-transformers")
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("CPU_NUM_THREADS", "2"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("CPU_NUM_THREADS", "2"))

DEVICE = "cpu"
BLIP_MODEL_ID = os.environ.get("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-base")
SENTENCE_MODEL_ID = os.environ.get(
    "SENTENCE_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
)
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "768"))

_blip_processor = None
_blip_model = None
_sentence_model = None
_blip_lock = threading.Lock()
_sentence_lock = threading.Lock()


def _offline_mode() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes"}


def _load_blip() -> None:
    global _blip_processor, _blip_model

    if _blip_processor is not None and _blip_model is not None:
        return

    with _blip_lock:
        if _blip_processor is not None and _blip_model is not None:
            return

        logger.info("Loading BLIP captioning model '%s' on CPU.", BLIP_MODEL_ID)
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            torch.set_num_threads(int(os.environ.get("CPU_NUM_THREADS", "2")))
            _blip_processor = BlipProcessor.from_pretrained(
                BLIP_MODEL_ID,
                cache_dir=TRANSFORMERS_CACHE,
                local_files_only=_offline_mode(),
            )
            _blip_model = BlipForConditionalGeneration.from_pretrained(
                BLIP_MODEL_ID,
                cache_dir=TRANSFORMERS_CACHE,
                low_cpu_mem_usage=True,
                local_files_only=_offline_mode(),
            ).to(DEVICE)
            _blip_model.eval()
            logger.info("BLIP model loaded from cache on CPU.")
        except Exception as exc:
            _blip_processor = None
            _blip_model = None
            logger.exception("Failed to load BLIP model.")
            raise RuntimeError(f"BLIP load error: {exc}") from exc


def _load_sentence_transformer() -> None:
    global _sentence_model

    if _sentence_model is not None:
        return

    with _sentence_lock:
        if _sentence_model is not None:
            return

        logger.info("Loading sentence-transformer '%s' on CPU.", SENTENCE_MODEL_ID)
        try:
            from sentence_transformers import SentenceTransformer

            _sentence_model = SentenceTransformer(
                SENTENCE_MODEL_ID,
                cache_folder=SENTENCE_TRANSFORMERS_HOME,
                device=DEVICE,
            )
            logger.info("Sentence-transformer loaded from cache.")
        except Exception as exc:
            _sentence_model = None
            logger.exception("Failed to load sentence-transformer model.")
            raise RuntimeError(f"SentenceTransformer load error: {exc}") from exc


def _prepare_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
        return image
    except Exception as exc:
        raise ValueError(f"Cannot decode image bytes: {exc}") from exc


def _generate_caption(image: Image.Image) -> str:
    _load_blip()

    import torch

    inputs = _blip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output_ids = _blip_model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=3,
            early_stopping=True,
        )

    caption = _blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def _embed_caption(caption: str) -> np.ndarray:
    _load_sentence_transformer()

    embedding = _sentence_model.encode(
        caption,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embedding, dtype=np.float32)


def _count_tokens(caption: str) -> int:
    if _blip_processor is not None:
        try:
            token_ids = _blip_processor.tokenizer.encode(
                caption, add_special_tokens=False
            )
            return len(token_ids)
        except Exception:
            logger.debug("BLIP tokenizer token count failed.", exc_info=True)

    return len(caption.split())


def encode_image(image_bytes: bytes) -> dict:
    image = _prepare_image(image_bytes)
    logger.debug("Image decoded for BLIP: size=%s, mode=%s", image.size, image.mode)

    caption = _generate_caption(image)
    embedding = _embed_caption(caption)

    return {
        "caption": caption,
        "embedding": embedding.tolist(),
        "token_count": _count_tokens(caption),
    }
