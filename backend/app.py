"""
app.py
------
Flask REST API for the Semantic Communication Demo.

Endpoints
---------
POST /encode     – Encode an uploaded image → caption + semantic embedding.
POST /transmit   – Transmit a semantic embedding through a noisy AWGN channel.
POST /decode     – Reconstruct an image from a noisy semantic embedding via Stable Diffusion.
POST /metrics    – Compute PSNR / SSIM / BLEU between two images.
GET  /health     – Liveness check.
"""

import logging
import os

# Hugging Face Spaces listens on 7860 by default. Set cache locations before
# importing transformers/diffusers/sentence-transformers anywhere downstream.
os.environ.setdefault("PORT", "7860")
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
os.environ.setdefault("DIFFUSERS_CACHE", os.path.join(os.environ["HF_HOME"], "diffusers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(os.environ["HF_HOME"], "sentence-transformers"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("CPU_NUM_THREADS", "2"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("CPU_NUM_THREADS", "2"))

from flask import Flask, request, jsonify
from flask_cors import CORS

from encoder import encode_image
from channel import simulate_channel
from decoder import decode_embedding
from metrics import compute_all_metrics

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))
CORS(app, resources={r"/*": {"origins": "*"}})   # allow Vite dev server

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def bad_request(msg: str, status: int = 400):
    return jsonify({"error": msg}), status


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ------------------------------------------------------------------
@app.post("/encode")
def encode():
    """
    Encode an uploaded image into a semantic caption + embedding.

    Accepts : multipart/form-data  →  field "image"  (any image format)

    Returns (200):
        {
            "caption":     str,         – BLIP-generated semantic description
            "embedding":   [float, …],  – 384-d all-MiniLM-L6-v2 vector
            "token_count": int          – subword tokens in the caption
        }

    Error responses:
        400 – missing / empty / unreadable image field
        415 – uploaded file is not a recognisable image
        500 – model inference failure
    """
    # ── Validate multipart field ─────────────────────────────────────────────
    if "image" not in request.files:
        return bad_request(
            "No 'image' field found in the multipart request. "
            "Send the file under the key 'image'."
        )

    image_file = request.files["image"]

    if image_file.filename == "":
        return bad_request("The 'image' field is present but has an empty filename.")

    image_bytes = image_file.read()
    if not image_bytes:
        return bad_request("The uploaded file is empty (0 bytes).")

    # ── Validate MIME type (loose check before hitting the model) ────────────
    content_type: str = image_file.content_type or ""
    if content_type and not content_type.startswith("image/"):
        return (
            jsonify({
                "error": (
                    f"Unsupported content type '{content_type}'. "
                    "Please upload a JPEG, PNG, WebP, or other image file."
                )
            }),
            415,
        )

    # ── Run the semantic encoder ─────────────────────────────────────────────
    try:
        result = encode_image(image_bytes)
        logger.info(
            "Encoded '%s' → caption=%r  embedding_dim=%d  tokens=%d",
            image_file.filename,
            result["caption"],
            len(result["embedding"]),
            result["token_count"],
        )
        return jsonify(result), 200

    except ValueError as exc:
        # Image decoding failure (corrupt file, unsupported format, etc.)
        logger.warning("Image decode error for '%s': %s", image_file.filename, exc)
        return (
            jsonify({
                "error": (
                    f"Could not decode the uploaded file as an image: {exc}. "
                    "Ensure the file is a valid JPEG, PNG, or WebP."
                )
            }),
            415,
        )

    except RuntimeError as exc:
        # Model load or inference failure
        logger.exception("Model inference failed.")
        return bad_request(f"Encoding model error: {exc}", 500)

    except Exception as exc:
        logger.exception("Unexpected error during encoding.")
        return bad_request(f"Unexpected encoding error: {exc}", 500)


# ------------------------------------------------------------------
@app.post("/transmit")
def transmit():
    """
    Simulate a noisy AWGN wireless channel on a semantic embedding.

    Request (JSON)
    --------------
    {
        "embedding"    : [float, ...],   <- 384-d vector from /encode
        "snr_db"       : float,          <- e.g. 30.0 / 10.0 / 0.0
        "image_width"  : int,            <- original image width  (optional)
        "image_height" : int             <- original image height (optional)
    }

    Response (200)
    --------------
    {
        "noisy_embedding"   : [float, ...],
        "snr_db"            : float,
        "signal_power"      : float,
        "noise_power"       : float,
        "snr_linear"        : float,
        "embedding_dim"     : int,
        "channel_condition" : "high" | "medium" | "low" | "very_low",
        "bandwidth_savings" : {          <- only when image dims provided
            "traditional_bytes" : int,
            "semantic_bytes"    : int,
            "compression_ratio" : float,
            "savings_percent"   : float,
            "embedding_dim"     : int,
            "image_pixels"      : int
        } | null
    }

    Error responses
    ---------------
    400 - missing / malformed fields
    422 - embedding is not a flat list, or image dims are non-positive
    500 - unexpected runtime error
    """
    data = request.get_json(silent=True)
    if not data:
        return bad_request("JSON body required.")

    # -- Required fields ------------------------------------------------------
    embedding = data.get("embedding")
    if embedding is None:
        return bad_request("Missing required field 'embedding'.")
    if not isinstance(embedding, list) or len(embedding) == 0:
        return bad_request("'embedding' must be a non-empty JSON array of floats.")

    snr_db = data.get("snr_db")
    if snr_db is None:
        return bad_request("Missing required field 'snr_db'.")
    try:
        snr_db = float(snr_db)
    except (TypeError, ValueError):
        return bad_request(f"'snr_db' must be a number, got {snr_db!r}.")

    # -- Optional image dimensions --------------------------------------------
    image_width  = data.get("image_width")
    image_height = data.get("image_height")

    if image_width is not None:
        try:
            image_width = int(image_width)
            if image_width <= 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"error": "'image_width' must be a positive integer."}), 422

    if image_height is not None:
        try:
            image_height = int(image_height)
            if image_height <= 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"error": "'image_height' must be a positive integer."}), 422

    # -- Simulate channel -----------------------------------------------------
    try:
        result = simulate_channel(
            embedding    = embedding,
            snr_db       = snr_db,
            image_width  = image_width,
            image_height = image_height,
        )

        logger.info(
            "Transmit  SNR=%.1f dB (%s)  dim=%d  signal=%.6f  noise=%.6f  bw=%s",
            result["snr_db"],
            result["channel_condition"],
            result["embedding_dim"],
            result["signal_power"],
            result["noise_power"],
            (
                f"{result['bandwidth_savings']['compression_ratio']:.1f}x"
                if result["bandwidth_savings"] else "n/a"
            ),
        )

        return jsonify(result), 200

    except ValueError as exc:
        logger.warning("Transmit validation error: %s", exc)
        return jsonify({"error": str(exc)}), 422

    except Exception as exc:
        logger.exception("Unexpected error during channel simulation.")
        return bad_request(f"Channel simulation error: {exc}", 500)


# ------------------------------------------------------------------
@app.post("/decode")
def decode():
    """
    Reconstruct an image from a noisy semantic embedding.

    Request (JSON)
    --------------
    {
        "noisy_embedding"  : [float, ...],   <- corrupted vector from /transmit
        "original_caption" : str,            <- caption from /encode
        "snr_db"           : float,          <- channel SNR used in /transmit
        "seed"             : int             <- optional, for reproducibility
    }

    Response (200)
    --------------
    {
        "reconstructed_image_b64" : str,          <- PNG data URI
        "recovered_caption"       : str,          <- prompt fed to SD
        "fidelity_level"          : str,          <- "high" | "medium" | "low"
        "snr_db"                  : float,
        "cosine_similarity"       : float | null, <- clean vs noisy embedding
        "image_size"              : [int, int],   <- [512, 512]
        "inference_steps"         : int
    }

    Fidelity rules
    --------------
        SNR > 20 dB  -> full caption          (high)
        5 < SNR <= 20 -> first 60% of words   (medium)
        SNR <= 5     -> first 30% of words    (low)

    Error responses
    ---------------
    400 - missing or malformed required fields
    422 - semantic validation failure (empty caption, non-finite SNR, etc.)
    500 - model load or SD inference failure
    """
    data = request.get_json(silent=True)
    if not data:
        return bad_request("JSON body required.")

    # -- Required fields ------------------------------------------------------
    noisy_embedding = data.get("noisy_embedding")
    if noisy_embedding is None:
        return bad_request("Missing required field 'noisy_embedding'.")
    if not isinstance(noisy_embedding, list) or len(noisy_embedding) == 0:
        return bad_request("'noisy_embedding' must be a non-empty JSON array of floats.")

    original_caption = data.get("original_caption")
    if not original_caption or not isinstance(original_caption, str):
        return bad_request("Missing or invalid required field 'original_caption' (must be a non-empty string).")
    if not original_caption.strip():
        return bad_request("'original_caption' must not be blank.")

    snr_db = data.get("snr_db")
    if snr_db is None:
        return bad_request("Missing required field 'snr_db'.")
    try:
        snr_db = float(snr_db)
    except (TypeError, ValueError):
        return bad_request(f"'snr_db' must be a number, got {snr_db!r}.")

    # -- Optional fields ------------------------------------------------------
    seed = data.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            return bad_request(f"'seed' must be an integer, got {seed!r}.")

    # -- Run decoder ----------------------------------------------------------
    try:
        result = decode_embedding(
            noisy_embedding  = noisy_embedding,
            original_caption = original_caption,
            snr_db           = snr_db,
            seed             = seed,
        )

        logger.info(
            "Decode  SNR=%.1f dB  fidelity=%s  caption=%r  cosine_sim=%s",
            result["snr_db"],
            result["fidelity_level"],
            result["recovered_caption"],
            f"{result['cosine_similarity']:.4f}" if result["cosine_similarity"] is not None else "n/a",
        )

        return jsonify(result), 200

    except ValueError as exc:
        logger.warning("Decode validation error: %s", exc)
        return jsonify({"error": str(exc)}), 422

    except RuntimeError as exc:
        logger.exception("Model error during decoding.")
        return bad_request(f"Decoder model error: {exc}", 500)

    except Exception as exc:
        logger.exception("Unexpected error during decoding.")
        return bad_request(f"Unexpected decode error: {exc}", 500)


# ------------------------------------------------------------------
@app.post("/metrics")
def metrics():
    """
    Evaluate semantic communication quality across four metrics.

    Request: multipart/form-data
    ----------------------------
    original_image        (file)    – uploaded reference image
    reconstructed_image_b64 (str)  – PNG data URI from /decode
    original_caption      (str)    – caption from /encode
    recovered_caption     (str)    – truncated caption from /decode
    compression_ratio     (float)  – from /transmit bandwidth_savings

    Response (200)
    --------------
    {
        "ssim": float,                     [0, 1]
        "psnr": float,                     dB
        "bleu": float,                     [0, 1]
        "bandwidth": {
            "traditional_bytes"  : int,
            "semantic_bytes"     : int,
            "compression_ratio"  : float,
            "savings_percent"    : float,
            "image_resolution"   : [int, int]
        },
        "interpretation": {
            "ssim_label"  : str,
            "psnr_label"  : str,
            "bleu_label"  : str,
            "overall"     : str
        },
        "captions": {
            "original"  : str,
            "recovered" : str
        }
    }

    Error responses
    ---------------
    400 – missing / malformed fields
    415 – non-image file or malformed base64 URI
    500 – computation failure
    """
    # -- original_image: multipart file -----------------------------------------
    if "original_image" not in request.files:
        return bad_request(
            "Missing 'original_image' file field in multipart form data."
        )
    orig_file = request.files["original_image"]
    if orig_file.filename == "":
        return bad_request("'original_image' field has an empty filename.")
    orig_bytes = orig_file.read()
    if not orig_bytes:
        return bad_request("'original_image' file is empty (0 bytes).")

    # -- reconstructed_image_b64: form text field --------------------------------
    recon_b64 = request.form.get("reconstructed_image_b64", "").strip()
    if not recon_b64:
        return bad_request(
            "Missing 'reconstructed_image_b64' form field "
            "(expected a data:image/…;base64,… string)."
        )
    if not recon_b64.startswith("data:image/"):
        return (
            jsonify({
                "error": (
                    "'reconstructed_image_b64' must be a base64 data URI "
                    "(e.g. data:image/png;base64,…)."
                )
            }),
            415,
        )

    # -- text fields -------------------------------------------------------------
    original_caption = request.form.get("original_caption", "").strip()
    if not original_caption:
        return bad_request("Missing or empty 'original_caption' form field.")

    recovered_caption = request.form.get("recovered_caption", "").strip()
    if not recovered_caption:
        return bad_request("Missing or empty 'recovered_caption' form field.")

    compression_ratio_raw = request.form.get("compression_ratio")
    if compression_ratio_raw is None:
        return bad_request("Missing 'compression_ratio' form field.")
    try:
        compression_ratio = float(compression_ratio_raw)
        if compression_ratio <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return bad_request(
            f"'compression_ratio' must be a positive number, "
            f"got {compression_ratio_raw!r}."
        )

    # -- Decode the reconstructed base64 URI to bytes ----------------------------
    try:
        _header, encoded = recon_b64.split(",", 1)
        import base64 as _b64
        recon_bytes = _b64.b64decode(encoded)
    except Exception as exc:
        return (
            jsonify({"error": f"Could not decode 'reconstructed_image_b64': {exc}"}),
            415,
        )

    # -- Compute all metrics -----------------------------------------------------
    try:
        result = compute_all_metrics(
            original_img_bytes = orig_bytes,
            recon_img_bytes    = recon_bytes,
            orig_caption       = original_caption,
            recov_caption      = recovered_caption,
            compression_ratio  = compression_ratio,
        )

        logger.info(
            "Metrics: SSIM=%.4f (%s)  PSNR=%.2f dB (%s)  "
            "BLEU=%.4f (%s)  BW-saving=%.2f%%",
            result["ssim"],
            result["interpretation"]["ssim_label"],
            result["psnr"],
            result["interpretation"]["psnr_label"],
            result["bleu"],
            result["interpretation"]["bleu_label"],
            result["bandwidth"]["savings_percent"],
        )

        return jsonify(result), 200

    except ValueError as exc:
        logger.warning("Metrics decode error: %s", exc)
        return jsonify({"error": str(exc)}), 415

    except Exception as exc:
        logger.exception("Unexpected error computing metrics.")
        return bad_request(f"Metrics computation error: {exc}", 500)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)
