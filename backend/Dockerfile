FROM python:3.11-slim

WORKDIR /app

ENV PORT=7860 \
    DEVICE=cpu \
    ENABLE_DIFFUSION=0 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/.cache/huggingface \
    TRANSFORMERS_CACHE=/data/.cache/huggingface/transformers \
    DIFFUSERS_CACHE=/data/.cache/huggingface/diffusers \
    SENTENCE_TRANSFORMERS_HOME=/data/.cache/huggingface/sentence-transformers \
    TOKENIZERS_PARALLELISM=false \
    CPU_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
