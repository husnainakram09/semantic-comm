FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/

ENV PORT=7860
ENV DEVICE=cpu
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "app.py"]