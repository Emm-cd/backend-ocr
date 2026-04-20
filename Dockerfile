FROM python:3.11-slim-bullseye

ENV PYTHONUNBUFFERED=1
ENV EASYOCR_MODULE_PATH=/app/.EasyOCR

RUN apt-get update && apt-get install -y \
    poppler-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ← Descarga modelos durante el BUILD (no en runtime)
RUN python -c "import easyocr; easyocr.Reader(['es', 'en'], gpu=False)"

COPY . .
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]