FROM python:3.11-slim-bullseye

# ── Logs en vivo ──────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1

# ── Dependencias del sistema ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

# ── Directorio de trabajo ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dependencias Python ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Código fuente ─────────────────────────────────────────────────────────────
COPY . .

# ── Arranque ──────────────────────────────────────────────────────────────────
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]