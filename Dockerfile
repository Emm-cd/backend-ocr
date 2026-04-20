FROM python:3.11-slim-bullseye

# ¡ESTA LÍNEA ES LA MAGIA! Obliga a Python a mostrar los logs en vivo
ENV PYTHONUNBUFFERED=1

# Instalamos Poppler y herramientas de video
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Usamos bash para asegurarnos de que la variable $PORT se lea bien
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]