# Usamos una versión un poco más robusta pero igual de ligera
FROM python:3.11-slim-bullseye

# Instalamos Poppler y la versión correcta de libgl1
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}