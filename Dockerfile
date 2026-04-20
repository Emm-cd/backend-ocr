# Usamos la versión estable que resolvía el problema de libgl1
FROM python:3.11-slim-bullseye

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

# Usamos sh -c para que evalúe la variable $PORT correctamente como número
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]