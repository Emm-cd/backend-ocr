# 1. Usamos una base oficial de Python
FROM python:3.11-slim

# 2. Instalamos Poppler y otras librerías que OpenCV/EasyOCR suelen necesitar
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Le decimos dónde vamos a trabajar
WORKDIR /app

# 4. Copiamos los requerimientos y los instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiamos todo el resto de tu código
COPY . .

# 6. Exponemos el puerto de Railway
EXPOSE $PORT

# 7. Arrancamos el servidor
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}