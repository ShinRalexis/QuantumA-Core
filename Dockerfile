FROM python:3.10-slim

WORKDIR /app

# Nessuna installazione apt-get per minimizzare l'uso del disco
COPY requirements.txt .

# FIX: Usiamo --extra-index-url per permettere a pip di scaricare pacchetti da più sorgenti
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

COPY . .

ENV QUANTUM_API_PORT=8227
ENV QUANTUM_API_HOST=0.0.0.0

EXPOSE 8227

CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8227"]
