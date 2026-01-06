FROM python:3.12-slim

# Runtime deps for OpenCV + FAISS + general
# Build deps for insightface (needs g++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    build-essential \
    g++ \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

ENV INSIGHTFACE_HOME=/app/.models
ENV NO_ALBUMENTATIONS_UPDATE=1

RUN pip install -e .

CMD ["faceid", "--help"]
