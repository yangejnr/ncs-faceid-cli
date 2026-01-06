FROM python:3.12-slim

# System libraries needed by: opencv-python (GUI/codec deps), faiss, and general runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better Docker layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

# Keep InsightFace models inside container filesystem (optional).
# If you prefer models persisted outside the container, mount a volume to /app/.models at runtime.
ENV INSIGHTFACE_HOME=/app/.models
ENV NO_ALBUMENTATIONS_UPDATE=1

# Install CLI
RUN pip install -e .

# Default command
CMD ["faceid", "--help"]
