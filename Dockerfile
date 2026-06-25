# Use a lightweight Debian-based Python image
FROM python:3.14-alpine

ARG GENEO_HOST=0.0.0.0:8000
ARG INSTALL_DEBUG_DEPS=false
ARG MINIO_HOST
ARG MINIO_ACCESS_KEY
ARG MINIO_SECRET_KEY
ARG MINIO_SECURE=false

# Set the working directory inside the container
WORKDIR /app

# Avoid writing .pyc files, enable unbuffered logs, and configure the app main.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GENEO_HOST=${GENEO_HOST} \
    DEBUG_MODE=false \
    CONTINUAL_LEARNING_MODE=true \
    SELECT_ALL_FEATURES=false \
    MINIO_HOST=${MINIO_HOST} \
    MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY} \
    MINIO_SECRET_KEY=${MINIO_SECRET_KEY} \
    MINIO_SECURE=${MINIO_SECURE}

# Copy the requirements files into the container
COPY requirements.txt requirements-debug.txt /app/

# Install product dependencies by default; add debug-only plotting dependencies when requested at build time.
RUN : "${MINIO_HOST:?MINIO_HOST build arg is required}" && \
    : "${MINIO_ACCESS_KEY:?MINIO_ACCESS_KEY build arg is required}" && \
    : "${MINIO_SECRET_KEY:?MINIO_SECRET_KEY build arg is required}" && \
    pip install --no-cache-dir --upgrade pip && \
    if [ "$INSTALL_DEBUG_DEPS" = "true" ]; then \
        pip install --no-cache-dir -r requirements-debug.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy the FastAPI application files into the container
COPY geneos_siemens.py geneos_siemens_core.py /app/

# Run the FastAPI app through the script main, which imports and starts Uvicorn.
CMD ["python", "geneos_siemens.py"]
