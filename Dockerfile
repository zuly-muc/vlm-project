# Self-contained CPU image: app + CPU-only torch + baked BLIP weights.
# The dataset and outputs are NEVER baked in — they are mounted at runtime.
FROM python:3.11-slim

# HF cache location baked into the image; weights land here at build time.
ENV HF_HOME=/opt/hf_cache \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) CPU-only torch first (its own index) so no multi-GB CUDA wheels are pulled.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

# 2) Remaining pinned runtime deps.
COPY requirements.lock ./
RUN grep -v '^torch==' requirements.lock | pip install -r /dev/stdin

# 3) Install the package itself.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# 4) Bake BLIP weights into the image (needs network only at build time).
COPY scripts ./scripts
RUN python scripts/bake_weights.py

# Default dataset/output contract (override with -e / -v at run time).
ENV LOADER=nuscenes \
    DATAROOT=/data \
    DB=/out/scenes.db

ENTRYPOINT ["vlm-project"]
CMD ["--help"]
