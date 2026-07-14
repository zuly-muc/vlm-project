# Self-contained CPU image: app + CPU-only torch + baked BLIP weights.
# The dataset and outputs are NEVER baked in, they are mounted at runtime.
#
# Python 3.12 is the interpreter the whole pipeline is tested on. Combined with
# the nuscenes-devkit 1.2.0 pin in requirements.lock, every dependency installs
# from a prebuilt wheel, the old 1.1.x line pulled Shapely<2.0 / matplotlib<3.6,
# which have no modern wheels and fail to compile in this toolchain-free image.
FROM python:3.12-slim

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

# Runtime is fully offline: the weights are baked, so forbid any hub network
# fetch at run time (this enforces the offline claim). Set AFTER baking, which
# is the one step that legitimately needs the network.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Default dataset/output contract (override with -e / -v at run time).
ENV LOADER=nuscenes \
    DATAROOT=/data \
    DB=/out/scenes.db

# Run as a non-root user (uid 10001). The baked HF cache is made world-readable
# so the container works whether it runs as this user or is overridden with
# `--user` to match a host. A host-mounted /out must be writable by the running
# uid; pass `--user "$(id -u):$(id -g)"` when writing to a host-owned directory.
RUN useradd --create-home --uid 10001 appuser \
    && chmod -R a+rwX /opt/hf_cache \
    && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["vlm-project"]
CMD ["--help"]
