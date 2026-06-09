FROM python:3.11-slim AS builder

WORKDIR /app

COPY sentinel/pyproject.toml sentinel/pyproject.toml
COPY sentinel/sentinel/ sentinel/sentinel/

# Install CPU-only torch first to avoid pulling CUDA (~2 GB → ~200 MB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir "sentinel/.[agent]"

# ---- runtime image ----
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only what runtime needs
COPY sentinel/sentinel/ sentinel/sentinel/
COPY sentinel/entrypoint.sh sentinel/entrypoint.sh
RUN chmod +x sentinel/entrypoint.sh

WORKDIR /app/sentinel

CMD ["./entrypoint.sh"]
