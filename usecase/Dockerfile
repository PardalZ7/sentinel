FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python sentinel package — pinned to commit so cache is never stale
RUN pip install --no-cache-dir \
    "sentinel[engine] @ git+https://github.com/PardalZ7/sentinel.git@6686e62eeb297f0b39d2918fec305f68a44ffcbb#subdirectory=sentinel"

# Install Node dependencies (workspace)
COPY package.json ./
COPY shared/ ./shared/
COPY apps/ ./apps/
COPY dashboard/ ./dashboard/
RUN npm install --workspaces --include-workspace-root

# Copy sentinel config and entrypoint
COPY sentinel.json .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 3000

CMD ["./entrypoint.sh"]
