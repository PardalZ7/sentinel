FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install sentinel from local source (build context must be repo root)
COPY sentinel/ /sentinel/
RUN pip install --no-cache-dir "/sentinel[engine]"

# Install Node dependencies (workspace)
COPY usecase/package.json ./
COPY usecase/shared/ ./shared/
COPY usecase/apps/ ./apps/
COPY usecase/dashboard/ ./dashboard/
RUN npm install --workspaces --include-workspace-root

# Copy sentinel config and entrypoint
COPY usecase/sentinel.json .
COPY usecase/entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 3000

CMD ["./entrypoint.sh"]
