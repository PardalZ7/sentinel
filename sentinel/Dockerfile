FROM python:3.11-slim

WORKDIR /app

COPY sentinel/ .

RUN pip install -e .

CMD ["sentinel", "start", "--config", "sentinel-cloud.json", "--mode", "agent"]
