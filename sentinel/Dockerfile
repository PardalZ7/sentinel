FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -e ".[agent]"

CMD ["sentinel", "start", "--config", "sentinel-cloud.json", "--mode", "agent"]
