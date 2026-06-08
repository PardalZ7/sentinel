FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -e "sentinel/.[agent]"
RUN chmod +x sentinel/entrypoint.sh

WORKDIR /app/sentinel

CMD ["./entrypoint.sh"]
