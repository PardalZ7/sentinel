FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install -e ".[agent]"
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
