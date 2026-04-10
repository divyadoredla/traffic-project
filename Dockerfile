FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY traffic_env/ ./traffic_env/
COPY inference.py .
COPY server/ ./server/
COPY openenv.yaml .
COPY README.md .

ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4.1-mini"

CMD ["python", "-u", "inference.py"]
