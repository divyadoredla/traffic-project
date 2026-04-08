FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY traffic_env/ ./traffic_env/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Environment variable defaults
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4.1-mini"


CMD ["python", "-u", "inference.py"]
