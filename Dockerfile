FROM python:3.11-slim

# Set environment variables for non-interactive and unbuffered python execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/house_price_pipeline.joblib

WORKDIR /app

# Install dependencies first for efficient layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and trained model artifact
COPY src/ ./src/
COPY models/ ./models/

# Expose internal FastAPI port
EXPOSE 8000

# Run uvicorn server binding to 0.0.0.0
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
