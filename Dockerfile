# Use Alpine Linux with Python - much smaller (~50MB vs ~150MB)
FROM python:3.11-alpine

# Set working directory inside container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies needed by Alpine
# Alpine uses apk instead of apt-get
RUN apk add --no-cache gcc musl-dev libffi-dev g++ cmake

# Copy requirements first (Docker caches this layer)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app/ ./app/
COPY data/ ./data/
COPY frontend/ ./frontend/

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]