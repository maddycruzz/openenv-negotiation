# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory — all files land here, all imports resolve correctly
WORKDIR /app

# Copy requirements first — Docker caches this layer
# If requirements don't change, pip install is skipped on rebuild
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Expose port 7860 — required by HuggingFace Spaces
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]