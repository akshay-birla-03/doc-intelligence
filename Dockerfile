FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8000

# Serve the FastAPI app with uvicorn.
CMD ["uvicorn", "docintel.api:app", "--host", "0.0.0.0", "--port", "8000"]
