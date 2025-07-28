# Base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y curl && apt-get clean

# Install uv (faster dependency resolver)
RUN curl -Ls https://astral.sh/uv/install.sh | sh

# Set work directory
WORKDIR /app

# Copy pyproject + lock file and install dependencies
COPY pyproject.toml uv.lock ./
RUN uv pip install -r <(uv pip compile pyproject.toml)

# Copy source code
COPY src/ ./src/

# Run the pipeline
ENTRYPOINT ["python", "src/run_pipeline.py"]
