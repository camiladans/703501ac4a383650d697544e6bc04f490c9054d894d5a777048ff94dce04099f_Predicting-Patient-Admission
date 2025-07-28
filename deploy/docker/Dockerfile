FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install curl and pipx (to install uv securely)
RUN apt-get update && apt-get install -y curl pipx && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pipx install uv && pipx ensurepath

# Set workdir
WORKDIR /app

# Copy pyproject files
COPY pyproject.toml uv.lock ./

# # Install dependencies using uv via pipx
# RUN /root/.local/bin/uv pip install --system --no-cache --requirement <(/root/.local/bin/uv pip compile pyproject.toml)

# Install dependencies using uv via pipx
RUN sh -c "/root/.local/bin/uv pip compile pyproject.toml -o /tmp/req.txt && /root/.local/bin/uv pip install --system --requirement /tmp/req.txt"

# Copy your source code
COPY src/ ./src/

ENTRYPOINT ["python", "src/run_pipeline.py"]
