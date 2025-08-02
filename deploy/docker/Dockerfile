# --- STAGE 1: Builder (using Airflow base image) ---
FROM apache/airflow:3.0.3 AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install curl and pipx for uv
USER root
RUN apt-get update && apt-get install -y curl pipx && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pipx install uv && pipx ensurepath

# Set workdir and copy dependency files
WORKDIR /app
COPY pyproject.toml uv.lock ./

# Compile and install dependencies using uv
RUN sh -c "/root/.local/bin/uv pip compile pyproject.toml -o /tmp/requirements.txt && /root/.local/bin/uv pip install --system --requirement /tmp/requirements.txt"

# --- STAGE 2: Final Runtime (also Airflow base image) ---
FROM apache/airflow:3.0.3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY src/ ./src/

# Optional: copy any config/scripts if needed
# COPY dags/ ./dags/
# COPY airflow.cfg /opt/airflow/

# Default entrypoint (used for standalone execution – overridden in Airflow containers)
ENTRYPOINT ["python", "src/run_pipeline.py"]
