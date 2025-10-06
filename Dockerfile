# Dockerfile for EV Charge Forecasting

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/

# Create necessary directories
RUN mkdir -p data logs outputs/models outputs/results outputs/plots outputs/processed

# Install dependencies
RUN uv sync

# Set entrypoint
ENTRYPOINT ["uv", "run", "python", "-m", "ev_charge_forecasting.cli"]

# Default command: show help
CMD ["--help"]
