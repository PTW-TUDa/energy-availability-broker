# syntax=docker/dockerfile:1

# Define build arguments for Python and Poetry
ARG PYTHON_VERSION=3.10
ARG POETRY_VERSION=2.1.4

# Use the official Python slim image as our base
FROM python:${PYTHON_VERSION}-slim AS final

# Set working directory
WORKDIR /app

# Environment variables to configure Poetry
ENV POETRY_HOME="/usr/local/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_CACHE_DIR="/root/.cache/pip" \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Make sure Poetry is on our PATH
ENV PATH="${POETRY_HOME}/bin:$PATH"

# Install system dependencies (curl, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential gcc g++ libgomp1 git tzdata \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python3 -

# 3) Copy the rest of your code
COPY . .

# Install Python dependencies
RUN poetry install

# Expose port 8000 for FastAPI
EXPOSE 8000

# 6) Set the command to run FastAPI via Poetry
CMD ["poetry", "run", "fastapi", "run", "./energy_information_service/main.py", "--host", "0.0.0.0", "--port", "8000"]
