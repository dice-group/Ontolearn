# ---- Base Stage ----
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Dependencies Stage ----
FROM base AS dependencies

# Copy only setup files first for better caching
COPY setup.py README.md ./

# Install CPU-only PyTorch first (smaller image), then the package
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ".[min]" --no-deps \
    && pip install -e ".[min]"

# ---- Production Stage ----
FROM dependencies AS production

# Copy the full project
COPY . .

# Re-install in editable mode with the full source
RUN pip install -e ".[min]"

# Expose the default webservice port
EXPOSE 8000

# Default command: run the webservice
# Override --path_knowledge_base or --endpoint_triple_store as needed
CMD ["ontolearn-webservice", "--path_knowledge_base", "KGs/Family/family-benchmark_rich_background.owl"]

# ---- Development Stage (includes test/doc dependencies) ----
FROM production AS development

RUN pip install -e ".[full]"

CMD ["bash"]

# ---- Test Stage ----
FROM development AS test

CMD ["pytest", "tests/", "-v", "--tb=short"]

