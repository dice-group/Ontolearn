# Docker Guide for Ontolearn

This guide explains how to build, run, and use Ontolearn with Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2.0+)
- Knowledge graphs downloaded (see [README.md](../../README.md) for instructions):
  ```bash
  wget https://files.dice-research.org/projects/Ontolearn/KGs.zip -O ./KGs.zip && unzip KGs.zip
  wget https://files.dice-research.org/projects/Ontolearn/LPs.zip -O ./LPs.zip && unzip LPs.zip
  ```

---

## Quick Start

### 1. Build and Run the Webservice

```bash
# Build the production image
docker compose build ontolearn

# Start the webservice (uses local KGs/Family ontology by default)
docker compose up ontolearn
```

The API will be available at **http://localhost:8000**.

Verify it's running:
```bash
curl http://localhost:8000/
# Expected: {"response":"Ontolearn Service is Running"}
```

### 2. Query the Concept Learning Endpoint

```bash
curl -X GET http://localhost:8000/cel \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TDL",
    "pos": ["http://www.benchmark.org/family#F2F28"],
    "neg": ["http://www.benchmark.org/family#F2F14", "http://www.benchmark.org/family#F2M18"]
  }'
```

---

## Usage Scenarios

### A. Webservice with Local Knowledge Base (Default)

```bash
# Start with the default Family ontology
docker compose up ontolearn

# Or specify a different knowledge base
docker compose run --rm -p 8000:8000 ontolearn \
  ontolearn-webservice \
  --path_knowledge_base KGs/Mutagenesis/mutagenesis.owl \
  --host 0.0.0.0 --port 8000
```

### B. Webservice with an External Triplestore

When using the `triplestore` profile, Ontolearn connects to a **pre-existing** triplestore.
You must have a triplestore server already running with the target dataset already loaded and
accessible at a SPARQL endpoint. No ontology upload is required.

Pass the endpoint via the `TRIPLE_STORE_ENDPOINT` environment variable:

**Option 1 — inline on the command line:**
```bash
TRIPLE_STORE_ENDPOINT=http://my-fuseki-host:3030/family/sparql \
  docker compose --profile triplestore up
```

**Option 2 — using a `.env` file:**
```bash
# Copy the example and edit it
cp .env.example .env
# Edit TRIPLE_STORE_ENDPOINT in .env, then:
docker compose --profile triplestore up
```

The Ontolearn webservice will be available at **http://localhost:8001** and will query the
endpoint you provided directly. No Fuseki (or other triplestore) container is started or
managed by Docker Compose.

> **Note:** If the triplestore runs on the same host as Docker, use
> `host.docker.internal` instead of `localhost` in the endpoint URL:
> ```bash
> TRIPLE_STORE_ENDPOINT=http://host.docker.internal:3030/family/sparql \
>   docker compose --profile triplestore up
> ```

### C. Run a Python Script

```bash
# Run main.py with default arguments
docker compose run --rm ontolearn python main.py

# Run with custom arguments
docker compose run --rm ontolearn python main.py \
  --model celoe \
  --knowledge_base_path KGs/Family/family-benchmark_rich_background.owl \
  --path_learning_problem examples/uncle_lp2.json

# Run any example script
docker compose run --rm ontolearn python examples/concept_learning_with_tdl_local_kb.py
```

### D. Development Environment

Get an interactive shell with the full development dependencies:

```bash
docker compose --profile dev run --rm dev bash
```

Inside the container you have access to the full source code (mounted from your host) and all dependencies including `pytest`, `ruff`, `dicee`, etc.

### E. Run Tests

```bash
docker compose --profile test run --rm test
```

Or run specific tests:
```bash
docker compose --profile dev run --rm dev pytest tests/test_tdl.py -v
```

---

## Docker Build Targets

The `Dockerfile` uses multi-stage builds. You can target a specific stage:

| Target        | Description                                   | Use Case                  |
|---------------|-----------------------------------------------|---------------------------|
| `production`  | Minimal install with all source code           | Webservice & scripts      |
| `development` | Full install with test/doc/lint dependencies   | Development & debugging   |
| `test`        | Same as development, runs `pytest` by default  | CI/CD testing             |

```bash
# Build only the production image
docker build --target production -t ontolearn:prod .

# Build the development image
docker build --target development -t ontolearn:dev .
```

---

## GPU Support (NVIDIA CUDA)

To use GPU acceleration for neural models (DRILL, NCES, etc.):

1. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Modify the `Dockerfile` — replace the PyTorch install line:
   ```dockerfile
   # Replace this:
   RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
   # With this:
   RUN pip install torch
   ```

3. Add GPU configuration to `docker-compose.yml` under the desired service:
   ```yaml
   ontolearn:
     # ...existing config...
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
   ```

4. Rebuild:
   ```bash
   docker compose build ontolearn
   ```

---

## Volume Mounts

| Host Path       | Container Path      | Purpose                         |
|-----------------|---------------------|---------------------------------|
| `./KGs`         | `/app/KGs`          | Knowledge graphs (ontologies)   |
| `./LPs`         | `/app/LPs`          | Learning problems               |
| `./checkpoints` | `/app/checkpoints`  | Model checkpoints               |

To add custom data, place files in the respective host directories — they will be available in the container.

---

## Environment Variables

| Variable               | Default | Description                                    |
|------------------------|---------|------------------------------------------------|
| `KNOWLEDGE_BASE_PATH`  | —       | Path to the OWL ontology file inside container |

---

## Common Commands Cheat Sheet

```bash
# Build everything
docker compose build

# Start webservice in background
docker compose up -d ontolearn

# View logs
docker compose logs -f ontolearn

# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v

# Rebuild without cache
docker compose build --no-cache ontolearn

# Shell into running container
docker exec -it ontolearn-webservice bash
```

---

## Troubleshooting

**Port already in use:**
```bash
# Change the host port mapping
docker compose run --rm -p 9000:8000 ontolearn ontolearn-webservice --host 0.0.0.0 --port 8000
```

**Out of memory during build:**
```bash
# Increase Docker memory limit in Docker Desktop settings, or use:
DOCKER_BUILDKIT=1 docker build --target production -t ontolearn .
```

**Knowledge graphs not found:**
Make sure you have downloaded and extracted KGs into the project root:
```bash
wget https://files.dice-research.org/projects/Ontolearn/KGs.zip -O ./KGs.zip && unzip KGs.zip
```

