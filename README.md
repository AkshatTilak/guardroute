# GuardRoute: Contract-Aware AI Gateway & MLOps Control Plane

GuardRoute is a production-grade AI control plane (inspired by the ModelGate architecture) designed to ingest customer contracts, extract SLAs and privacy constraints, route user prompts to the optimal model, record standardized LLM metrics using OpenTelemetry, and monitor semantic drift.

## System Layout

Every workspace directory conforms to the standardized layout:

- `config/`: Configuration files (YAML, system settings)
- `deploy/`: Deployment configurations (K8s, Docker, Spark)
- `docs/`: Architectural documents & diagrams
- `scripts/`: Utility and automation scripts
- `src/`: Core source code
  - `api/`: API controllers and gateways (FastAPI)
  - `core/`: Core configurations, environments, constants
  - `database/`: Relational database schemas & client wrappers
  - `vectors/`: Vector database setups, collections, index configs
  - `agents/`: LangGraph, LangChain, or smolagents definition
  - `utils/`: Helper utilities (logging, formatting)
- `tests/`: Complete test suite (unit, integration, and evaluation)

## Getting Started

### Prerequisites

- Python >= 3.11, < 3.13
- [Poetry](https://python-poetry.org/) (for dependency management)

### Installation

1. Clone or navigate to the workspace:
   ```bash
   cd c:\Akshat\proj\guardroute
   ```

2. Copy `.env.example` to `.env` and fill in required fields:
   ```bash
   cp .env.example .env
   ```

3. Install project dependencies:
   ```bash
   poetry install
   ```

### Running the Gateway

Start the FastAPI application proxy gateway locally:
```bash
poetry run uvicorn src.main:app --reload
```

### Running Tests

Execute the unit and integration test suite:
```bash
poetry run pytest
```
