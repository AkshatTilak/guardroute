import os
import pytest

# Inject mock environment variables required for initialization before importing app code
os.environ["DATABASE_URL"] = "postgresql+asyncpg://postgres:postgres@localhost:5432/guardroute_test"
os.environ["OPENROUTER_API_KEY"] = "mock_openrouter_api_key"
os.environ["APP_ENV"] = "testing"
