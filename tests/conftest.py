import os
import pytest
from common.config import settings

# Inject mock environment variables required for initialization before importing app code
os.environ["DATABASE_URL"] = "postgresql+asyncpg://contained:changeme@localhost:5432/contained_platform"
os.environ["OPENROUTER_API_KEY"] = "mock_openrouter_api_key"
os.environ["APP_ENV"] = "testing"

# Force settings singleton to reflect the test environment (preventing cache issues)
settings.APP_ENV = "testing"
settings.DATABASE_URL = "postgresql+asyncpg://contained:changeme@localhost:5432/contained_platform"
settings.OPENROUTER_API_KEY = "mock_openrouter_api_key"
