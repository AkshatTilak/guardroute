import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # App Settings
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    port: int = Field(default=8000, validation_alias="PORT")
    host: str = Field(default="0.0.0.0", validation_alias="HOST")

    # Database Settings
    database_url: str = Field(..., validation_alias="DATABASE_URL")
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")

    # API Keys & External Providers
    openrouter_api_key: str = Field(..., validation_alias="OPENROUTER_API_KEY")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    # Local Router Configuration
    router_model_path: str = Field(
        default="models/Arch-Router-1.5B-Q8_0.gguf",
        validation_alias="ROUTER_MODEL_PATH",
    )

    # Telemetry Configuration
    otel_service_name: str = Field(default="guardroute-gateway", validation_alias="OTEL_SERVICE_NAME")
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Instantiate settings instance immediately to validate configurations on import.
# Note: Ensure required environment variables are set in the environment or .env file before importing.
settings = Settings()
