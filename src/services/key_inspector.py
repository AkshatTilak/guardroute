"""GuardRoute API Key Inspector & Model Route Availability Service (sub_08_02)."""

import logging
from typing import Dict, Any
from common.config.settings import get_settings

logger = logging.getLogger("guardroute.services.key_inspector")


class APIKeyInspector:
    """Inspects environment and provider keys to compute availability status flags for routing."""

    @staticmethod
    def inspect_keys() -> Dict[str, bool]:
        settings = get_settings()
        return {
            "openai": bool(getattr(settings, "OPENAI_API_KEY", None)),
            "gemini": bool(getattr(settings, "GOOGLE_API_KEY", None)),
            "google": bool(getattr(settings, "GOOGLE_API_KEY", None)),
            "anthropic": bool(getattr(settings, "ANTHROPIC_API_KEY", None)),
            "openrouter": bool(getattr(settings, "OPENROUTER_API_KEY", None)),
            "groq": bool(getattr(settings, "GROQ_API_KEY", None)),
            "cerebras": bool(getattr(settings, "CEREBRAS_API_KEY", None)),
        }

    @classmethod
    def get_model_status_flag(cls, provider: str, mode: str) -> str:
        """Return status flag: 'ready', 'missing_key', or 'local_only'."""
        if mode == "local":
            return "local_only"
        keys = cls.inspect_keys()
        is_available = keys.get(provider.lower(), False)
        return "ready" if is_available else "missing_key"
