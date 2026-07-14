"""GuardRoute runtime safety guardrails.

Implements pre-flight prompt injection checks and post-flight PII scrubbing
and toxicity filtering.
"""

import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger("guardroute.agents.guardrails")

# Pre-flight injection patterns
INJECTION_REGEXES = [
    re.compile(r"ignore\s+(?:the\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"bypass\s+(?:the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a\s+[^.\n]+", re.IGNORECASE),
    re.compile(r"system\s+override", re.IGNORECASE),
    re.compile(r"new\s+instructions\s*:", re.IGNORECASE),
    re.compile(r"override\s+(?:the\s+)?prompt", re.IGNORECASE),
    re.compile(r"dan\s+mode", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
]

# Post-flight PII patterns
PII_PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "API_KEY": re.compile(r"\b(?:api[-_]?key|sk_live_[a-zA-Z0-9]+|sk_test_[a-zA-Z0-9]+|sk-[a-zA-Z0-9]{20,})\b", re.IGNORECASE),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

# Simple list of toxicity trigger words for simulation
TOXICITY_KEYWORDS = ["bastard", "idiot", "asshole", "bitch", "crap", "fuck", "shit"]


def clean_html_tags(text: str) -> str:
    """Strips HTML/script tags from input text."""
    # Simple regex to strip tags
    clean_text = re.sub(r"<[^>]*>", "", text)
    return clean_text


def check_prompt_injection(prompt: str) -> Tuple[bool, Optional[str]]:
    """Checks if a user prompt contains typical prompt injection/jailbreak patterns.
    
    Returns:
        (is_safe, error_message)
    """
    for regex in INJECTION_REGEXES:
        if regex.search(prompt):
            logger.warning("Prompt injection attempt detected: matched pattern '%s'", regex.pattern)
            try:
                from common.observability.logger import log_security_event
                log_security_event("PROMPT_INJECTION_DETECTED", {"matched_pattern": regex.pattern, "prompt_snippet": prompt[:100]})
            except Exception as e:
                logger.error("Failed to log prompt injection security event: %s", e)
            return False, "Prompt rejected: security policy violation (suspicious injection pattern detected)."
            
    return True, None


def scrub_pii(text: str) -> str:
    """Redacts PII like emails, phone numbers, API keys, and SSNs from text."""
    scrubbed = text
    for pii_type, regex in PII_PATTERNS.items():
        scrubbed = regex.sub(f"[REDACTED_{pii_type}]", scrubbed)
    return scrubbed


def check_toxicity(text: str, threshold: float = 0.1) -> float:
    """Computes a simulated toxicity score based on trigger words.
    
    Returns:
        Toxicity score (0.0 to 1.0)
    """
    text_lower = text.lower()
    matches = 0
    words = text_lower.split()
    if not words:
        return 0.0

    for word in words:
        # Strip punctuation
        clean_word = re.sub(r"[^\w]", "", word)
        if clean_word in TOXICITY_KEYWORDS:
            matches += 1
            
    score = matches / len(words)
    # Cap score at 1.0
    return min(score * 10.0, 1.0)


async def check_hallucination_grounding(response: str, context: str) -> Tuple[bool, Optional[str]]:
    """Verify that the response is grounded in the retrieved context using the active completion model.
    
    Returns:
        (is_grounded, error_message)
    """
    if not context or not context.strip():
        return True, None

    prompt = (
        "You are a Hallucination Grounding Validator.\n"
        "Analyze the following Response and Context. Determine if the Response is fully grounded in and supported by the Context.\n"
        "If there are statements in the Response that contradict or cannot be inferred from the Context, mark it as ungrounded.\n"
        "Respond with EXACTLY 'GROUNDED' or 'UNGROUNDED' on the first line, followed by a brief reason.\n\n"
        f"Context:\n{context}\n\n"
        f"Response:\n{response}"
    )

    try:
        from common.clients.litellm import completion_with_fallback
        from common.models.registry import get_active_model

        model_spec = await get_active_model("completion")
        model_name = model_spec.model_id

        res = await completion_with_fallback(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        content = res.choices[0].message.content.strip()
        if content.upper().startswith("UNGROUNDED"):
            # Extract reason if present after newline/spaces
            reason = content[10:].strip()
            if reason.startswith(":") or reason.startswith("-"):
                reason = reason[1:].strip()
            return False, f"Response rejected: failed hallucination grounding check. Reason: {reason or 'Response not supported by retrieved context.'}"
        return True, None
    except Exception as e:
        logger.warning("Failed to run hallucination grounding check: %s. Assuming grounded.", e)
        return True, None

