"""Single LLM entry point. Swap providers here and nowhere else.

Every other module calls `generate(prompt, system)` and never talks to a
provider SDK directly, so switching from Groq to something else is a
one-file change.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2

# Repo root .env, e.g. GROQ_API_KEY=... -- loaded once at import time so
# GROQ_API_KEY is visible via os.environ without needing to export it in
# the shell first. A shell-exported value always wins if both are set.
_ENV_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".env"
try:
    from dotenv import load_dotenv

    load_dotenv(_ENV_PATH)
except ImportError:
    pass


def generate(prompt: str, system: str) -> str:
    """Call the default LLM provider (Groq) and return its text response.

    Raises on any failure (missing key, network error, provider error) --
    callers are expected to catch this and fall back to a deterministic
    templated answer rather than show the user a broken response.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    from groq import Groq

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Anthropic alternative (commented out). To switch providers, replace the
# body of generate() above with this block and set ANTHROPIC_API_KEY.
# ---------------------------------------------------------------------------
#
# def generate(prompt: str, system: str) -> str:
#     import os
#     import anthropic
#
#     api_key = os.environ.get("ANTHROPIC_API_KEY")
#     if not api_key:
#         raise RuntimeError("ANTHROPIC_API_KEY is not set")
#
#     client = anthropic.Anthropic(api_key=api_key)
#     response = client.messages.create(
#         model="claude-sonnet-4-5",
#         max_tokens=1024,
#         temperature=DEFAULT_TEMPERATURE,
#         system=system,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return response.content[0].text.strip()
