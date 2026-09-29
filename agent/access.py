"""Shared-passcode gate for the public voice demo.

The Render deploy is a public URL, and every turn spends Groq free-tier
quota and touches the Shopify dev store's carts. A single shared passcode
(DEMO_PASSCODE, handed to whoever should try the demo) keeps strangers
from using it without adding accounts or a login system. When
DEMO_PASSCODE is unset, e.g. local development, there's no gate at all.
"""

from __future__ import annotations

import os
import secrets


def passcode_required() -> bool:
    return bool(os.environ.get("DEMO_PASSCODE"))


def passcode_ok(provided: str | None) -> bool:
    expected = os.environ.get("DEMO_PASSCODE")
    if not expected:
        return True
    if provided is None:
        return False
    # Constant-time compare, so response timing doesn't leak how many
    # leading characters of a guess were right.
    return secrets.compare_digest(provided.encode(), expected.encode())
