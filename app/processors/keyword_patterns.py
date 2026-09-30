"""
SafeSight AI — Keyword Hard-Block Patterns
============================================
Zero-dependency module containing pre-compiled regex patterns for
Tier-1 instant keyword blocking. Separated from tier1_gate.py so it
can be imported in lightweight test environments without torch/structlog.
"""

from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compiled hard-block patterns
# ─────────────────────────────────────────────────────────────────────────────

HARD_BLOCK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(kill\s+yourself|kys)\b", re.IGNORECASE),
    re.compile(r"\b(n[i1!]gg[ae3]r|ch[i1]nk|sp[i1]c|f[a4]gg[o0]t)\b", re.IGNORECASE),
    re.compile(r"\b(rape|raped|raping)\b", re.IGNORECASE),
    re.compile(r"\bcp\b|\bchild.?porn\b|\bpedo(phile)?\b", re.IGNORECASE),
    re.compile(r"\b(bomb|explosive).{0,15}(build|make|detonate)\b", re.IGNORECASE),
    re.compile(r"\b(suicide|self.?harm).{0,20}method\b", re.IGNORECASE),
]


def keyword_hard_block(text: str) -> tuple[bool, str]:
    """
    Check text against the hard-block pattern list.

    Returns:
        (is_blocked: bool, matched_trigger: str)
    """
    for pattern in HARD_BLOCK_PATTERNS:
        match = pattern.search(text)
        if match:
            return True, match.group(0)
    return False, ""
