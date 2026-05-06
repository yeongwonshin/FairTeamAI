from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


DEFAULT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "development": {
        "code": 0.38,
        "document": 0.22,
        "slide": 0.10,
        "meeting": 0.18,
        "role": 0.12,
    },
    "report": {
        "code": 0.08,
        "document": 0.45,
        "slide": 0.15,
        "meeting": 0.20,
        "role": 0.12,
    },
    "presentation": {
        "code": 0.08,
        "document": 0.22,
        "slide": 0.35,
        "meeting": 0.20,
        "role": 0.15,
    },
    "balanced": {
        "code": 0.24,
        "document": 0.26,
        "slide": 0.16,
        "meeting": 0.20,
        "role": 0.14,
    },
}


@dataclass(frozen=True)
class RiskThresholds:
    low_contribution_share: float = 0.12
    very_low_contribution_share: float = 0.07
    high_overclaim_gap: float = 0.20
    overload_share: float = 0.45
    conflict_keyword_weight: float = 0.10
