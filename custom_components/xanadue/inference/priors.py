"""Time-of-day priors for area inference.

Priors are keyed by (person_slug, hour_of_day, area) and stored as JSON.
They're updated incrementally by xanadue.correct service calls.

Default priors are uniform until corrections are received.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Optional

from ..const import DEFAULT_SMOOTHING_ALPHA


def hour_bucket(timestamp: Optional[float] = None) -> int:
    """Get the hour bucket (0-23) for a timestamp."""
    if timestamp is None:
        timestamp = time.time()
    return int(time.localtime(timestamp).tm_hour)


class PriorStore:
    """Manages time-of-day priors for a single Xanadue instance.

    Priors stored as:
        {
            "0": {"kitchen": 0.1, "family_room": 0.1, "master": 0.6, ...},
            "1": {...},
            ...
        }

    With Laplace smoothing applied to raw counts.
    """

    def __init__(self, priors_path: str, areas: list[str], alpha: float = DEFAULT_SMOOTHING_ALPHA):
        self.path = priors_path
        self.areas = areas
        self.alpha = alpha
        # Raw counts: {hour_str: {area: count}} — keys are str for JSON compat
        self._counts: dict[str, dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        """Load priors from disk, or initialize empty."""
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._counts = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._counts = {}
        # Ensure all 24 hours exist
        for h in range(24):
            if str(h) not in self._counts:
                self._counts[str(h)] = {}

    def _save(self) -> None:
        """Save priors to disk atomically."""
        tmp = self.path + ".tmp"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(self._counts, f, indent=2)
        os.replace(tmp, self.path)

    def add_correction(
        self,
        area: str,
        weight: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """Add a ground-truth correction, updating the prior for this hour."""
        h = hour_bucket(timestamp)
        hour_key = str(h)

        if area not in self._counts.get(hour_key, {}):
            self._counts.setdefault(hour_key, {})[area] = 0.0

        self._counts[hour_key][area] += weight
        self._save()

    def get_prior(self, hour: int) -> dict[str, float]:
        """Get the prior distribution over areas for a given hour.

        Returns a normalized distribution using Laplace smoothing.
        """
        hour_key = str(hour)
        raw = self._counts.get(hour_key, {})

        # Total weight for this hour
        total = sum(raw.values()) + self.alpha * len(self.areas)

        if total == 0:
            # No data at all → uniform
            n = len(self.areas) if self.areas else 1
            return {r: 1.0 / n for r in self.areas}

        # Laplace-smoothed distribution
        result = {}
        for area in self.areas:
            count = raw.get(area, 0.0)
            result[area] = (count + self.alpha) / total

        return result

    def get_prior_now(self) -> dict[str, float]:
        """Get the prior distribution for the current hour."""
        return self.get_prior(hour_bucket())

    def has_data(self) -> bool:
        """Whether any corrections have been received."""
        return any(
            any(v > 0 for v in areas.values())
            for areas in self._counts.values()
        )
