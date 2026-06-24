"""Data store — manages persistence for a single Xanadue instance.

Each Xanadue gets a directory under <ha_config>/xanadue_data/<slug>/ containing:
  - priors.json:         fitted time-of-day priors
  - ground_truth.jsonl:  every correction (append-only)
  - posterior_log.jsonl: every posterior update (append-only)
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional


def get_data_dir(hass_config_dir: str, slug: str) -> str:
    """Get the data directory for a Xanadue instance."""
    path = os.path.join(hass_config_dir, "xanadue_data", slug)
    os.makedirs(path, exist_ok=True)
    return path


class DataStore:
    """Manages on-disk persistence for a single Xanadue."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.priors_path = os.path.join(data_dir, "priors.json")
        self.ground_truth_path = os.path.join(data_dir, "ground_truth.jsonl")
        self.posterior_log_path = os.path.join(data_dir, "posterior_log.jsonl")

    def append_ground_truth(
        self,
        area: str,
        source: str = "manual",
        weight: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """Append a ground-truth correction to the JSONL log."""
        if timestamp is None:
            timestamp = time.time()

        entry = {
            "ts": timestamp,
            "area": area,
            "source": source,
            "weight": weight,
        }
        with open(self.ground_truth_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def append_posterior_log(self, estimate_dict: dict) -> None:
        """Append a posterior update to the JSONL log."""
        entry = dict(estimate_dict)
        entry["ts"] = entry.get("ts", time.time())
        with open(self.posterior_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_priors_path(self) -> str:
        """Path to priors.json for this instance."""
        return self.priors_path

    def get_correction_stats(self) -> dict:
        """Count corrections by source."""
        stats = {"manual": 0, "auto": 0, "total": 0}
        if not os.path.exists(self.ground_truth_path):
            return stats
        with open(self.ground_truth_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    source = entry.get("source", "manual")
                    stats[source] = stats.get(source, 0) + 1
                    stats["total"] += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        return stats
