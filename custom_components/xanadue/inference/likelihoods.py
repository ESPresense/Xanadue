"""Likelihood models for each sensor type.

P(observation | area) for BLE, motion, and GPS sensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class Observation:
    """A single sensor observation at a point in time."""

    entity_id: str
    kind: str          # "ble", "motion", "gps"
    state: str         # raw HA state
    area: Optional[str] = None  # observed area (BLE) or area (motion)
    confidence: float = 1.0     # BLE confidence (0-1)
    age_seconds: float = 0.0    # how old this observation is


def ble_likelihood(obs: Observation, candidate_area: str) -> float:
    """P(BLE observation | area = candidate_area).

    If the BLE tracker reports area X with confidence c:
      P(o | X) = c
      P(o | Y) = (1 - c) / (N - 1)   for Y != X (leakage)
    """
    if obs.area is None:
        return 1.0  # uninformative

    if candidate_area == obs.area:
        return obs.confidence

    # Leak remaining probability mass equally to other areas
    # N is unknown here, so we use a fixed small leak rate
    # The caller normalizes, so the relative weighting is what matters
    return (1.0 - obs.confidence) * 0.15


def motion_likelihood(obs: Observation, candidate_area: str) -> float:
    """P(motion observation | area = candidate_area).

    If motion is ON in area X:
      P(on | X) = recency-weighted high
      P(on | Y) = small leak (someone walked past)
    If motion is OFF in area X:
      P(off | X) = high (but decays — person may be stationary)
      P(off | Y) = uninformative
    """
    is_on = obs.state == "on"

    if is_on:
        if candidate_area == obs.area:
            # Recency decay: fresh motion = strong, stale = weaker
            decay = math.exp(-obs.age_seconds / 120.0)  # 2-min half-life
            return 0.8 * decay + 0.2  # floor of 0.2 so stale motion isn't zero
        else:
            return 0.1  # leak — someone walked past. Must stay below match floor (0.2)
                       # so a stale ON sensor still weakly favors its own area.
    else:
        # Motion OFF
        if candidate_area == obs.area:
            # Motion off in candidate area = person might be sitting still
            return 0.5  # uninformative-but-slightly-negative
        else:
            return 1.0  # no information from motion being off elsewhere


def gps_likelihood(obs: Observation, candidate_area: str) -> float:
    """P(GPS observation | area = candidate_area).

    GPS zone.home is uninformative at area level — the person is
    *somewhere* in the house. zone.not_home means they're away.
    """
    if obs.state in ("not_home", "away"):
        return 0.01  # very unlikely to be in any area if GPS says not_home

    # state is "home" or a zone name → uninformative for which area
    return 1.0


def compute_likelihood(obs: Observation, candidate_area: str) -> float:
    """Dispatch to the right likelihood model by sensor kind."""
    if obs.kind == "ble":
        return ble_likelihood(obs, candidate_area)
    elif obs.kind == "motion":
        return motion_likelihood(obs, candidate_area)
    elif obs.kind == "gps":
        return gps_likelihood(obs, candidate_area)
    return 1.0  # unknown → uninformative
