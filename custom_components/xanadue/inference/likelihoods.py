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


# Home Assistant device_tracker states that mean "in the house but no room
# information" rather than a named zone. `home` is the state inside `zone.home`
# (HA uses the zone's name, minus the `zone.` prefix), and the empty/unknown set
# covers a tracker that has not reported yet.
_GPS_NON_ZONE_STATES = ("", "unknown", "unavailable", "none")


def gps_indicates_home(state: Optional[str]) -> bool:
    """True when a GPS state means "in the house", room unspecified."""
    return (state or "").strip().lower() in _GPS_NON_ZONE_STATES + ("home",)


def gps_indicates_away(state: Optional[str]) -> bool:
    """True when a GPS state puts the person outside the house.

    Anything that is not the home zone and not a missing value counts as away,
    which covers `not_home`, `away`, and every named away zone (`work`,
    `school`, `shops`, iCloud3 activity locations).
    """
    return not gps_indicates_home(state)


def gps_likelihood(obs: Observation, candidate_area: str) -> float:
    """P(GPS observation | area = candidate_area).

    GPS is a coarse signal: `home` means the person is *somewhere* in the house,
    which says nothing about which room, so it is uninformative at area level.
    Any other zone state means they are *not* in the house.

    The previous version only recognised the literal strings `not_home`/`away`,
    so every named away zone — `work`, `school`, `shops`, iCloud3 activity
    locations — fell through to the `1.0` branch and was silently treated as
    "home, area unknown". That collapsed the posterior to the prior and let a
    flat tie decide the reported room (issue #2). In Home Assistant a
    device_tracker's state inside a zone is the zone's name (`zone.home` →
    `"home"`), so matching that one string is enough to split home from
    everywhere else without reaching into `hass`.
    """
    state = (obs.state or "").strip().lower()

    # No usable signal at all — treat as uninformative rather than "away", so a
    # temporarily unknown tracker does not assert the person has left the house.
    if state in ("", "unknown", "unavailable", "none"):
        return 1.0

    # At home (or in the home zone): somewhere in the house, room unknown.
    if state == "home":
        return 1.0

    # Any other zone, including `not_home`/`away`, means not in the house.
    return 0.01


def compute_likelihood(obs: Observation, candidate_area: str) -> float:
    """Dispatch to the right likelihood model by sensor kind."""
    if obs.kind == "ble":
        return ble_likelihood(obs, candidate_area)
    elif obs.kind == "motion":
        return motion_likelihood(obs, candidate_area)
    elif obs.kind == "gps":
        return gps_likelihood(obs, candidate_area)
    return 1.0  # unknown → uninformative
