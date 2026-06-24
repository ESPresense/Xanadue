"""Classify sensors by entity domain and naming patterns.

Pure functions — no HA dependencies, fully testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SensorKind(str, Enum):
    """What kind of signal a sensor provides."""

    BLE = "ble"        # ESPresense / BLE room-level tracker
    MOTION = "motion"  # binary_sensor.*_occupancy
    GPS = "gps"        # device_tracker with zone states (home/not_home)
    UNKNOWN = "unknown"


@dataclass
class ClassifiedSensor:
    """A sensor classified into a kind with optional metadata."""

    entity_id: str
    kind: SensorKind
    area_hint: Optional[str] = None
    person_hint: Optional[str] = None


# Tokens that indicate BLE room-level trackers (vs GPS zone trackers).
# These need to be specific enough to NOT match plain iPhone GPS trackers
# like "device_tracker.darrells_iphone" — BLE trackers typically have a
# hardware/model identifier (e.g. "15_pro", "series_9") or explicit BLE
# tooling prefix (espresense, beacon, tag).
_BLE_TOKENS = frozenset({
    "espresense", "beacon", "tag",
})

# Patterns indicating a hardware/model suffix common in BLE room tracker names.
# Match things like "phone_darrell_15_pro" or "watch_darrell_series_9".
_BLE_MODEL_PATTERNS = (
    re.compile(r"\d+[_]?pro"),
    re.compile(r"series[_]?\d+"),
    re.compile(r"watch\d*"),
)

# Tokens that indicate occupancy/motion sensors
_MOTION_TOKENS = frozenset({
    "occupancy", "motion", "presence",
})


def classify(entity_id: str, person_name: str = "") -> ClassifiedSensor:
    """Classify a sensor by its entity_id.

    Args:
        entity_id: The HA entity_id (e.g. "device_tracker.phone_darrell_15_pro")
        person_name: The person's name (e.g. "Darrell") for person_hint matching

    Returns:
        ClassifiedSensor with kind and optional room/person hints.
    """
    domain, _, object_id = entity_id.partition(".")
    oid_lower = object_id.lower()

    # Binary sensors → motion
    if domain == "binary_sensor":
        for token in _MOTION_TOKENS:
            if token in oid_lower:
                # Extract room hint by stripping known suffixes
                area_hint = oid_lower
                for suffix in ("_occupancy", "_motion", "_presence", "_combo"):
                    area_hint = area_hint.replace(suffix, "")
                # Also strip sub-location qualifiers like _sink_motion
                for sub in ("_sink_motion", "_sink", "_ecobee"):
                    area_hint = area_hint.replace(sub, "")
                person_hint = person_name.lower() if person_name else None
                return ClassifiedSensor(
                    entity_id=entity_id,
                    kind=SensorKind.MOTION,
                    area_hint=area_hint,
                    person_hint=person_hint,
                )
        return ClassifiedSensor(entity_id=entity_id, kind=SensorKind.UNKNOWN)

    # Device trackers → BLE or GPS
    if domain == "device_tracker":
        is_ble = any(token in oid_lower for token in _BLE_TOKENS)
        if not is_ble:
            # Check for hardware/model pattern (e.g. "phone_darrell_15_pro")
            is_ble = any(p.search(oid_lower) for p in _BLE_MODEL_PATTERNS)

        if is_ble:
            person_hint = None
            if person_name and person_name.lower() in oid_lower:
                person_hint = person_name.lower()
            return ClassifiedSensor(
                entity_id=entity_id,
                kind=SensorKind.BLE,
                person_hint=person_hint,
            )
        # Default device_tracker → GPS (zone-based: home/not_home)
        return ClassifiedSensor(
            entity_id=entity_id,
            kind=SensorKind.GPS,
            person_hint=person_name.lower() if person_name else None,
        )

    return ClassifiedSensor(entity_id=entity_id, kind=SensorKind.UNKNOWN)


def classify_all(
    entity_ids: list[str], person_name: str = ""
) -> list[ClassifiedSensor]:
    """Classify a list of sensors."""
    return [classify(eid, person_name) for eid in entity_ids]


def extract_areas(sensors: list[ClassifiedSensor]) -> list[str]:
    """Extract the set of rooms from classified sensors.

    Rooms come from:
    - BLE tracker states (dynamic, resolved at runtime)
    - Motion sensor room_hints (static, from entity_id names)
    """
    rooms: set[str] = set()
    for s in sensors:
        if s.kind == SensorKind.MOTION and s.area_hint:
            rooms.add(s.area_hint)
    return sorted(rooms)
