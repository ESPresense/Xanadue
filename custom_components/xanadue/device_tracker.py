"""Device tracker platform for Xanadue.

Publishes one `device_tracker.<slug>` per Xanadue config entry.
The state is the inferred area name (e.g. "family_room"); this composes
natively with HA's `person.*` entities, maps, and presence automations.

Rich inference metadata (confidence, entropy, alternatives, observations_used)
is packed into a single `xanadue` sub-attribute to avoid polluting the
device_tracker namespace.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from homeassistant.components.device_tracker import SourceType, TrackerEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Xanadue device_tracker entity."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([XanadueTracker(coordinator, entry)])


class XanadueTracker(CoordinatorEntity, TrackerEntity):
    """A device_tracker entity representing one Xanadue person."""

    _attr_icon = "mdi:account-location"

    def __init__(self, coordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self.coordinator = coordinator
        self._attr_unique_id = f"xanadue_{coordinator.slug}"
        # Use explicit name (not has_entity_name) to avoid HA compounding
        # device name + entity name into a mangled entity_id like
        # device_tracker.xanadue_xanadue_darrell_xanadue_darrell_location
        self._attr_name = f"Xanadue: {coordinator.person_name}"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, coordinator.slug)},
            "name": f"Xanadue: {coordinator.person_name}",
            "manufacturer": "ESPresense",
            "model": "Xanadue",
            "sw_version": "0.2.10",
        }

    @property
    def source_type(self) -> SourceType:
        """Return the source type of this tracker."""
        return SourceType.ROUTER  # fused signal source, not a single type

    @property
    def location_name(self) -> str | None:
        """Return the inferred area as the device location.

        Returns the Bayesian best-guess area (e.g. "family_room").
        Falls back to the GPS zone ("home", "not_home") when room-level
        signals are unavailable, so the tracker is always meaningful.
        """
        estimate = self.coordinator.data
        if estimate is None:
            return None
        # If engine has a real area, use it
        if estimate.area and estimate.area != "unknown":
            return estimate.area
        # Fall back to GPS zone when no room-level data
        for obs in estimate.observations_used:
            if obs.get("kind") == "gps":
                return obs.get("observed")
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return inference metadata under a single namespaced attribute."""
        estimate = self.coordinator.data
        if estimate is None:
            return {
                "xanadue": {
                    "slug": self.coordinator.slug,
                    "confidence": 0.0,
                    "entropy": 0.0,
                    "alternatives": [],
                    "observations_used": [],
                    "last_update": None,
                }
            }

        return {
            "xanadue": {
                "slug": self.coordinator.slug,
                "confidence": estimate.confidence,
                "entropy": estimate.entropy,
                "alternatives": estimate.alternatives,
                "observations_used": estimate.observations_used,
                "last_update": datetime.fromtimestamp(
                    estimate.timestamp, tz=timezone.utc
                ).isoformat(),
            }
        }

    @callback
    def _handle_coordinator_update(self) -> None:
        """Forward coordinator updates to HA's entity state."""
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """Register state update callback."""
        await super().async_added_to_hass()
        self.async_on_remove(
            self.coordinator.async_add_listener(self._handle_coordinator_update)
        )
