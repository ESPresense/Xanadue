"""Sensor platform for Xanadue."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    ATTR_CONFIDENCE,
    ATTR_ENTROPY,
    ATTR_ALTERNATIVES,
    ATTR_OBSERVATIONS_USED,
    ATTR_LAST_UPDATE,
    ATTR_XANADUE_SLUG,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Xanadue sensor."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([XanadueAreaSensor(coordinator, entry)])


class XanadueAreaSensor(CoordinatorEntity, SensorEntity):
    """Sensor that reports the inferred area for a person."""

    _attr_has_entity_name = True
    _attr_icon = "mdi:account-location"

    def __init__(self, coordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self.coordinator = coordinator
        self._attr_unique_id = f"xanadue_{coordinator.slug}"
        self._attr_name = f"{coordinator.name} Current Area"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, coordinator.slug)},
            "name": f"Xanadue: {coordinator.name}",
            "manufacturer": "ESPresense",
            "model": "Xanadue",
            "sw_version": "0.1.0",
        }

    @property
    def native_value(self) -> str:
        """Return the inferred area."""
        estimate = self.coordinator.data
        if estimate is None:
            return "unknown"
        return estimate.area

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional state attributes."""
        estimate = self.coordinator.data
        if estimate is None:
            return {
                ATTR_CONFIDENCE: 0,
                ATTR_ENTROPY: 0,
                ATTR_ALTERNATIVES: [],
                ATTR_OBSERVATIONS_USED: [],
                ATTR_LAST_UPDATE: None,
                ATTR_XANADUE_SLUG: self.coordinator.slug,
            }

        return {
            ATTR_CONFIDENCE: estimate.confidence,
            ATTR_ENTROPY: estimate.entropy,
            ATTR_ALTERNATIVES: estimate.alternatives,
            ATTR_OBSERVATIONS_USED: estimate.observations_used,
            ATTR_LAST_UPDATE: datetime.fromtimestamp(
                estimate.timestamp, tz=timezone.utc
            ).isoformat(),
            ATTR_XANADUE_SLUG: self.coordinator.slug,
        }
