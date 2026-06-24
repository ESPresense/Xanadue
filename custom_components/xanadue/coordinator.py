"""Data update coordinator for Xanadue.

Subscribes to HA state changes for configured sensors, feeds them to the
Bayesian engine, and exposes the result as sensor state.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .classify import classify, classify_all, extract_areas, SensorKind
from .const import (
    CONF_NAME,
    CONF_SENSORS,
    DEFAULT_UPDATE_INTERVAL,
    DEFAULT_ENTROPY_THRESHOLD,
    AUTO_LABEL_WEIGHT,
    AUTO_LABEL_STABILITY_SECONDS,
    DOMAIN,
)
from .data.store import DataStore, get_data_dir
from .inference.bayesian import BayesianEngine, AreaEstimate
from .inference.likelihoods import Observation
from .inference.priors import PriorStore, hour_bucket

_LOGGER = logging.getLogger(__name__)


class XanadueCoordinator(DataUpdateCoordinator):
    """Coordinates sensor data → Bayesian inference → sensor state for one person."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        self.entry = entry
        self.hass = hass
        self._person_name: str = entry.data[CONF_NAME]
        self.slug: str = self._person_name.lower().strip().replace(" ", "_")
        self.sensor_ids: list[str] = entry.data.get(CONF_SENSORS, [])

        # Classify sensors
        self.classified = classify_all(self.sensor_ids, self.name)

        # Extract static areas from motion sensor names
        static_areas = extract_areas(self.classified)

        # Data storage
        self.data_dir = get_data_dir(hass.config.config_dir, self.slug)
        self.store = DataStore(self.data_dir)

        # Prior store (loads from disk or initializes)
        self.prior_store = PriorStore(
            priors_path=self.store.get_priors_path(),
            areas=static_areas or ["unknown"],
        )

        # Inference engine
        self.engine = BayesianEngine(
            areas=static_areas or ["unknown"],
            prior_store=self.prior_store,
        )

        # State tracking
        self._last_state_change: dict[str, float] = {}  # entity_id → timestamp
        self._auto_label_timer: Optional[asyncio.TimerHandle] = None
        self._stable_since: Optional[float] = None

        super().__init__(
            hass,
            _LOGGER,
            name=f"xanadue_{self.slug}",
            update_interval=timedelta(seconds=DEFAULT_UPDATE_INTERVAL),
        )

    @property
    def person_name(self) -> str:
        """Return the person's display name (not the coordinator logger name)."""
        return self._person_name

    async def async_config_entry_first_refresh(self) -> None:
        """First refresh + start listening to state changes."""
        # Start listening to state changes on configured sensors
        self._unsub = async_track_state_change_event(
            self.hass, self.sensor_ids, self._handle_state_change
        )
        await super().async_config_entry_first_refresh()
        _LOGGER.info(
            "[Xanadue] Coordinator started for '%s' with %d sensors, areas: %s",
            self._person_name,
            len(self.sensor_ids),
            self.engine.areas,
        )

    async def async_shutdown(self) -> None:
        """Clean up on unload."""
        if hasattr(self, "_unsub"):
            self._unsub()

    @callback
    def _handle_state_change(self, event) -> None:
        """Handle a state change event from HA."""
        entity_id = event.data.get("entity_id", "")
        new_state = event.data.get("new_state")
        if new_state is None:
            return
        if new_state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
            return

        self._last_state_change[entity_id] = time.time()

        # Trigger an immediate refresh
        self.async_update_listeners()

    def _collect_observations(self) -> list[Observation]:
        """Collect current observations from all configured sensors."""
        observations = []
        now = time.time()

        for sensor in self.classified:
            state_obj = self.hass.states.get(sensor.entity_id)
            if state_obj is None or state_obj.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
                continue

            last_change = self._last_state_change.get(
                sensor.entity_id,
                dt_util.as_timestamp(state_obj.last_changed) if state_obj.last_changed else now,
            )
            age = now - last_change

            if sensor.kind == SensorKind.BLE:
                # BLE tracker: state is the area name, confidence from attributes
                confidence = state_obj.attributes.get("confidence", 0.8)
                observations.append(Observation(
                    entity_id=sensor.entity_id,
                    kind="ble",
                    state=state_obj.state,
                    area=state_obj.state,
                    confidence=confidence,
                    age_seconds=age,
                ))
            elif sensor.kind == SensorKind.MOTION:
                observations.append(Observation(
                    entity_id=sensor.entity_id,
                    kind="motion",
                    state=state_obj.state,
                    area=sensor.area_hint,
                    age_seconds=age,
                ))
            elif sensor.kind == SensorKind.GPS:
                observations.append(Observation(
                    entity_id=sensor.entity_id,
                    kind="gps",
                    state=state_obj.state,
                    age_seconds=age,
                ))

        return observations

    async def _async_update_data(self) -> AreaEstimate:
        """Fetch the latest data and run inference."""
        observations = self._collect_observations()

        estimate = self.engine.infer(observations)

        # Log posterior
        log_entry = {
            "ts": estimate.timestamp,
            "area": estimate.area,
            "confidence": estimate.confidence,
            "entropy": estimate.entropy,
            "posterior": {k: round(v, 4) for k, v in estimate.posterior.items()},
        }
        try:
            self.store.append_posterior_log(log_entry)
        except Exception:
            pass  # don't let logging break inference

        # Check for auto-label conditions
        self._check_auto_label(estimate)

        return estimate

    def _check_auto_label(self, estimate: AreaEstimate) -> None:
        """Check if conditions are met for auto-labeling.

        Auto-label fires when:
        - Entropy < threshold
        - At least 2 independent signal sources agree
        - State has been stable for AUTO_LABEL_STABILITY_SECONDS
        """
        now = time.time()

        if estimate.entropy < DEFAULT_ENTROPY_THRESHOLD:
            # Check signal agreement
            source_kinds = set()
            for obs in estimate.observations_used:
                source_kinds.add(obs.get("kind"))

            if len(source_kinds) >= 2:
                # Track stability
                if self._stable_since is None:
                    self._stable_since = now
                elif now - self._stable_since >= AUTO_LABEL_STABILITY_SECONDS:
                    # Stable long enough → auto-label
                    self.prior_store.add_correction(
                        area=estimate.area,
                        weight=AUTO_LABEL_WEIGHT,
                    )
                    self.store.append_ground_truth(
                        area=estimate.area,
                        source="auto",
                        weight=AUTO_LABEL_WEIGHT,
                    )
                    _LOGGER.debug(
                        "[Xanadue] Auto-label: %s → %s (conf=%.2f, entropy=%.2f)",
                        self.slug,
                        estimate.area,
                        estimate.confidence,
                        estimate.entropy,
                    )
                    return
        else:
            self._stable_since = None

    async def apply_correction(self, area: str, weight: float = 1.0) -> None:
        """Apply a manual correction and refresh."""
        self.prior_store.add_correction(area=area, weight=weight)
        self.store.append_ground_truth(area=area, source="manual", weight=weight)

        # Add area to engine's area space if new
        if area not in self.engine.areas:
            self.engine.areas.append(area)
            self.prior_store.areas.append(area)

        # Trigger refresh
        await self.async_refresh()
