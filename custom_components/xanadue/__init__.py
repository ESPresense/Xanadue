"""Xanadue — per-person area inference for Home Assistant.

HA imports are lazy so pure-Python modules (classify, inference, data) can
be imported in isolation for testing without homeassistant installed.
"""

from .const import DOMAIN, CONF_NAME, CONF_SENSORS

import logging
_LOGGER = logging.getLogger(__name__)

# Defer HA imports until setup hooks are called
_PLATFORMS = ["device_tracker"]


async def async_setup(hass, config):
    """Set up via YAML configuration.yaml."""
    # Lazy import to keep pure modules testable
    import voluptuous as vol

    if DOMAIN not in config:
        return True

    for xanadue_config in config[DOMAIN]:
        hass.async_create_task(
            hass.config_entries.flow.async_init(
                DOMAIN,
                context={"source": "import"},
                data=xanadue_config,
            )
        )

    return True


async def async_setup_entry(hass, entry):
    """Set up a Xanadue config entry."""
    from homeassistant.core import ServiceCall
    import voluptuous as vol

    from .coordinator import XanadueCoordinator
    from .data.correct import handle_correction

    coordinator = XanadueCoordinator(hass, entry)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator

    await coordinator.async_config_entry_first_refresh()

    await hass.config_entries.async_forward_entry_setups(entry, _PLATFORMS)

    # Register services on first entry
    if not hass.services.has_service(DOMAIN, "correct"):
        async def handle_correct(call: "ServiceCall") -> None:
            await handle_correction(hass, call)

        hass.services.async_register(
            DOMAIN,
            "correct",
            handle_correct,
            schema=vol.Schema(
                {
                    vol.Required("xanadue"): str,
                    vol.Required("area"): str,
                    vol.Optional("duration"): int,
                }
            ),
        )

    # --- Sensor management services ---
    # Three services sharing one helper:
    #   xanadue.set_sensors  — replace entire list (bulk)
    #   xanadue.add_sensor   — append a single sensor
    #   xanadue.remove_sensor — remove a single sensor

    async def _find_entry(hass, name: str):
        """Find a Xanadue config entry by person name (case-insensitive)."""
        for e in hass.config_entries.async_entries(DOMAIN):
            if e.data.get("name", "").lower() == name.lower():
                return e
        return None

    async def _update_sensors(hass, entry, new_sensors: list[str]) -> None:
        """Write new sensor list to config entry and reload."""
        new_data = dict(entry.data)
        new_data["sensors"] = new_sensors
        hass.config_entries.async_update_entry(entry, data=new_data)
        await hass.config_entries.async_reload(entry.entry_id)

    if not hass.services.has_service(DOMAIN, "set_sensors"):
        async def handle_set_sensors(call: "ServiceCall") -> None:
            name = call.data["name"]
            sensors = list(call.data["sensors"])
            entry = await _find_entry(hass, name)
            if entry is None:
                _LOGGER.warning("[Xanadue] set_sensors: no entry for '%s'", name)
                return
            await _update_sensors(hass, entry, sensors)
            _LOGGER.info("[Xanadue] set_sensors '%s': %d sensors", name, len(sensors))

        hass.services.async_register(
            DOMAIN, "set_sensors", handle_set_sensors,
            schema=vol.Schema({
                vol.Required("name"): str,
                vol.Required("sensors"): [str],
            }),
        )

    if not hass.services.has_service(DOMAIN, "add_sensor"):
        async def handle_add_sensor(call: "ServiceCall") -> None:
            name = call.data["name"]
            sensor = call.data["sensor"]
            entry = await _find_entry(hass, name)
            if entry is None:
                _LOGGER.warning("[Xanadue] add_sensor: no entry for '%s'", name)
                return
            current = list(entry.data.get("sensors", []))
            if sensor in current:
                _LOGGER.info("[Xanadue] add_sensor '%s': '%s' already present", name, sensor)
                return
            current.append(sensor)
            await _update_sensors(hass, entry, current)
            _LOGGER.info("[Xanadue] add_sensor '%s': added '%s' (%d total)", name, sensor, len(current))

        hass.services.async_register(
            DOMAIN, "add_sensor", handle_add_sensor,
            schema=vol.Schema({
                vol.Required("name"): str,
                vol.Required("sensor"): str,
            }),
        )

    if not hass.services.has_service(DOMAIN, "remove_sensor"):
        async def handle_remove_sensor(call: "ServiceCall") -> None:
            name = call.data["name"]
            sensor = call.data["sensor"]
            entry = await _find_entry(hass, name)
            if entry is None:
                _LOGGER.warning("[Xanadue] remove_sensor: no entry for '%s'", name)
                return
            current = list(entry.data.get("sensors", []))
            if sensor not in current:
                _LOGGER.info("[Xanadue] remove_sensor '%s': '%s' not present", name, sensor)
                return
            current.remove(sensor)
            await _update_sensors(hass, entry, current)
            _LOGGER.info("[Xanadue] remove_sensor '%s': removed '%s' (%d total)", name, sensor, len(current))

        hass.services.async_register(
            DOMAIN, "remove_sensor", handle_remove_sensor,
            schema=vol.Schema({
                vol.Required("name"): str,
                vol.Required("sensor"): str,
            }),
        )

    return True


async def async_unload_entry(hass, entry):
    """Unload a Xanadue config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, _PLATFORMS)
    if unload_ok:
        coordinator = hass.data[DOMAIN].pop(entry.entry_id)
        await coordinator.async_shutdown()
    return unload_ok
