"""Xanadue — per-person area inference for Home Assistant.

HA imports are lazy so pure-Python modules (classify, inference, data) can
be imported in isolation for testing without homeassistant installed.
"""

from .const import DOMAIN, CONF_NAME, CONF_SENSORS

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

    return True


async def async_unload_entry(hass, entry):
    """Unload a Xanadue config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, _PLATFORMS)
    if unload_ok:
        coordinator = hass.data[DOMAIN].pop(entry.entry_id)
        await coordinator.async_shutdown()
    return unload_ok
