"""Config flow for Xanadue."""

from __future__ import annotations

from homeassistant import config_entries
from homeassistant.core import callback
import voluptuous as vol

from .const import CONF_NAME, CONF_SENSORS, DOMAIN

import logging

_LOGGER = logging.getLogger(__name__)


def _slug(name: str) -> str:
    """Derive a slug from a name."""
    return name.lower().strip().replace(" ", "_")


class XanadueConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Xanadue."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Step 1: name + sensors."""
        errors = {}

        if user_input is not None:
            name = user_input[CONF_NAME]
            slug = _slug(name)

            # Prevent duplicate Xanadues for the same name
            for entry in self._async_current_entries():
                if entry.data.get(CONF_NAME, "").lower() == name.lower():
                    errors["base"] = "already_configured"
                    break

            if not errors:
                return self.async_create_entry(
                    title=f"Xanadue: {name}",
                    data={
                        CONF_NAME: name,
                        CONF_SENSORS: user_input[CONF_SENSORS],
                    },
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_NAME): str,
                    vol.Required(CONF_SENSORS): vol.All(
                        vol.ensure_list, [str]
                    ),
                }
            ),
            errors=errors,
            description_placeholders={
                "name_example": "Darrell",
                "sensors_example": "device_tracker.phone_darrell_15_pro, binary_sensor.family_occupancy, ...",
            },
        )

    async def async_step_import(self, import_config=None):
        """Import from YAML configuration."""
        if import_config is None:
            return

        name = import_config[CONF_NAME]
        slug = _slug(name)

        # Check for existing entry
        for entry in self._async_current_entries():
            if entry.data.get(CONF_NAME, "").lower() == name.lower():
                # Update existing entry
                self.hass.config_entries.async_update_entry(
                    entry,
                    data=import_config,
                )
                return self.async_abort(reason="already_configured")

        return self.async_create_entry(
            title=f"Xanadue: {name}",
            data=import_config,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> "XanadueOptionsFlow":
        """Get options flow."""
        return XanadueOptionsFlow(config_entry)


class XanadueOptionsFlow(config_entries.OptionsFlow):
    """Options flow for editing sensors list."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Manage the sensors list."""
        current_sensors = self.config_entry.data.get(CONF_SENSORS, [])

        if user_input is not None:
            # Update the config entry data
            new_data = dict(self.config_entry.data)
            new_data[CONF_SENSORS] = user_input[CONF_SENSORS]
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data=new_data,
            )
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_SENSORS,
                        default=current_sensors,
                    ): vol.All(vol.ensure_list, [str]),
                }
            ),
        )
