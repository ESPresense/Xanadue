"""Handle xanadue.correct service calls.

Updates priors and triggers a coordinator refresh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import logging

from ..const import DOMAIN, MANUAL_WEIGHT

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant, ServiceCall

_LOGGER = logging.getLogger(__name__)


async def handle_correction(hass: "HomeAssistant", call: "ServiceCall") -> None:
    """Process a xanadue.correct service call.

    Expected service data:
        xanadue: str  (the name/slug of the Xanadue instance)
        area: str     (the area the person is actually in)
        duration: int (optional — how long they've been there, in seconds)
    """
    # Lazy import to avoid circular dependency: coordinator → data → correct → coordinator
    from ..coordinator import XanadueCoordinator  # noqa: F811

    target_name = call.data["xanadue"].lower().strip().replace(" ", "_")
    area = call.data["area"]
    duration = call.data.get("duration")

    # Find the coordinator for this Xanadue instance
    coordinator: XanadueCoordinator | None = None
    for entry_id, coord in hass.data.get(DOMAIN, {}).items():
        if coord.slug == target_name:
            coordinator = coord
            break

    if coordinator is None:
        _LOGGER.warning(
            "xanadue.correct: no Xanadue instance found for '%s'. "
            "Available: %s",
            target_name,
            [c.slug for c in hass.data.get(DOMAIN, {}).values()],
        )
        return

    # Weight correction by duration if provided (longer stay = stronger signal)
    weight = MANUAL_WEIGHT
    if duration and duration > 0:
        weight = MANUAL_WEIGHT * min(duration / 600.0, 5.0)  # cap at 5x

    # Apply correction
    await coordinator.apply_correction(area, weight=weight)

    _LOGGER.info(
        "[Xanadue] Correction applied: %s → %s (weight=%.2f)",
        target_name,
        area,
        weight,
    )
