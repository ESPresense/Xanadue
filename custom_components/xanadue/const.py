"""Constants for Xanadue."""

DOMAIN = "xanadue"

# Config keys
CONF_NAME = "name"
CONF_SENSORS = "sensors"

# Defaults
DEFAULT_UPDATE_INTERVAL = 5  # seconds
DEFAULT_ENTROPY_THRESHOLD = 0.5  # nats
DEFAULT_SMOOTHING_ALPHA = 2.0  # Laplace smoothing
AUTO_LABEL_WEIGHT = 0.1
MANUAL_WEIGHT = 1.0
AUTO_LABEL_STABILITY_SECONDS = 300  # 5 minutes

# Entity attributes
ATTR_CONFIDENCE = "confidence"
ATTR_ENTROPY = "entropy"
ATTR_ALTERNATIVES = "alternatives"
ATTR_OBSERVATIONS_USED = "observations_used"
ATTR_LAST_UPDATE = "last_update"
ATTR_XANADUE_SLUG = "xanadue_slug"
