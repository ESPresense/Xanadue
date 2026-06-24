# Xanadue

> *Xanadue — Xanadu, but you can afford it.*

In the late 1990s, Bill Gates built Xanadu 2.0 — a smart home that knew who was in every room. It cost over $80 million.

**Xanadue** is named after that house. It's a Home Assistant integration that fuses BLE, motion, GPS, and person entities into per-person area assignments with confidence — so you can build the kind of ambient, area-aware automations that Xanadu 2.0 made famous. What you do with that signal is up to you.

It is, emphatically, free.

## How it works

Each Xanadue instance tracks one person. Give it a name and a list of sensors — Xanadue handles the rest:

```yaml
xanadue:
  - name: George
    sensors:
      - device_tracker.phone_george
      - device_tracker.watch_george
      - binary_sensor.family_room_occupancy
      - binary_sensor.kitchen_occupancy
      - binary_sensor.living_room_occupancy
      - binary_sensor.bedroom_occupancy
      - binary_sensor.office_occupancy
      - device_tracker.george_phone
```

The integration fuses those signals with a Bayesian model, producing one sensor per person:

| Entity | State | Attributes |
|---|---|---|
| `device_tracker.george` | `family_room` | `source_type`, `xanadue: {confidence, entropy, alternatives, observations_used, slug}` |

The state is the inferred area name, which composes directly with HA's `person.*` entities — add `device_tracker.george` to your `person.george` entity's `device_trackers` list and `person.george` becomes driven by Xanadue.

## Teaching Xanadue

Xanadue learns from corrections. When it's wrong, tell it:

```yaml
service: xanadue.correct
data:
  xanadue: george
  area: kitchen
```

That correction feeds the time-of-day priors. After a few evenings of correcting "10 PM → family area," the prior learns your routine. No priors UI, no YAML tuning — just tell it where you are and it gets smarter.

## Installation

### HACS

1. Add this repo as a custom repository in HACS (category: Integration)
2. Install
3. Restart Home Assistant
4. Add via Settings → Devices & Services → Add Integration → Xanadue

### Manual

1. Copy `custom_components/xanadue/` to your HA `custom_components/` directory
2. Restart Home Assistant

## License

MIT
