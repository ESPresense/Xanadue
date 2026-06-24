# Xanadue

> *Xanadue — Xanadu, but you can afford it.*

In the late 1990s, Bill Gates built Xanadu 2.0 — a smart home that knew who was in every room. It cost over $80 million.

**Xanadue** is named after that house. It's a Home Assistant integration that fuses BLE, motion, GPS, and person entities into per-person area assignments with confidence — so you can build the kind of ambient, area-aware automations that Xanadu 2.0 made famous. What you do with that signal is up to you.

It is, emphatically, free.

## How it works

Each Xanadue instance tracks one person. Give it a name and a list of sensors — Xanadue handles the rest:

```yaml
xanadue:
  - name: Darrell
    sensors:
      - device_tracker.phone_darrell_15_pro
      - device_tracker.watch_darrells_series_9
      - binary_sensor.family_occupancy
      - binary_sensor.kitchen_occupancy
      - binary_sensor.living_occupancy
      - binary_sensor.master_occupancy
      - binary_sensor.den_occupancy
      - device_tracker.darrells_iphone
```

The integration fuses those signals with a Bayesian model, producing one sensor per person:

| Entity | State | Attributes |
|---|---|---|
| `sensor.xanadue_darrell_current_area` | `family_room` | `confidence`, `entropy`, `alternatives`, `observations_used` |

## Teaching Xanadue

Xanadue learns from corrections. When it's wrong, tell it:

```yaml
service: xanadue.correct
data:
  xanadue: darrell
  area: kitchen
```

That correction feeds the time-of-day priors. After a few evenings of correcting "10 PM → family room," the prior learns your routine. No priors UI, no YAML tuning — just tell it where you are and it gets smarter.

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
