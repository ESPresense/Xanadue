"""Tests for the sensor classifier."""

import pytest
from custom_components.xanadue.classify import (
    classify,
    classify_all,
    extract_areas,
    SensorKind,
    ClassifiedSensor,
)


class TestClassify:
    def test_ble_tracker_phone(self):
        result = classify("device_tracker.phone_george_15_pro", "George")
        assert result.kind == SensorKind.BLE
        assert result.person_hint == "george"

    def test_ble_tracker_watch(self):
        result = classify("device_tracker.watch_georges_series_9", "George")
        assert result.kind == SensorKind.BLE

    def test_ble_tracker_espresense(self):
        result = classify("device_tracker.espresense_beacon_1", "George")
        assert result.kind == SensorKind.BLE

    def test_gps_tracker_plain_iphone(self):
        result = classify("device_tracker.georges_iphone", "George")
        assert result.kind == SensorKind.GPS
        assert result.person_hint == "george"

    def test_gps_tracker_not_ble(self):
        """A device_tracker without BLE tokens should be GPS."""
        result = classify("device_tracker.some_random_tracker", "")
        assert result.kind == SensorKind.GPS

    def test_motion_sensor_occupancy(self):
        result = classify("binary_sensor.family_occupancy", "George")
        assert result.kind == SensorKind.MOTION
        assert result.area_hint == "family"

    def test_motion_sensor_presence(self):
        result = classify("binary_sensor.family_presence_occupancy", "George")
        assert result.kind == SensorKind.MOTION

    def test_motion_sensor_kitchen_sink(self):
        """Complex motion sensor name should strip sub-location qualifiers."""
        result = classify("binary_sensor.kitchen_sink_motion_occupancy", "George")
        assert result.kind == SensorKind.MOTION
        assert "kitchen" in result.area_hint

    def test_unknown_domain(self):
        result = classify("sensor.temperature", "")
        assert result.kind == SensorKind.UNKNOWN

    def test_unknown_binary_sensor(self):
        result = classify("binary_sensor.door", "")
        assert result.kind == SensorKind.UNKNOWN


class TestClassifyAll:
    def test_mixed_sensors(self):
        entities = [
            "device_tracker.phone_george_15_pro",
            "device_tracker.watch_georges_series_9",
            "binary_sensor.family_occupancy",
            "binary_sensor.kitchen_occupancy",
            "device_tracker.georges_iphone",
        ]
        results = classify_all(entities, "George")
        assert len(results) == 5
        assert results[0].kind == SensorKind.BLE
        assert results[1].kind == SensorKind.BLE
        assert results[2].kind == SensorKind.MOTION
        assert results[3].kind == SensorKind.MOTION
        assert results[4].kind == SensorKind.GPS


class TestExtractRooms:
    def test_extract_rooms_from_motion(self):
        sensors = classify_all([
            "binary_sensor.family_occupancy",
            "binary_sensor.kitchen_occupancy",
            "binary_sensor.living_occupancy",
        ], "George")
        rooms = extract_areas(sensors)
        assert "family" in rooms
        assert "kitchen" in rooms
        assert "living" in rooms

    def test_no_rooms_from_ble_only(self):
        sensors = classify_all([
            "device_tracker.phone_george_15_pro",
        ], "George")
        rooms = extract_areas(sensors)
        assert rooms == []
