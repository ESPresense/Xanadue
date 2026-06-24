"""Tests for the Bayesian inference engine."""

import os
import tempfile
import math
import pytest

from custom_components.xanadue.inference.bayesian import BayesianEngine
from custom_components.xanadue.inference.likelihoods import Observation, compute_likelihood
from custom_components.xanadue.inference.priors import PriorStore


@pytest.fixture
def tmp_priors_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "priors.json")


@pytest.fixture
def rooms():
    return ["family_room", "kitchen", "living_room", "master", "den"]


@pytest.fixture
def prior_store(tmp_priors_path, rooms):
    return PriorStore(priors_path=tmp_priors_path, rooms=rooms)


@pytest.fixture
def engine(prior_store, rooms):
    return BayesianEngine(rooms=rooms, prior_store=prior_store)


class TestLikelihoods:
    def test_ble_matching_room(self):
        obs = Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="kitchen",
            room="kitchen",
            confidence=0.9,
        )
        assert compute_likelihood(obs, "kitchen") == 0.9

    def test_ble_non_matching_room(self):
        obs = Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="kitchen",
            room="kitchen",
            confidence=0.9,
        )
        # Should be much lower than matching
        result = compute_likelihood(obs, "family_room")
        assert result < 0.5

    def test_motion_on_matching(self):
        obs = Observation(
            entity_id="binary_sensor.kitchen_occupancy",
            kind="motion",
            state="on",
            room="kitchen",
            age_seconds=5,
        )
        result = compute_likelihood(obs, "kitchen")
        assert result > 0.5

    def test_motion_on_non_matching(self):
        obs = Observation(
            entity_id="binary_sensor.kitchen_occupancy",
            kind="motion",
            state="on",
            room="kitchen",
            age_seconds=5,
        )
        result = compute_likelihood(obs, "family_room")
        assert result < 0.5

    def test_gps_home_uninformative(self):
        obs = Observation(
            entity_id="device_tracker.iphone",
            kind="gps",
            state="home",
        )
        # GPS home is uninformative at room level
        assert compute_likelihood(obs, "kitchen") == 1.0
        assert compute_likelihood(obs, "family_room") == 1.0

    def test_gps_not_home_near_zero(self):
        obs = Observation(
            entity_id="device_tracker.iphone",
            kind="gps",
            state="not_home",
        )
        assert compute_likelihood(obs, "kitchen") < 0.1


class TestBayesianEngine:
    def test_uniform_prior_no_data(self, engine):
        """With no corrections, posterior should be roughly uniform."""
        obs = [Observation(
            entity_id="device_tracker.iphone",
            kind="gps",
            state="home",
        )]
        result = engine.infer(obs)
        # With uniform prior and uninformative GPS, confidence should be low
        assert result.confidence < 0.5
        assert result.entropy > 1.0  # high uncertainty

    def test_ble_strong_signal(self, engine):
        """BLE pointing at a room should dominate."""
        obs = [Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="kitchen",
            room="kitchen",
            confidence=0.95,
            age_seconds=1,
        )]
        result = engine.infer(obs)
        assert result.room == "kitchen"
        assert result.confidence > 0.5

    def test_motion_only_inference(self, engine):
        """Motion in one room should point to that room (Piper's case)."""
        obs = [
            Observation(
                entity_id="binary_sensor.family_occupancy",
                kind="motion",
                state="on",
                room="family_room",
                age_seconds=10,
            ),
            Observation(
                entity_id="binary_sensor.kitchen_occupancy",
                kind="motion",
                state="off",
                room="kitchen",
                age_seconds=60,
            ),
        ]
        result = engine.infer(obs)
        assert result.room == "family_room"
        assert result.confidence > 0.3  # motion-only is weaker than BLE

    def test_correction_shifts_prior(self, engine, prior_store, rooms):
        """After corrections, the prior should influence the posterior."""
        # Add many corrections for family_room at the current hour
        for _ in range(20):
            prior_store.add_correction("family_room", weight=1.0)

        # Now even with just GPS (uninformative), family_room should win
        obs = [Observation(
            entity_id="device_tracker.iphone",
            kind="gps",
            state="home",
        )]
        result = engine.infer(obs)
        assert result.room == "family_room"
        assert result.confidence > 0.4

    def test_alternatives_populated(self, engine):
        obs = [Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="kitchen",
            room="kitchen",
            confidence=0.6,
            age_seconds=5,
        )]
        result = engine.infer(obs)
        assert len(result.alternatives) > 0
        assert all("room" in a and "probability" in a for a in result.alternatives)

    def test_entropy_decreases_with_strong_signal(self, engine):
        """Strong BLE signal should produce lower entropy than weak signals."""
        strong_obs = [Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="kitchen",
            room="kitchen",
            confidence=0.95,
            age_seconds=1,
        )]
        weak_obs = [Observation(
            entity_id="device_tracker.iphone",
            kind="gps",
            state="home",
        )]

        strong_result = engine.infer(strong_obs)
        weak_result = engine.infer(weak_obs)

        assert strong_result.entropy < weak_result.entropy

    def test_stale_observations_filtered(self, engine):
        """Observations older than max_age should be ignored."""
        stale_obs = [Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="kitchen",
            room="kitchen",
            confidence=0.95,
            age_seconds=999999,  # very stale
        )]
        result = engine.infer(stale_obs)
        # Stale BLE should have no effect → roughly uniform
        assert result.confidence < 0.5
