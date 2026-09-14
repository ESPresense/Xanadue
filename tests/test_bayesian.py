"""Tests for the Bayesian inference engine."""

import os
import tempfile
import math
import pytest

from custom_components.xanadue.inference.bayesian import BayesianEngine
from custom_components.xanadue.inference.likelihoods import (
    Observation,
    compute_likelihood,
    gps_indicates_away,
    gps_indicates_home,
)
from custom_components.xanadue.inference.priors import PriorStore


@pytest.fixture
def tmp_priors_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "priors.json")


@pytest.fixture
def areas():
    return ["family_room", "kitchen", "living_room", "master", "den"]


@pytest.fixture
def prior_store(tmp_priors_path, areas):
    return PriorStore(priors_path=tmp_priors_path, areas=areas)


@pytest.fixture
def engine(prior_store, areas):
    return BayesianEngine(areas=areas, prior_store=prior_store)


class TestLikelihoods:
    def test_ble_matching_room(self):
        obs = Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="kitchen",
            area="kitchen",
            confidence=0.9,
        )
        assert compute_likelihood(obs, "kitchen") == 0.9

    def test_ble_non_matching_room(self):
        obs = Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="kitchen",
            area="kitchen",
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
            area="kitchen",
            age_seconds=5,
        )
        result = compute_likelihood(obs, "kitchen")
        assert result > 0.5

    def test_motion_on_non_matching(self):
        obs = Observation(
            entity_id="binary_sensor.kitchen_occupancy",
            kind="motion",
            state="on",
            area="kitchen",
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
        # GPS home is uninformative at area level
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
        """BLE pointing at a area should dominate."""
        obs = [Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="kitchen",
            area="kitchen",
            confidence=0.95,
            age_seconds=1,
        )]
        result = engine.infer(obs)
        assert result.area == "kitchen"
        assert result.confidence > 0.5

    def test_motion_only_inference(self, engine):
        """Motion in one area should point to that area (Piper's case)."""
        obs = [
            Observation(
                entity_id="binary_sensor.family_occupancy",
                kind="motion",
                state="on",
                area="family_room",
                age_seconds=10,
            ),
            Observation(
                entity_id="binary_sensor.kitchen_occupancy",
                kind="motion",
                state="off",
                area="kitchen",
                age_seconds=60,
            ),
        ]
        result = engine.infer(obs)
        assert result.area == "family_room"
        assert result.confidence > 0.3  # motion-only is weaker than BLE

    def test_correction_shifts_prior(self, engine, prior_store, areas):
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
        assert result.area == "family_room"
        assert result.confidence > 0.4

    def test_alternatives_populated(self, engine):
        obs = [Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="kitchen",
            area="kitchen",
            confidence=0.6,
            age_seconds=5,
        )]
        result = engine.infer(obs)
        assert len(result.alternatives) > 0
        assert all("area" in a and "probability" in a for a in result.alternatives)

    def test_entropy_decreases_with_strong_signal(self, engine):
        """Strong BLE signal should produce lower entropy than weak signals."""
        strong_obs = [Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="kitchen",
            area="kitchen",
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
            area="kitchen",
            confidence=0.95,
            age_seconds=999999,  # very stale
        )]
        result = engine.infer(stale_obs)
        # Stale BLE should have no effect → roughly uniform
        assert result.confidence < 0.5


class TestNoInformationPath:
    """Issue #2: `infer()` must not name a room when the observations carry no
    area information. The winner on a tied posterior is an artifact of sort
    order, and publishing it reported a room for a person who was not home.
    """

    def test_gps_away_zone_returns_unknown(self, engine):
        """A named away zone (`work`) is away, not "home, area unknown"."""
        obs = [Observation(
            entity_id="device_tracker.iphone",
            kind="gps",
            state="work",
            age_seconds=274.9,
        )]
        result = engine.infer(obs)
        assert result.area == "unknown"
        assert result.informative is False

    def test_gps_not_home_returns_unknown(self, engine):
        obs = [Observation(
            entity_id="device_tracker.iphone",
            kind="gps",
            state="not_home",
            age_seconds=60,
        )]
        assert engine.infer(obs).area == "unknown"

    def test_no_observations_returns_unknown(self, engine):
        assert engine.infer([]).area == "unknown"

    def test_ble_area_outside_model_returns_unknown(self, engine):
        """An area not in the model leaks equally to every candidate → no info."""
        obs = [Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="not_a_configured_area",
            area="not_a_configured_area",
            confidence=0.8,
            age_seconds=5,
        )]
        assert engine.infer(obs).area == "unknown"

    def test_motion_off_only_returns_unknown(self, engine):
        """Off-motion is documented as uninformative; alone it must not name a room."""
        obs = [Observation(
            entity_id="binary_sensor.kitchen_occupancy",
            kind="motion",
            state="off",
            area="kitchen",
            age_seconds=30,
        )]
        assert engine.infer(obs).area == "unknown"

    def test_stale_ble_returns_unknown(self, engine):
        obs = [Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="kitchen",
            area="kitchen",
            confidence=0.95,
            age_seconds=999999,
        )]
        assert engine.infer(obs).area == "unknown"

    def test_away_zone_suppresses_learned_prior(self, prior_store, areas):
        """A learned prior must not resurrect a room for someone who is away.

        The posterior here has a strict winner (the prior's mode), so the tie
        test alone would publish it. GPS placing the person elsewhere is
        stronger evidence than any room-level prior.
        """
        ts = 1_700_000_000
        for _ in range(40):
            prior_store.add_correction("family_room", weight=1.0, timestamp=ts)
        engine = BayesianEngine(areas=areas, prior_store=prior_store)

        # Sanity: the prior really is peaked, and GPS-home still uses it.
        home = engine.infer(
            [Observation(entity_id="device_tracker.iphone", kind="gps", state="home")],
            timestamp=ts,
        )
        assert home.area == "family_room"

        away = engine.infer(
            [Observation(
                entity_id="device_tracker.iphone",
                kind="gps",
                state="work",
                age_seconds=274.9,
            )],
            timestamp=ts,
        )
        assert away.area == "unknown"

    def test_informative_estimates_still_name_a_room(self, engine):
        """The sentinel must not swallow real evidence."""
        ble = engine.infer([Observation(
            entity_id="device_tracker.phone",
            kind="ble",
            state="kitchen",
            area="kitchen",
            confidence=0.9,
            age_seconds=1,
        )])
        assert ble.area == "kitchen"
        assert ble.informative is True

        motion = engine.infer([Observation(
            entity_id="binary_sensor.den_occupancy",
            kind="motion",
            state="on",
            area="den",
            age_seconds=5,
        )])
        assert motion.area == "den"

    def test_local_evidence_outranks_lagging_away_zone(self, engine):
        """Zone updates lag; a BLE tag in the kitchen means they are in the kitchen.

        Someone who just got home still reads as "work" on a polled tracker
        while the room-level signal already sees them inside.
        """
        obs = [
            Observation(
                entity_id="device_tracker.iphone",
                kind="gps",
                state="work",
                age_seconds=274.9,
            ),
            Observation(
                entity_id="device_tracker.phone",
                kind="ble",
                state="kitchen",
                area="kitchen",
                confidence=0.9,
                age_seconds=2,
            ),
        ]
        assert engine.infer(obs).area == "kitchen"

    def test_away_zone_with_only_off_motion_stays_unknown(self, engine):
        """Off-motion is not evidence of presence, so it cannot override away."""
        obs = [
            Observation(
                entity_id="device_tracker.iphone",
                kind="gps",
                state="work",
                age_seconds=274.9,
            ),
            Observation(
                entity_id="binary_sensor.kitchen_occupancy",
                kind="motion",
                state="off",
                area="kitchen",
                age_seconds=30,
            ),
        ]
        assert engine.infer(obs).area == "unknown"

    def test_alternatives_populated_when_uninformative(self, engine):
        """Callers read `alternatives` for debugging; keep it populated."""
        result = engine.infer([])
        assert len(result.alternatives) > 0
        assert all("area" in a and "probability" in a for a in result.alternatives)


class TestGpsZoneClassification:
    """`gps_likelihood` must classify zones by whether they are the home zone,
    not by matching the literal strings `not_home`/`away` (issue #2)."""

    def test_named_away_zone_is_away(self):
        for state in ("work", "school", "shops", "gym"):
            obs = Observation(
                entity_id="device_tracker.iphone", kind="gps", state=state
            )
            assert compute_likelihood(obs, "kitchen") < 0.1, state
            assert gps_indicates_away(state) is True, state

    def test_home_zone_is_home(self):
        assert gps_indicates_home("home") is True
        assert gps_indicates_away("home") is False

    def test_missing_states_are_not_away(self):
        """An unknown tracker must not assert the person has left the house."""
        for state in ("", "unknown", "unavailable", None):
            assert gps_indicates_away(state) is False, state
            obs = Observation(
                entity_id="device_tracker.iphone", kind="gps", state=state or ""
            )
            assert compute_likelihood(obs, "kitchen") == 1.0, state

    def test_explicit_not_home_and_away_still_away(self):
        for state in ("not_home", "away", "NOT_HOME", " away "):
            assert gps_indicates_away(state) is True, state

    def test_named_zone_is_case_insensitive(self):
        obs = Observation(
            entity_id="device_tracker.iphone", kind="gps", state="Work"
        )
        assert compute_likelihood(obs, "kitchen") < 0.1
