"""Bayesian area inference engine.

Maintains a categorical posterior over areas for a single person,
updated incrementally as new observations arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
import time

from .likelihoods import (
    Observation,
    compute_likelihood,
    gps_indicates_away,
)
from .priors import PriorStore, hour_bucket

# The sentinel `area` for "no observation carried area information".
UNKNOWN_AREA = "unknown"

# Two posteriors within this relative difference of the maximum are considered
# tied, so neither is a winner. Compared against the max (not an absolute
# epsilon) so the test behaves the same for a 5-area and a 20-area deployment.
TIE_RELATIVE_EPSILON = 1e-9


def tied_at_max(posterior: dict[str, float]) -> int:
    """Number of areas sharing the maximum posterior probability.

    On a fully tied posterior this is the whole area list, and the argument
    passed to ``max()`` is decided by dict ordering rather than by evidence.
    """
    if not posterior:
        return 0
    top = max(posterior.values())
    return sum(1 for p in posterior.values() if top - p <= abs(top) * TIE_RELATIVE_EPSILON)


def is_uninformative(posterior: dict[str, float]) -> bool:
    """True when the posterior has no strict winner, so no area can be named.

    This asks whether any observation actually discriminated between areas. A
    strict winner (exactly one area at the maximum) means it did; a tie at the
    maximum means it did not, and the "winner" is an artifact of sort order.

    Deliberately *not* ``confidence <= 1/N + eps`` or ``entropy >= ln(N) - eps``:
    those both fail on real inputs. A lone motion-OFF reading yields
    ``0.0952 > 1/11`` over 11 areas while still being a meaningless location, and
    the BLE leak spreads mass unevenly once ``N`` grows, so neither the
    probability nor the entropy comparison lines up with "no information". The
    tie test is exact, count-independent, and catches every known path.
    """
    return tied_at_max(posterior) > 1


def has_away_observation(fresh_obs: list[Observation]) -> bool:
    """True when a GPS observation puts the person outside the house."""
    return any(
        o.kind == "gps" and gps_indicates_away(o.state)
        for o in fresh_obs
    )


def has_room_level_evidence(fresh_obs: list[Observation], areas: list[str]) -> bool:
    """True when an observation directly places the person in a known area.

    A BLE reading for an area in the model, or motion that is currently ON,
    means the person's device is physically in that room. That is direct local
    evidence, and it outranks a GPS zone: zone updates lag by minutes (iCloud3
    polls), so a person who just walked in still reads as "work" for a while.
    """
    return any(
        (o.kind == "ble" and o.area in areas)
        or (o.kind == "motion" and o.state == "on")
        for o in fresh_obs
    )


@dataclass
class AreaEstimate:
    """The output of a Bayesian area inference step."""

    area: str                          # best guess, or UNKNOWN_AREA when the posterior is uninformative
    confidence: float                  # posterior probability of best guess
    entropy: float                     # Shannon entropy in nats
    alternatives: list[dict]           # [{area, probability}, ...] sorted desc
    observations_used: list[dict]      # [{source, observed, weight}, ...]
    posterior: dict[str, float]        # full posterior distribution
    timestamp: float                   # when this estimate was computed

    @property
    def informative(self) -> bool:
        """True when the observations actually discriminated between areas.

        A uniform posterior means every observation was equally likely in every
        area (GPS to a named away zone, `not_home`, a BLE area outside the model,
        motion-off only, or no observations at all). The winner on such a
        posterior is a tiebreak on float ordering, not a measurement, so callers
        must not publish it as a location.
        """
        return self.area != UNKNOWN_AREA


class BayesianEngine:
    """Per-person Bayesian area inference.

    On each update:
        P(area | observations, time) ∝ π(area, time) · ∏ P(obs_i | area)

    The prior is time-of-day dependent and learned from corrections.
    """

    def __init__(
        self,
        areas: list[str],
        prior_store: PriorStore,
        max_observation_age: float = 600.0,  # 10 min
    ):
        self.areas = areas
        self.prior_store = prior_store
        self.max_observation_age = max_observation_age
        self._last_estimate: Optional[AreaEstimate] = None

    def infer(self, observations: list[Observation], timestamp: Optional[float] = None) -> AreaEstimate:
        """Compute posterior over areas given current observations.

        Args:
            observations: List of current sensor observations
            timestamp: Optional timestamp (defaults to now)

        Returns:
            AreaEstimate with best guess, confidence, entropy, alternatives
        """
        if timestamp is None:
            timestamp = time.time()

        # Filter out stale observations — but keep active occupancy signals.
        # A motion sensor that's ON means someone is still there, regardless
        # of how long ago it triggered. Only drop stale OFF/unknown readings.
        fresh_obs = [
            o for o in observations
            if o.age_seconds <= self.max_observation_age
            or (o.kind == "motion" and o.state == "on")
        ]

        # Get time-of-day prior
        prior = self.prior_store.get_prior(hour_bucket(timestamp))

        # Compute posterior: P(area | obs) ∝ prior(area) · ∏ P(obs_i | area)
        log_posterior: dict[str, float] = {}

        for area in self.areas:
            log_p = math.log(max(prior.get(area, 1e-10), 1e-10))

            for obs in fresh_obs:
                likelihood = compute_likelihood(obs, area)
                log_p += math.log(max(likelihood, 1e-10))

            log_posterior[area] = log_p

        # Normalize via log-sum-exp
        max_log = max(log_posterior.values())
        exp_vals = {r: math.exp(lp - max_log) for r, lp in log_posterior.items()}
        total = sum(exp_vals.values())
        posterior = {r: ev / total for r, ev in exp_vals.items()}

        # Sort areas by posterior descending
        sorted_areas = sorted(posterior.items(), key=lambda x: x[1], reverse=True)

        # Shannon entropy in nats
        entropy = -sum(
            p * math.log(p) for p in posterior.values() if p > 1e-10
        )

        # Observations used (for transparency/debugging)
        obs_used = [
            {
                "source": o.entity_id,
                "kind": o.kind,
                "observed": o.area or o.state,
                "age_seconds": round(o.age_seconds, 1),
            }
            for o in fresh_obs
        ]

        # Two independent reasons to refuse to name a room:
        #   1. the posterior has no strict winner (tie at the max), or
        #   2. a GPS reading places the person outside the house and no local
        #      room-level signal contradicts it.
        # Either way `area` becomes UNKNOWN_AREA and device_tracker falls through
        # to its GPS branch rather than publishing an invented room.
        #
        # The room-level check matters because zones lag: someone who just got
        # home still reads as "work" on a polled tracker (iCloud3), while the BLE
        # tag and the motion sensor already see them in the kitchen. Local
        # evidence wins; GPS only decides when the house itself is silent.
        uninformative = (
            is_uninformative(posterior)
            or (has_away_observation(fresh_obs) and not has_room_level_evidence(fresh_obs, self.areas))
        )

        best_area = UNKNOWN_AREA if uninformative else sorted_areas[0][0]
        best_prob = sorted_areas[0][1]

        # Alternatives: the top 3 excluding the winner when we have one, or the
        # top 3 outright when the posterior is uniform (there is no winner).
        alt_slice = sorted_areas[:3] if uninformative else sorted_areas[1:4]
        alternatives = [
            {"area": r, "probability": round(p, 4)}
            for r, p in alt_slice
        ]

        estimate = AreaEstimate(
            area=best_area,
            confidence=round(best_prob, 4),
            entropy=round(entropy, 4),
            alternatives=alternatives,
            observations_used=obs_used,
            posterior=posterior,
            timestamp=timestamp,
        )

        self._last_estimate = estimate
        return estimate

    @property
    def last_estimate(self) -> Optional[AreaEstimate]:
        """The most recent inference result."""
        return self._last_estimate
