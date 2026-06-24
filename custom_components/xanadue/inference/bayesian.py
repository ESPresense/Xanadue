"""Bayesian room inference engine.

Maintains a categorical posterior over rooms for a single person,
updated incrementally as new observations arrive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
import time

from .likelihoods import Observation, compute_likelihood
from .priors import PriorStore, hour_bucket


@dataclass
class AreaEstimate:
    """The output of a Bayesian room inference step."""

    area: str                          # best guess
    confidence: float                  # posterior probability of best guess
    entropy: float                     # Shannon entropy in nats
    alternatives: list[dict]           # [{room, probability}, ...] sorted desc
    observations_used: list[dict]      # [{source, observed, weight}, ...]
    posterior: dict[str, float]        # full posterior distribution
    timestamp: float                   # when this estimate was computed


class BayesianEngine:
    """Per-person Bayesian room inference.

    On each update:
        P(room | observations, time) ∝ π(room, time) · ∏ P(obs_i | room)

    The prior is time-of-day dependent and learned from corrections.
    """

    def __init__(
        self,
        rooms: list[str],
        prior_store: PriorStore,
        max_observation_age: float = 600.0,  # 10 min
    ):
        self.areas = rooms
        self.prior_store = prior_store
        self.max_observation_age = max_observation_age
        self._last_estimate: Optional[AreaEstimate] = None

    def infer(self, observations: list[Observation], timestamp: Optional[float] = None) -> AreaEstimate:
        """Compute posterior over rooms given current observations.

        Args:
            observations: List of current sensor observations
            timestamp: Optional timestamp (defaults to now)

        Returns:
            AreaEstimate with best guess, confidence, entropy, alternatives
        """
        if timestamp is None:
            timestamp = time.time()

        # Filter out stale observations
        fresh_obs = [
            o for o in observations
            if o.age_seconds <= self.max_observation_age
        ]

        # Get time-of-day prior
        prior = self.prior_store.get_prior(hour_bucket(timestamp))

        # Compute posterior: P(room | obs) ∝ prior(room) · ∏ P(obs_i | room)
        log_posterior: dict[str, float] = {}

        for room in self.areas:
            log_p = math.log(max(prior.get(room, 1e-10), 1e-10))

            for obs in fresh_obs:
                likelihood = compute_likelihood(obs, room)
                log_p += math.log(max(likelihood, 1e-10))

            log_posterior[room] = log_p

        # Normalize via log-sum-exp
        max_log = max(log_posterior.values())
        exp_vals = {r: math.exp(lp - max_log) for r, lp in log_posterior.items()}
        total = sum(exp_vals.values())
        posterior = {r: ev / total for r, ev in exp_vals.items()}

        # Sort areas by posterior descending
        sorted_areas = sorted(posterior.items(), key=lambda x: x[1], reverse=True)

        best_area = sorted_areas[0][0]
        best_prob = sorted_areas[0][1]

        # Shannon entropy in nats
        entropy = -sum(
            p * math.log(p) for p in posterior.values() if p > 1e-10
        )

        # Alternatives (top 3 after best)
        alternatives = [
            {"area": r, "probability": round(p, 4)}
            for r, p in sorted_areas[1:4]
        ]

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
