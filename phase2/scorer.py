from enum import Enum
from feature_extractor import FeatureVector


class RiskTier(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class RiskScorer:
    LOW_THRESHOLD = 35
    HIGH_THRESHOLD = 65

    # A completed or near-completed cross-turn digit run is close to a
    # structural signal, not a soft one -- diluting it into the weighted
    # average (as MODERATE would if the other components are low) under-
    # reacts to the clearest evidence this scorer ever sees. Force at
    # least MODERATE regardless of the weighted sum once it's this high.
    # A completed or near-completed cross-turn digit run is close to a
    # structural signal, not a soft one -- diluting it into the weighted
    # average (as MODERATE would if the other components are low) under-
    # reacts to the clearest evidence this scorer ever sees. Force at
    # least MODERATE regardless of the weighted sum once it's this high.
    HISTORY_HARD_FLOOR = 0.7

    # A confirmed platform-keyword + bare-handle co-occurrence within the
    # window is close to the digit-history case in confidence -- it's the
    # whole reason this feature exists. Without a floor, its 0.10 weight
    # in the general formula is too small to be decisive on its own when
    # everything else about the message is otherwise quiet (a short
    # message with no other signal, exactly like a bare handle reply).
    PLATFORM_HARD_FLOOR = 1.0

    WEIGHTS = {
        "run_score": 0.25,
        "keyword_score": 0.20,
        "pattern_score": 0.15,
        "history_score": 0.15,
        "link_suspicion_score": 0.15,
        "cross_turn_platform_score": 0.10,
    }

    def score(self, f: FeatureVector) -> int:
        run_score = min(f.max_digit_run / 10, 1.0)
        raw = (
            self.WEIGHTS["run_score"] * run_score
            + self.WEIGHTS["keyword_score"] * f.contact_kw_score
            + self.WEIGHTS["pattern_score"] * f.pattern_score
            + self.WEIGHTS["history_score"] * f.history_score
            + self.WEIGHTS["link_suspicion_score"] * f.link_suspicion_score
            + self.WEIGHTS["cross_turn_platform_score"] * f.cross_turn_platform_score
        )
        score = round(raw * 100)

        if f.history_score >= self.HISTORY_HARD_FLOOR:
            score = max(score, self.LOW_THRESHOLD)  # guarantee at least MODERATE

        if f.cross_turn_platform_score >= self.PLATFORM_HARD_FLOOR:
            score = max(score, self.LOW_THRESHOLD)  # guarantee at least MODERATE

        return min(score, 100)

    def classify(self, score: int) -> RiskTier:
        if score < self.LOW_THRESHOLD:
            return RiskTier.LOW
        if score < self.HIGH_THRESHOLD:
            return RiskTier.MODERATE
        return RiskTier.HIGH