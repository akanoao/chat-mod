import re
import time
from feature_extractor import FeatureExtractor, DIGIT_RUN_PATTERN, PLATFORM_KEYWORDS, BARE_HANDLE_SHAPE
from scorer import RiskScorer, RiskTier
from llm_client import QwenClient


class InterventionHandler:
    def __init__(self, redis_client, scorer: RiskScorer, llm_client: QwenClient, logger):
        self.redis = redis_client
        self.scorer = scorer
        self.llm = llm_client
        self.logger = logger

    def handle(self, text: str, room_id: str, sender_id: str, speaker: str, msg_id: str,
               features, tier: RiskTier, score: int) -> dict:

        if tier == RiskTier.LOW:
            self.logger.log(msg_id, tier, score, features, action="pass")
            return {"action": "pass", "text": text}

        if tier == RiskTier.MODERATE:
            return self._handle_moderate(text, room_id, sender_id, msg_id, features, score)

        return self._handle_high(text, room_id, sender_id, speaker, msg_id, features, score)

    # ------------------------------------------------------------------
    def _handle_moderate(self, text, room_id, sender_id, msg_id, features, score) -> dict:
        count = self.redis.increment_suspicion(sender_id)
        if count >= self.redis.SUSPICION_ESCALATION:
            self.logger.log(msg_id, RiskTier.HIGH, score, features,
                             action="llm_escalation", reason="escalated_by_suspicion")
            return self._escalate(text, room_id, sender_id, msg_id, features, score,
                                   escalated_from_suspicion=True)

        masked = self._targeted_mask(text, features)
        self.logger.log(msg_id, RiskTier.MODERATE, score, features, action="soft_mask")
        return {"action": "soft_mask", "text": masked}

    def _targeted_mask(self, text: str, features) -> str:
        """Mask only the specific ambiguous token(s) that triggered
        suspicion -- never the surrounding keywords. A digit run gets
        masked because the digits are the sensitive payload; a bare
        handle-shaped token gets masked for the same reason; the platform
        keyword itself ("insta", "whatsapp") is left untouched since it
        isn't sensitive on its own and stripping it just makes the
        message read as mangled nonsense without protecting anything."""
        masked = text

        if features.cross_turn_platform_score >= 0.2:
            def mask_if_bare_handle(match):
                token = match.group()
                if BARE_HANDLE_SHAPE.match(token) and not any(
                    kw in token.lower() for kw in PLATFORM_KEYWORDS
                ):
                    return "█" * len(token)
                return token
            masked = re.sub(r'\S+', mask_if_bare_handle, masked)

        if features.max_digit_run > 0 or features.history_score > 0:
            masked = DIGIT_RUN_PATTERN.sub(lambda m: "█" * len(m.group()), masked)

        return masked

    # ------------------------------------------------------------------
    def _handle_high(self, text, room_id, sender_id, speaker, msg_id, features, score) -> dict:
        self.logger.log(msg_id, RiskTier.HIGH, score, features, action="llm_escalation")
        return self._escalate(text, room_id, sender_id, msg_id, features, score,
                               escalated_from_suspicion=False)

    def _escalate(self, text, room_id, sender_id, msg_id, features, score,
                  escalated_from_suspicion: bool) -> dict:
        prior_turns = self.redis.get_recent_turns(room_id, sender_id)
        turns_for_model = prior_turns + [{"speaker": "user", "text": text}]

        result = self.llm.generate_options(turns_for_model)

        if not result["valid"]:
            self.logger.log(msg_id, RiskTier.HIGH, score, features,
                             action="llm_fallback_block", reason=result.get("fallback_reason"))
            self.redis.clear_after_escalation(room_id, sender_id)
            return {"action": "block", "text": None,
                    "reason": "model output failed validation, blocked rather than passed through"}

        if not result["contains_violation"]:
            # Model disagreed with the scorer -- log this explicitly, it's
            # exactly the kind of disagreement worth reviewing for the
            # weekly retrain rather than silently trusting either side.
            self.logger.log(msg_id, RiskTier.HIGH, score, features,
                             action="llm_override_pass", reason="model says no violation")
            return {"action": "pass", "text": text}

        # Phase 6 backstop applies here regardless of how good the model looked in eval
        clean_options = self._phase6_gate(result["options"])
        self.redis.clear_after_escalation(room_id, sender_id)
        return {"action": "llm_escalation", "options": clean_options,
                "escalated_from_suspicion": escalated_from_suspicion}

    def _phase6_gate(self, options: list) -> list:
        """Deterministic final check -- if a hallucinated option still
        contains a leak-shaped span, drop that specific chip rather than
        trusting the model's own judgment on its output."""
        clean = []
        for opt in options:
            digit_runs = DIGIT_RUN_PATTERN.findall(opt)
            if any(len(r) >= 7 for r in digit_runs):
                continue  # drop this chip, don't send it to the UI
            clean.append(opt)
        return clean


# ----------------------------------------------------------------------
# ORCHESTRATOR -- ties Phase 1 output into scoring, intervention, and
# updates Redis state for the NEXT message in this conversation.
# ----------------------------------------------------------------------

def run_phase2(phase1_result: dict, speaker: str, redis_client, extractor: FeatureExtractor,
               scorer: RiskScorer, handler: InterventionHandler) -> dict:
    room_id = phase1_result["room_id"]
    sender_id = phase1_result["sender_id"]
    msg_id = phase1_result["msg_id"]
    text = phase1_result["final_text"]

    # Score BEFORE recording this message, so history_score reflects
    # everything prior to (not including) this message -- then record
    # afterward so the NEXT message sees this one in its own history.
    features = extractor.extract(text, room_id, sender_id)
    score = scorer.score(features)
    tier = scorer.classify(score)

    result = handler.handle(text, room_id, sender_id, speaker, msg_id, features, tier, score)

    redis_client.record_message(room_id, sender_id, speaker, text, msg_id)

    return {**result, "tier": tier.value, "score": score, "msg_id": msg_id}