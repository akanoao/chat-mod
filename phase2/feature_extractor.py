import re
from dataclasses import dataclass
from typing import List, Dict

DIGIT_RUN_PATTERN = re.compile(r'\d+')
PHONE_PATTERN = re.compile(r'\d{3}[-\s]?\d{3}[-\s]?\d{4}')

CONTACT_KEYWORDS = ["call", "whatsapp", "contact", "reach me", "text me", "number is", "hmu", "dm me"]
QUANTITY_KEYWORDS = ["units", "qty", "pcs", "model", "sku", "$", "price"]

DOMAIN_TOKEN = re.compile(r'\b[a-zA-Z0-9-]+\.(com|org|net|io|co|me|ly|to|gg)\b', re.IGNORECASE)
TRAILING_PATH_LIKE = re.compile(r'/[^\s]+')

PLATFORM_KEYWORDS = ["insta", "instagram", "telegram", "snap", "snapchat", "discord",
                      "line", "kik", "wa", "whatsapp", "fb", "facebook"]
# A bare handle: alphanumeric/underscore/dot, no spaces, plausible length.
# Deliberately loose -- this is a soft signal, not a structural match, so it
# leans on co-occurrence with a platform keyword to mean anything.
BARE_HANDLE_SHAPE = re.compile(r'^[\w.]{3,20}$')


@dataclass
class FeatureVector:
    max_digit_run: int
    digit_density: float
    contact_kw_score: float
    pattern_score: float
    history_score: float
    link_suspicion_score: float
    cross_turn_platform_score: float


class FeatureExtractor:
    def __init__(self, redis_client):
        self.redis = redis_client

    def extract(self, text: str, room_id: str, sender_id: str) -> FeatureVector:
        digit_runs = DIGIT_RUN_PATTERN.findall(text)
        max_run = max((len(r) for r in digit_runs), default=0)
        total_digits = sum(len(r) for r in digit_runs)
        digit_density = total_digits / max(len(text), 1)

        lowered = text.lower()
        contact_hits = sum(kw in lowered for kw in CONTACT_KEYWORDS)
        quantity_hits = sum(kw in lowered for kw in QUANTITY_KEYWORDS)
        contact_kw_score = max(0.0, min(1.0, 0.5 + contact_hits * 0.15 - quantity_hits * 0.15))

        pattern_score = 1.0 if PHONE_PATTERN.search(text) else 0.0

        prior_digits = self.redis.get_recent_digits(room_id, sender_id)
        combined_len = len(prior_digits) + total_digits
        history_score = min(combined_len / 10, 1.0)

        link_suspicion_score = self._link_fragment_suspicion(text)
        cross_turn_platform_score = self._cross_turn_platform_score(text, room_id, sender_id)

        return FeatureVector(
            max_digit_run=max_run,
            digit_density=digit_density,
            contact_kw_score=contact_kw_score,
            pattern_score=pattern_score,
            history_score=history_score,
            link_suspicion_score=link_suspicion_score,
            cross_turn_platform_score=cross_turn_platform_score,
        )

    def _link_fragment_suspicion(self, text: str) -> float:
        """Catches cases Phase 1's regex couldn't cleanly bound (decoy tokens,
        broken paths) -- a domain-shaped token with SOMETHING path-like
        trailing it, even if the exact span isn't extractable."""
        domain_hits = list(DOMAIN_TOKEN.finditer(text))
        if not domain_hits:
            return 0.0
        for m in domain_hits:
            window = text[m.end():m.end() + 40]
            if TRAILING_PATH_LIKE.search(window):
                return 1.0
        return 0.3  # bare domain match, no trailing path signal

    def _cross_turn_platform_score(self, current_text: str, room_id: str, sender_id: str) -> float:
        """Symmetric: a platform keyword and a bare-handle-shaped token can
        appear in either order across the recent turns, not just
        keyword-then-handle. Checks every ordered pair within the window
        (current message included) rather than assuming a fixed order."""
        recent_turns = self.redis.get_recent_turns(room_id, sender_id)
        texts = [t["text"] for t in recent_turns] + [current_text]
        if len(texts) < 2:
            return 0.0

        def has_platform_kw(s: str) -> bool:
            low = s.lower()
            return any(kw in low for kw in PLATFORM_KEYWORDS)

        def has_bare_handle(s: str) -> bool:
            token = s.strip()
            return bool(BARE_HANDLE_SHAPE.match(token)) and not has_platform_kw(token)

        kw_present = any(has_platform_kw(t) for t in texts)
        handle_present = any(has_bare_handle(t) for t in texts)

        if kw_present and handle_present:
            return 1.0
        if kw_present or handle_present:
            return 0.2  # only one half of the pattern seen so far -- weak signal
        return 0.0
