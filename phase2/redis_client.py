import json
import time
import re
from typing import List, Dict

DIGIT_RUN_PATTERN = re.compile(r'\d+')


class RedisWindowClient:
    """All state Phase 2 needs, keyed consistently by room_id + sender_id."""

    WINDOW_TTL = 120           # digit window sliding TTL, seconds
    SUSPICION_TTL = 600        # suspicion counter sliding TTL, seconds
    SUSPICION_ESCALATION = 3   # N moderate flags in the window -> force next to HIGH
    WINDOW_DIGIT_CAP = 15
    CONV_TTL = 180             # conversation buffer TTL -- longer than digit window,
                                # handle-completion patterns tend to span more real time
    CONV_MAX_TURNS = 5

    def __init__(self, redis_conn):
        self.r = redis_conn

    # ---- key helpers ----
    def _digit_key(self, room_id, sender_id):
        return f"digitwin:{room_id}:{sender_id}"

    def _suspicion_key(self, sender_id):
        return f"suspicion:{sender_id}"

    def _conv_key(self, room_id, sender_id):
        return f"convwin:{room_id}:{sender_id}"

    # ---- digit window ----
    def get_recent_digits(self, room_id, sender_id) -> str:
        return self.r.get(self._digit_key(room_id, sender_id)) or ""

    def append_digits(self, room_id, sender_id, digits: str):
        if not digits:
            return
        key = self._digit_key(room_id, sender_id)
        combined = (self.get_recent_digits(room_id, sender_id) + digits)[-self.WINDOW_DIGIT_CAP:]
        self.r.set(key, combined, ex=self.WINDOW_TTL)

    def clear_digit_window(self, room_id, sender_id):
        """Call after an escalation is handled so stale digits don't
        contribute to scoring a conversation that's already been dealt with."""
        self.r.delete(self._digit_key(room_id, sender_id))

    # ---- suspicion counter ----
    def increment_suspicion(self, sender_id) -> int:
        key = self._suspicion_key(sender_id)
        count = self.r.incr(key)
        if count == 1:
            self.r.expire(key, self.SUSPICION_TTL)
        return count

    def reset_suspicion(self, sender_id):
        self.r.delete(self._suspicion_key(sender_id))

    # ---- conversation window (for Phase 3 context) ----
    def get_recent_turns(self, room_id, sender_id) -> List[Dict]:
        raw = self.r.get(self._conv_key(room_id, sender_id))
        return json.loads(raw) if raw else []

    def append_turn(self, room_id, sender_id, speaker: str, text: str, msg_id: str):
        turns = self.get_recent_turns(room_id, sender_id)
        turns.append({"speaker": speaker, "text": text, "msg_id": msg_id, "ts": time.time()})
        turns = turns[-self.CONV_MAX_TURNS:]
        self.r.set(self._conv_key(room_id, sender_id), json.dumps(turns), ex=self.CONV_TTL)

    def clear_conversation(self, room_id, sender_id):
        self.r.delete(self._conv_key(room_id, sender_id))

    # ---- called once per message, before scoring ----
    def record_message(self, room_id, sender_id, speaker: str, text: str, msg_id: str):
        digits = "".join(DIGIT_RUN_PATTERN.findall(text))
        self.append_digits(room_id, sender_id, digits)
        self.append_turn(room_id, sender_id, speaker, text, msg_id)

    # ---- called after Phase 3 handles an escalation ----
    def clear_after_escalation(self, room_id, sender_id):
        self.clear_digit_window(room_id, sender_id)
        self.clear_conversation(room_id, sender_id)
        # suspicion counter intentionally NOT reset here -- a sender who
        # already triggered one escalation should stay flagged for the
        # remainder of the suspicion window, not get a clean slate
