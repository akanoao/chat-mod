import json
import requests
from typing import List, Dict, Optional

# Must match section 3 of the fine-tuning guide EXACTLY -- any drift here
# is effectively querying the model with a prompt shape it wasn't trained
# on, which silently degrades quality without throwing an error anywhere.
SYSTEM_PROMPT = (
    "You detect contact-sharing attempts in vacation-rental chat messages "
    "(phone numbers, emails, social or payment handles), including ones "
    "split across multiple messages. Given the conversation so far, "
    "respond with JSON: {\"contains_violation\": bool, \"options\": "
    "[...]}. If contains_violation is true, options must contain exactly "
    "3 clean rewrite sentences preserving any legitimate intent, with no "
    "contact information. If false, options must be an empty list."
)

MIN_OPTION_WORDS = 4  # same bar used in the QC pipeline -- catches the
                       # category-tag-instead-of-sentence failure mode


def render_turns(turns: List[Dict]) -> str:
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)


class QwenClient:
    def __init__(self, endpoint: str = "http://localhost:8080/v1/chat/completions", timeout: float = 3.0):
        self.endpoint = endpoint
        self.timeout = timeout

    def generate_options(self, turns: List[Dict]) -> Dict:
        """Returns {"contains_violation": bool, "options": [...], "valid": bool}.
        valid=False means the model output failed validation and the
        caller should fall back to a safe default, not trust the content."""
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_turns(turns)},
        ]

        try:
            resp = requests.post(
                self.endpoint,
                json={
                    "messages": prompt_messages,
                    "temperature": 0.3,  # low temperature -- this is a
                                          # structured-output task, not creative writing
                    "max_tokens": 200,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
        except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
            return self._fallback(reason=f"request_or_parse_error: {e}")

        return self._validate(parsed)

    def _validate(self, parsed: Dict) -> Dict:
        if "contains_violation" not in parsed:
            return self._fallback(reason="missing contains_violation")

        contains_violation = bool(parsed["contains_violation"])
        options = parsed.get("options", [])

        if not contains_violation:
            return {"contains_violation": False, "options": [], "valid": True}

        if len(options) != 3:
            return self._fallback(reason=f"expected 3 options, got {len(options)}")

        for opt in options:
            if not isinstance(opt, str) or len(opt.split()) < MIN_OPTION_WORDS:
                return self._fallback(reason="option looks like a tag, not a sentence")

        return {"contains_violation": True, "options": options, "valid": True}

    def _fallback(self, reason: str) -> Dict:
        """Model output couldn't be trusted -- fail safe, not open. Treat
        as a confirmed violation with no usable rewrite, so the message
        still gets blocked rather than silently passed through, and flag
        it for the retraining log rather than pretending nothing happened."""
        return {
            "contains_violation": True,
            "options": [],
            "valid": False,
            "fallback_reason": reason,
        }
