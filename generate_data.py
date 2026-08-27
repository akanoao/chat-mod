"""
Synthetic data generator — alternates between Gemini 3.1 Flash-Lite and
Gemini 3.5 Flash-Lite to combine two separate 500 req/day quotas into one
run. Respects each model's 15 req/min limit independently, uses structured
output (Pydantic schema) so every response is guaranteed schema-valid,
writes incrementally to JSONL, and resumes correctly if you re-run it
(counts what's already in the output file before generating more).

Requires: pip install google-genai pydantic
Set GEMINI_API_KEY in your environment before running.

Coercion category removed per your last message — only contact_evasion
and clean_control remain. Category targets below total ~2,400 base
examples; the language-mix pass at the end adds ~25% more on top,
landing around 3,000 total. Bump the "target" values if you want to get
back toward the original 3,800.
"""

import os
import json
import time
import random
import collections
from typing import List
from pydantic import BaseModel
from google import genai
from google.genai import types

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]  # verify exact
                                                               # model ids in
                                                               # AI Studio before running
OUTPUT_FILE = "synthetic_raw.jsonl"
RPM_LIMIT = 15          # requests per minute, per model
RPD_LIMIT = 500         # requests per day, per model
MIN_INTERVAL = 60.0 / RPM_LIMIT  # seconds between requests to the SAME model

SHARED_SYSTEM_PROMPT = """You are generating synthetic training data for a safety
classifier used in a vacation-rental chat app. The classifier detects attempts to
share contact information (phone, email, social/payment handles) to move a booking
off-platform.

Generate realistic guest/host messages set in an Airbnb-style vacation rental
context (bookings, checkin/checkout, amenities, payments, local recommendations).
Vary names, locations, phrasing length, and sentence structure across every
example in the batch — do not reuse the same sentence template twice."""

# ---------------------------------------------------------------------
# SCHEMA — model only outputs content; script fills id/intent/metadata after
# ---------------------------------------------------------------------

class Turn(BaseModel):
    speaker: str
    text: str

class Delivery(BaseModel):
    multiturn: bool
    language_mix: bool

class Label(BaseModel):
    contains_violation: bool
    options: List[str]

class SyntheticExample(BaseModel):
    subtype: str
    delivery: Delivery
    turns: List[Turn]
    label: Label

class BatchResponse(BaseModel):
    examples: List[SyntheticExample]

# ---------------------------------------------------------------------
# CATEGORY PROMPTS (base generation — before the language-mix pass)
# ---------------------------------------------------------------------

CATEGORIES = {
    "digit_obfuscation": {
        "intent": "contact_evasion", "target": 450, "batch": 20,
        "prompt": """Generate {N} new examples, subtype "digit_obfuscation".
Each message contains a 10-digit phone number disguised using ONE of these
techniques, varied across the batch: spelled-out number words,
digit-letter substitution (leetspeak), spacing between digits, symbol
separators (dashes/dots), or mixed into ordinary sentence filler. Give a
plausible non-contact reason for the message (coordinating checkin,
asking to move to WhatsApp, sharing a "direct line").
Set delivery.multiturn=false, delivery.language_mix=false.
label.contains_violation=true. Write 3 clean rewrite options per example
that preserve any legitimate scheduling/logistics intent while removing
all contact info."""
    },
    "digit_script_mixed": {
        "intent": "contact_evasion", "target": 150, "batch": 15,
        "prompt": """Generate {N} new examples, subtype "digit_script_mixed".
Each contains a 10-digit phone number written in non-Latin numerals
(Bengali, Arabic-Indic) or as emoji digit characters. Vary which script
per example. delivery.multiturn=false, delivery.language_mix=false.
label.contains_violation=true, with 3 rewrite options each."""
    },
    "email_obfuscation": {
        "intent": "contact_evasion", "target": 300, "batch": 20,
        "prompt": """Generate {N} new examples, subtype "email_obfuscation".
Disguise an email using verbal separators ("at", "dot", bracketed
"[at]"/"[dot]"), letter-spacing, or a mix. Vary domain (gmail, yahoo,
hotmail, rediffmail, outlook) and the reason given (invoice, ID scan,
deposit, documents). delivery.multiturn=false, delivery.language_mix=false.
label.contains_violation=true, 3 rewrite options preserving the stated
reason without any contact info."""
    },
    "social_handle": {
        "intent": "contact_evasion", "target": 300, "batch": 20,
        "prompt": """Generate {N} new examples, subtype "social_handle".
Cover Telegram, Instagram, Discord, WhatsApp links (wa.me, t.me). Some
should wrap the handle in a real URL (t.me/username), some should be a
bare handle next to a platform name with no URL structure.
delivery.multiturn=false, delivery.language_mix=false.
label.contains_violation=true, 3 rewrite options."""
    },
    "payment_handle": {
        "intent": "contact_evasion", "target": 150, "batch": 15,
        "prompt": """Generate {N} new examples, subtype "payment_handle".
Cover UPI/VPA style payment handles (paytm, okhdfcbank, okicici) shared to
avoid platform payment fees. delivery.multiturn=false,
delivery.language_mix=false. label.contains_violation=true, 3 rewrite
options that redirect payment through the platform instead."""
    },
    "link_decoy": {
        "intent": "contact_evasion", "target": 150, "batch": 15,
        "prompt": """Generate {N} new examples, subtype "link_decoy". Each
message contains a real domain or shortened link (t.me/..., bit.ly/...,
or domain+path) immediately followed by extra characters designed to
break naive regex matching: a bracketed word like "(ignore)", an
underscore-joined multi-segment path, or a second space-separated
fragment continuing the path. Vary the decoy style across the batch.
delivery.multiturn=false, delivery.language_mix=false.
label.contains_violation=true, 3 rewrite options."""
    },
    "clean_digit_heavy": {
        "intent": "clean_control", "target": 450, "batch": 25,
        "prompt": """Generate {N} new examples, subtype "clean_digit_heavy".
Legitimate vacation-rental messages containing digits that are NOT
contact info: booking references, prices, dates, guest counts, distances,
times. delivery.multiturn=false, delivery.language_mix=false.
label.contains_violation=false, label.options=[] for every example."""
    },
    "clean_keyword_adjacent": {
        "intent": "clean_control", "target": 250, "batch": 20,
        "prompt": """Generate {N} new examples, subtype
"clean_keyword_adjacent". Messages that mention a platform keyword
("call", "insta", "link", "whatsapp") WITHOUT sharing any actual contact
info — e.g. "check the front desk if you need to call," "I'll post
photos on my Instagram sometime." delivery.multiturn=false,
delivery.language_mix=false. label.contains_violation=false,
label.options=[]. These are the most important examples to get right —
make them genuinely adjacent to a false positive."""
    },
    "clean_ordinary": {
        "intent": "clean_control", "target": 200, "batch": 20,
        "prompt": """Generate {N} new examples, subtype "clean_ordinary".
Ordinary vacation-rental property Q&A with zero digits and zero contact
keywords — amenities, house rules, recommendations, complaints unrelated
to coercion. delivery.multiturn=false, delivery.language_mix=false.
label.contains_violation=false, label.options=[]."""
    },
}

LANGUAGE_MIX_PROMPT = """Here are {N} existing training examples in English,
including their original label (contains_violation and, where present,
their 3 rewrite options):
{examples_json}

Rewrite each one as a natural Hindi-English code-switched (Hinglish)
message, the way a real guest or host would type casually — not a formal
translation. Keep the same subtype and contains_violation status.

If label.options in the input contains 3 rewrite sentences, translate
those exact 3 sentences into the same Hinglish register — output
label.options as 3 full rewritten sentences, in the same order, NOT a
category name or a single word. If label.options in the input is empty
(clean_control examples), output label.options as an empty list — do not
invent a placeholder like "none".

Set delivery.language_mix=true, keep delivery.multiturn the same as the
original. Return the same schema, one output example per input example,
in the same order."""

# ---------------------------------------------------------------------
# RATE LIMITER — tracks per-minute and per-day usage independently per model
# ---------------------------------------------------------------------

class ModelLimiter:
    def __init__(self, name):
        self.name = name
        self.last_call = 0.0
        self.daily_count = 0
        self.exhausted = False

    def wait_if_needed(self):
        elapsed = time.time() - self.last_call
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)

    def record_call(self):
        self.last_call = time.time()
        self.daily_count += 1
        if self.daily_count >= RPD_LIMIT:
            self.exhausted = True


class ModelRotator:
    """Alternates between models, skipping any that hit their daily cap."""
    def __init__(self, model_names):
        self.limiters = {m: ModelLimiter(m) for m in model_names}
        self._cycle = collections.deque(model_names)

    def next_model(self):
        for _ in range(len(self._cycle)):
            model = self._cycle[0]
            self._cycle.rotate(-1)
            if not self.limiters[model].exhausted:
                return model
        raise RuntimeError("All models have hit their daily request cap.")

    def limiter_for(self, model):
        return self.limiters[model]


# ---------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
rotator = ModelRotator(MODELS)
seen_texts = set()  # cheap exact-dedup pass; run embedding-based dedup separately after


def normalize_for_dedup(text: str) -> str:
    return " ".join(text.lower().split())


def call_model(model_name: str, prompt: str, schema) -> BatchResponse:
    limiter = rotator.limiter_for(model_name)
    limiter.wait_if_needed()
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SHARED_SYSTEM_PROMPT,
                temperature=1.0,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        limiter.record_call()
        return BatchResponse.model_validate_json(resp.text)
    except Exception as e:
        limiter.record_call()  # still counts against quota even on error, usually
        print(f"  [warn] {model_name} call failed: {e}")
        return BatchResponse(examples=[])


def existing_count_by_subtype(path: str) -> dict:
    counts = collections.Counter()
    if not os.path.exists(path):
        return counts
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                counts[obj["subtype"]] += 1
                for t in obj["turns"]:
                    seen_texts.add(normalize_for_dedup(t["text"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return counts


def generate_category(name: str, cfg: dict, out_f, existing: dict):
    have = existing.get(name, 0)
    target = cfg["target"]
    if have >= target:
        print(f"[{name}] already at target ({have}/{target}), skipping")
        return
    print(f"[{name}] generating {target - have} more (have {have}/{target})")

    while have < target:
        batch_size = min(cfg["batch"], target - have)
        model = rotator.next_model()
        prompt = cfg["prompt"].format(N=batch_size)
        result = call_model(model, prompt, BatchResponse)

        added = 0
        for ex in result.examples:
            key = normalize_for_dedup(ex.turns[0].text) if ex.turns else None
            if key and key in seen_texts:
                continue  # drop exact/near-exact duplicate
            if key:
                seen_texts.add(key)

            record = {
                "id": f"{name}_{have + added:04d}",
                "intent": cfg["intent"],
                "subtype": ex.subtype or name,
                "delivery": ex.delivery.model_dump(),
                "turns": [t.model_dump() for t in ex.turns],
                "label": ex.label.model_dump(),
                "metadata": {"source": model, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            }
            out_f.write(json.dumps(record) + "\n")
            added += 1

        out_f.flush()
        have += added
        print(f"  [{name}] +{added} (model={model}, total={have}/{target})")

        if added == 0:
            print(f"  [{name}] batch produced 0 usable examples — check prompt or schema")
            break


def run_language_mix_pass(source_path: str, out_f, fraction: float = 0.25, batch_size: int = 10):
    with open(source_path, encoding="utf-8") as f:
        all_examples = [json.loads(line) for line in f]

    sample_size = int(len(all_examples) * fraction)
    sample = random.sample(all_examples, min(sample_size, len(all_examples)))
    print(f"[language_mix] rewriting {len(sample)} examples into Hinglish")

    for i in range(0, len(sample), batch_size):
        chunk = sample[i:i + batch_size]
        model = rotator.next_model()
        prompt = LANGUAGE_MIX_PROMPT.format(
            N=len(chunk),
            examples_json=json.dumps([
                {"turns": c["turns"], "subtype": c["subtype"], "label": c["label"]}
                for c in chunk
            ]),
        )
        result = call_model(model, prompt, BatchResponse)

        for orig, ex in zip(chunk, result.examples):
            record = {
                "id": orig["id"] + "_hi",
                "intent": orig["intent"],
                "subtype": orig["subtype"],
                "delivery": {**ex.delivery.model_dump(), "language_mix": True},
                "turns": [t.model_dump() for t in ex.turns],
                "label": ex.label.model_dump(),
                "metadata": {"source": model, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "derived_from": orig["id"]},
            }
            out_f.write(json.dumps(record) + "\n")
        out_f.flush()
        print(f"  [language_mix] batch {i // batch_size + 1} done (model={model})")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    existing = existing_count_by_subtype(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a") as out_f:
        # for name, cfg in CATEGORIES.items():
        #     generate_category(name, cfg, out_f, existing)

        run_language_mix_pass(OUTPUT_FILE, out_f)

    print(f"\nDone. Output in {OUTPUT_FILE}")
    print("Next: run embedding-based near-duplicate detection (section 6 of the guide) "
          "before this data touches training — exact-match dedup above only catches literal repeats.")