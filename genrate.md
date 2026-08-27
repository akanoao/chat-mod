# Synthetic data generation guide — local, RTX 4090

## 1. Taxonomy: intent vs. delivery

Restructuring your sample data into two axes instead of five flat categories:

**Intent (what the message is trying to do — the actual label):**

- `contact_evasion` — trying to move contact off-platform (phone, email, social handle, payment handle)
- `coercion` — extortion/blackmail via reviews or refunds
- `clean_control` — legitimate message, must NOT be flagged (includes digit-heavy legit text and keyword-adjacent-but-safe text)

**Delivery variant (how it's phrased — cross-cut every intent, don't isolate as its own bucket):**

- obfuscation subtype (spelled digits, symbol substitution, spacing, mixed-script numerals, emoji digits, verbal email separators)
- `multiturn_split` (any intent, split across 2-5 turns, optionally with filler turns interleaved)
- `language_mix` (Hindi-English code-switching, applicable to any intent)

Every generation prompt below asks for a specific intent, and you separately pass a "delivery instruction" that tells the model whether this batch should be single-turn/clean, split-turn, or code-switched. That's what gets you real coverage of the intersections instead of three isolated corners of the space.

---

## 2. Volume and split

Target: **~3,800 total examples** for the LoRA fine-tune. Contact evasion and its close variants dominate since that's the primary task; clean controls are intentionally large because false positives are the costliest failure mode for UX; coercion is included at a meaningful but smaller volume since (per the note above) you may end up training it as a separate lightweight head or a separate model entirely.

| Intent                       | Subtype                                                                                       | Count     | Delivery mix                                |
| ---------------------------- | --------------------------------------------------------------------------------------------- | --------- | ------------------------------------------- |
| contact_evasion              | digit (spelled/symbol/spaced/leetspeak)                                                       | 450       | 70% single-turn, 30% split                  |
| contact_evasion              | digit (script-mixed: Bengali/Arabic-Indic/emoji)                                              | 150       | single-turn only                            |
| contact_evasion              | email (verbal at/dot, spaced letters)                                                         | 300       | 70% single-turn, 30% split                  |
| contact_evasion              | social/messaging handle (telegram, insta, discord, wa.me links)                               | 300       | 60% single-turn, 40% split                  |
| contact_evasion              | payment handle (UPI/VPA, paytm)                                                               | 150       | single-turn only                            |
| contact_evasion              | link with decoy/trailing junk (the slash-decoy pattern)                                       | 150       | single-turn only                            |
| **contact_evasion subtotal** |                                                                                               | **1,500** |                                             |
| clean_control                | digit-heavy legit (booking refs, prices, dates, quantities)                                   | 450       | single-turn                                 |
| clean_control                | keyword-adjacent-but-safe ("insta" mentioned re: photos, "call" re: unrelated topic)          | 250       | single-turn                                 |
| clean_control                | ordinary property Q&A (no digits, no keywords — pure negatives)                               | 200       | single-turn                                 |
| **clean_control subtotal**   |                                                                                               | **900**   |                                             |
| coercion                     | refund threats                                                                                | 180       | single-turn                                 |
| coercion                     | review blackmail / quid pro quo                                                               | 180       | single-turn                                 |
| coercion                     | near-miss legitimate complaints (controls — should NOT be flagged)                            | 140       | single-turn                                 |
| **coercion subtotal**        |                                                                                               | **500**   |                                             |
| —                            | language_mix pass: regenerate ~25% of every subtype above in Hindi-English code-switched form | **~900**  | replaces/augments a slice of each row above |

Running total ≈ 1,500 + 900 + 500 + 900 ≈ **3,800**. Adjust the language-mix percentage up if your actual user base skews more code-switched than 25% — that number should come from a look at real chat logs if you have any, not a guess.

---

## 3. Local generation stack for an RTX 4090 (24GB)

You don't need frontier-model quality for this — you need consistent instruction-following and decent bilingual fluency for the Hinglish pass. Recommendation:

- **Primary model: `Qwen2.5-14B-Instruct-AWQ`** (4-bit, ~9GB weights). Comfortably fits with huge headroom for batch concurrency and long context, fast enough to generate thousands of examples in a reasonable session, and Qwen's instruction tuning handles code-switched Hindi-English well — relevant for your `language_mix` pass specifically.
- **Optional higher-quality pass: `Qwen2.5-32B-Instruct-AWQ`** (4-bit, ~19-20GB weights) for the categories that most reward nuance — `coercion` in particular, where subtlety between a genuine complaint and a veiled threat matters more than raw volume.
- **Serving**: run either through **vLLM** locally (`vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ --quantization awq`), which gives you an OpenAI-compatible endpoint plus **guided JSON decoding** (`response_format={"type": "json_object"}` or a JSON schema via `guided_json`) so every generation is schema-valid by construction rather than something you're regex-parsing out of free text afterward.

---

## 4. Output schema (branches by intent)

```json
{
	"id": "syn_000001",
	"intent": "contact_evasion",
	"subtype": "digit_spelled",
	"delivery": { "multiturn": false, "language_mix": false },
	"turns": [{ "speaker": "guest", "text": "..." }],
	"label": {
		"contains_violation": true,
		"options": ["rewrite 1", "rewrite 2", "rewrite 3"]
	},
	"metadata": { "source": "qwen2.5-14b-instruct-awq", "generated_at": "..." }
}
```

For `coercion`, `label` looks different — no rewrite options, since there's no legitimate intent to preserve:

```json
"label": {
  "contains_violation": true,
  "action": "flag_for_moderator",
  "violation_type": "refund_threat"
}
```

For `clean_control`:

```json
"label": {
  "contains_violation": false,
  "options": []
}
```

---

## 5. Prompts

### 5a. Shared system prompt (prepend to every category call)

```
You are generating synthetic training data for a safety classifier used in
a vacation-rental chat app. The classifier detects two things: (1) attempts
to share contact information (phone, email, social/payment handles) to move
a booking off-platform, and (2) coercive messages threatening a bad review
or demanding a refund under threat.

Generate realistic guest/host messages set in an Airbnb-style vacation
rental context (bookings, checkin/checkout, amenities, payments, local
recommendations). Vary names, locations, phrasing length, and sentence
structure across every example — do not reuse the same sentence template
twice in a batch. Output must be valid JSON matching the schema provided.
Do not include any explanation outside the JSON.
```

### 5b. contact_evasion — digit obfuscation

```
Generate {N} new examples of intent "contact_evasion", subtype
"digit_obfuscation". Each message should contain a 10-digit phone number
disguised using ONE of these techniques, varied across the batch:
spelled-out number words, digit-letter substitution (leetspeak), spacing
between digits, symbol separators (dashes/dots), or mixed with ordinary
sentence filler. The surrounding sentence should have a plausible
non-contact reason to be talking (coordinating checkin, asking to move to
WhatsApp, sharing a "direct line").

Seed examples for style reference (do not copy these, generate new ones):
{paste 3-4 examples from CE-01, CE-03, CE-14, CE-16, CE-23 here}

For each example, also write 3 clean rewrite options that preserve
whatever legitimate scheduling/logistics intent was in the message while
removing all contact information.

Return a JSON array of {N} objects matching this schema:
{schema from section 4}
```

### 5c. contact_evasion — email obfuscation

```
Generate {N} new examples of intent "contact_evasion", subtype
"email_obfuscation". Disguise an email address using verbal separators
("at", "dot", bracketed "[at]"/"[dot]"), letter-spacing, or a mix. Vary
domain (gmail, yahoo, hotmail, rediffmail, outlook) and the reason given
for sharing it (invoice, ID scan, deposit, documents).

Seed examples: {paste CE-05, CE-06, CE-07, CE-22, CE-27}

Same output schema as above, with 3 rewrite options per example that
preserve the stated reason (e.g. "I'll send the invoice through the app")
without any contact info.
```

### 5d. contact_evasion — social/payment handles

```
Generate {N} new examples of intent "contact_evasion", subtype
"social_handle" or "payment_handle" (mix both across the batch). Cover
Telegram, Instagram, Discord, WhatsApp links (wa.me, t.me), and UPI/VPA
payment handles (paytm, okhdfcbank, okicici style). Some examples should
wrap the handle in a real URL (t.me/username); some should be a bare
handle next to a platform name with no URL structure at all.

Seed examples: {paste CE-08, CE-10, CE-11, CE-12, CE-21, CE-24, CE-26}

Same schema. Rewrite options should preserve the underlying reason (e.g.
"I'll share the photos through the app") without the handle.
```

### 5e. contact_evasion — link with decoy/trailing junk

```
Generate {N} new examples of intent "contact_evasion", subtype
"link_decoy". Each message contains a real domain or shortened link
(t.me/..., bit.ly/..., or a full domain+path) immediately followed by
extra characters designed to break naive regex matching: a bracketed
word like "(ignore)", an underscore-joined multi-segment path, or a
second space-separated fragment that continues the path. Vary the decoy
style across the batch — don't repeat the same decoy shape twice.

Seed examples: {paste CE-17, CE-18, CE-26 plus the "(ignore)/villa_owner_goa" case}

Same schema.
```

### 5f. clean_control — the most important category to get right

```
Generate {N} new examples of intent "clean_control" for a vacation rental
chat. These must NOT contain any contact information, but should be
adversarially close to triggering a false positive — pick from:
- digit-heavy legitimate content: booking references, prices, dates,
  guest counts, distances, times (seed: CL-06, CL-07, CL-13, CL-26)
- messages that mention a platform keyword without any actual handle:
  "check out my Instagram for photos of past stays" said abstractly, "call
  the front desk if needed" without a number
- ordinary property Q&A with zero digits or keywords (seed: CL-01, CL-12,
  CL-32, CL-36)

label.contains_violation must be false and label.options must be an empty
array for every example in this batch. Vary sentence length and topic
across bookings, amenities, complaints-that-are-not-coercive, and
logistics.

Seed examples: {paste 5-6 from the CL-* set}
```

### 5g. coercion

```
Generate {N} new examples of intent "coercion" for a vacation rental
chat — messages that threaten a negative review or demand a refund/
discount under implicit or explicit threat, distinct from a genuine
complaint. Also generate {N/3} "near-miss" controls: genuine complaints
that mention dissatisfaction but contain NO threat or demand tied to the
review/refund — these must be labeled contains_violation: false. The
near-miss controls are the most valuable examples in this batch — make
them as close as possible in surface language to the real coercion
examples while genuinely lacking the threat structure.

Seed examples (coercive): {paste CO-01 through CO-10}
Seed examples (near-miss, non-coercive): {paste CO-11, CO-12}

Output schema:
{
  "id": "...", "intent": "coercion", "subtype": "refund_threat" | "review_blackmail" | "near_miss",
  "turns": [...],
  "label": {"contains_violation": bool, "action": "flag_for_moderator" | null, "violation_type": "..." | null}
}
```

### 5h. Language-mix pass (run after the above, on a sampled subset)

```
Here are {N} existing training examples in English:
{paste a batch of already-generated examples, all intents mixed}

Rewrite each one as a natural Hindi-English code-switched (Hinglish)
message, the way a real guest or host would type casually — not a formal
translation. Keep the same intent, subtype, and violation status. If the
original had rewrite options, translate those into the same Hinglish
register. Preserve the id but append "_hi" to it. Return the same JSON
schema as the input.

Seed style reference: {paste LM-07, LM-08, LM-12}
```

---

## 6. Post-generation QC pipeline

1. **Schema validation** — reject/retry anything that doesn't parse against the schema (guided JSON decoding should make this rare, but still validate).
2. **Near-duplicate detection** — embed every `turns` text (a small sentence-transformer model is fine, runs easily alongside the generation model) and flag pairs above a cosine similarity threshold (~0.92) for review. LLM-generated synthetic batches drift toward repeating structures even with diversity instructions — this step is not optional, it's where most of your actual data quality problem will live.
3. **Manual spot-check** — sample ~5% per category, read them, specifically checking that `clean_control` examples are genuinely unambiguous negatives and that `coercion` near-misses are genuinely non-coercive.
4. **Category balance check** — confirm final counts roughly match the target table in section 2 after dedup removal (dedup will hit some categories harder than others, usually the narrower ones like payment handles — regenerate to top up rather than shipping an imbalanced set).
5. **Freeze a slice for the Golden Evaluation Set before this data touches training** — pull ~150-200 examples across every subtype, set them aside untouched, never included in any training run. This is the set every future retrain gets measured against.
