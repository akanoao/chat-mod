"""
Post-generation QC pipeline (section 6 of the synthetic data guide):
1. Schema validation — drop/report anything malformed.
2. Near-duplicate detection via sentence embeddings, not exact-match.
3. Category balance check against target counts.
4. Random spot-check sample export (~5% per category) for manual review.

Requires: pip install sentence-transformers pydantic numpy

Usage:
    python qc_pipeline.py --input synthetic_raw.jsonl --output synthetic_clean.jsonl
"""

import json
import argparse
import collections
from typing import List, Optional
from pydantic import BaseModel, ValidationError
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# SCHEMA — matches the final record shape written by the generation script
# ---------------------------------------------------------------------

class Turn(BaseModel):
    speaker: str
    text: str

class Delivery(BaseModel):
    multiturn: bool
    language_mix: bool

class Label(BaseModel):
    contains_violation: bool
    options: List[str] = []
    action: Optional[str] = None
    violation_type: Optional[str] = None

class FullExample(BaseModel):
    id: str
    intent: str
    subtype: str
    delivery: Delivery
    turns: List[Turn]
    label: Label
    metadata: dict

# ---------------------------------------------------------------------
# TARGET COUNTS — for the balance check, mirrors the volume table
# ---------------------------------------------------------------------

TARGET_COUNTS = {
    "digit_obfuscation": 450,
    "digit_script_mixed": 150,
    "email_obfuscation": 300,
    "social_handle": 300,
    "payment_handle": 150,
    "link_decoy": 150,
    "clean_digit_heavy": 450,
    "clean_keyword_adjacent": 250,
    "clean_ordinary": 200,
}

NEAR_DUP_DROP_THRESHOLD = 0.92   # auto-drop pairs at or above this similarity
NEAR_DUP_REVIEW_THRESHOLD = 0.85  # log but don't auto-drop between this and the drop threshold
SPOT_CHECK_FRACTION = 0.05


# ---------------------------------------------------------------------
# STEP 1 — SCHEMA VALIDATION
# ---------------------------------------------------------------------

def load_and_validate(path: str):
    valid, invalid = [], []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                example = FullExample.model_validate(raw)
                valid.append(example)
            except (json.JSONDecodeError, ValidationError) as e:
                invalid.append({"line": line_num, "error": str(e), "raw": line[:200]})
    print(f"[schema] {len(valid)} valid, {len(invalid)} invalid")
    if invalid:
        with open("qc_invalid_records.jsonl", "w", encoding="utf-8") as f:
            for item in invalid:
                f.write(json.dumps(item) + "\n")
        print(f"[schema] invalid records logged to qc_invalid_records.jsonl")
    return valid


# ---------------------------------------------------------------------
# STEP 1b — OPTIONS QUALITY CHECK
# Pydantic's List[str] is satisfied by ["phone_number"] just as much as by
# three real sentences, so this catches the failure mode schema validation
# can't: a violation example whose "options" are category tags/placeholders
# instead of actual rewritten sentences.
# ---------------------------------------------------------------------

MIN_OPTION_WORDS = 4  # a real rewrite sentence should clear this easily

def check_options_quality(examples: List[FullExample]):
    flagged = []
    for ex in examples:
        if not ex.label.contains_violation:
            if ex.label.options:
                flagged.append((ex.id, "clean_control example has non-empty options", ex.label.options))
            continue
        if len(ex.label.options) != 3:
            flagged.append((ex.id, f"expected 3 options, got {len(ex.label.options)}", ex.label.options))
            continue
        for opt in ex.label.options:
            if len(opt.split()) < MIN_OPTION_WORDS:
                flagged.append((ex.id, "option looks like a category tag, not a sentence", ex.label.options))
                break

    print(f"[options] {len(flagged)} examples flagged out of {len(examples)}")
    if flagged:
        with open("qc_flagged_options.jsonl", "w") as f:
            for id_, reason, options in flagged:
                f.write(json.dumps({"id": id_, "reason": reason, "options": options}) + "\n")
        print(f"[options] logged to qc_flagged_options.jsonl — these need regeneration, "
              f"not just review; a category-tag option won't teach the model to rewrite anything")
    flagged_ids = {id_ for id_, _, _ in flagged}
    return [ex for ex in examples if ex.id not in flagged_ids], flagged_ids


# ---------------------------------------------------------------------
# STEP 2 — NEAR-DUPLICATE DETECTION (embedding-based, not exact-match)
# ---------------------------------------------------------------------

def example_text(ex: FullExample) -> str:
    return " ".join(t.text for t in ex.turns)


class UnionFind:
    """Groups near-duplicate examples into clusters so we keep one per cluster
    instead of just dropping every pairwise match (which can cascade oddly)."""
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def dedup_near_duplicates(examples: List[FullExample], model_name="all-MiniLM-L6-v2"):
    print(f"[dedup] embedding {len(examples)} examples with {model_name}...")
    model = SentenceTransformer(model_name)
    texts = [example_text(ex) for ex in examples]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    # cosine similarity via dot product on normalized embeddings
    sim_matrix = embeddings @ embeddings.T
    n = len(examples)
    uf = UnionFind(n)
    review_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if sim >= NEAR_DUP_DROP_THRESHOLD:
                uf.union(i, j)
            elif sim >= NEAR_DUP_REVIEW_THRESHOLD:
                review_pairs.append((i, j, float(sim)))

    clusters = collections.defaultdict(list)
    for idx in range(n):
        clusters[uf.find(idx)].append(idx)

    kept_indices = []
    dropped_count = 0
    for root, members in clusters.items():
        if len(members) == 1:
            kept_indices.append(members[0])
        else:
            # keep the longest example in the cluster (usually the most detailed/informative)
            best = max(members, key=lambda idx: len(texts[idx]))
            kept_indices.append(best)
            dropped_count += len(members) - 1

    print(f"[dedup] dropped {dropped_count} near-duplicates "
          f"(similarity >= {NEAR_DUP_DROP_THRESHOLD}), kept {len(kept_indices)}")

    if review_pairs:
        with open("qc_review_pairs.jsonl", "w", encoding="utf-8") as f:
            for i, j, sim in sorted(review_pairs, key=lambda x: -x[2]):
                f.write(json.dumps({
                    "similarity": round(sim, 4),
                    "example_a": examples[i].id, "text_a": texts[i],
                    "example_b": examples[j].id, "text_b": texts[j],
                }) + "\n")
        print(f"[dedup] {len(review_pairs)} borderline pairs "
              f"({NEAR_DUP_REVIEW_THRESHOLD}-{NEAR_DUP_DROP_THRESHOLD} similarity) "
              f"logged to qc_review_pairs.jsonl for manual review — not auto-dropped")

    return [examples[i] for i in kept_indices]


# ---------------------------------------------------------------------
# STEP 3 — CATEGORY BALANCE CHECK
# ---------------------------------------------------------------------

def check_balance(examples: List[FullExample]):
    counts = collections.Counter(ex.subtype for ex in examples)
    print("\n[balance] subtype counts vs targets:")
    shortfalls = {}
    for subtype, target in TARGET_COUNTS.items():
        actual = counts.get(subtype, 0)
        pct = (actual / target * 100) if target else 0
        flag = "  <-- regenerate to top up" if pct < 90 else ""
        print(f"  {subtype:28s} {actual:4d} / {target:4d} ({pct:5.1f}%){flag}")
        if pct < 90:
            shortfalls[subtype] = target - actual

    unexpected = set(counts) - set(TARGET_COUNTS)
    if unexpected:
        print(f"[balance] subtypes not in target table (language-mix variants, etc.): "
              f"{dict((s, counts[s]) for s in unexpected)}")

    if shortfalls:
        print(f"\n[balance] shortfalls after dedup — feed these counts back into "
              f"the generation script's targets and re-run to top up: {shortfalls}")
    return shortfalls


# ---------------------------------------------------------------------
# STEP 4 — SPOT-CHECK SAMPLE EXPORT
# ---------------------------------------------------------------------

def export_spot_check(examples: List[FullExample], fraction=SPOT_CHECK_FRACTION):
    import random
    by_subtype = collections.defaultdict(list)
    for ex in examples:
        by_subtype[ex.subtype].append(ex)

    sample = []
    for subtype, group in by_subtype.items():
        n = max(1, int(len(group) * fraction))
        sample.extend(random.sample(group, min(n, len(group))))

    with open("qc_spot_check_sample.jsonl", "w", encoding="utf-8") as f:
        for ex in sample:
            f.write(ex.model_dump_json() + "\n")

    print(f"\n[spot-check] {len(sample)} examples across {len(by_subtype)} subtypes "
          f"exported to qc_spot_check_sample.jsonl for manual review — "
          f"specifically verify clean_control examples are genuinely unambiguous negatives")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="synthetic_raw.jsonl")
    parser.add_argument("--output", default="synthetic_clean.jsonl")
    args = parser.parse_args()

    valid = load_and_validate(args.input)
    quality_ok, flagged_ids = check_options_quality(valid)
    deduped = dedup_near_duplicates(quality_ok)
    check_balance(deduped)
    export_spot_check(deduped)

    with open(args.output, "w", encoding="utf-8") as f:
        for ex in deduped:
            f.write(ex.model_dump_json() + "\n")

    print(f"\n[done] {len(deduped)} clean examples written to {args.output}")
    print("Still manual: read qc_spot_check_sample.jsonl and qc_review_pairs.jsonl "
          "before this data goes anywhere near training.")


if __name__ == "__main__":
    main()