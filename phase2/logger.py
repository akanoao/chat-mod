import json
import time
import dataclasses


class EvalLogger:
    """Writes one JSON line per Phase 2 decision. This is the raw material
    for both the golden-eval-adjacent monitoring and the weekly retrain
    pool -- every call site in phase2_pipeline.py/InterventionHandler
    routes through here."""

    def __init__(self, path: str = "phase2_events.jsonl"):
        self.path = path

    def log(self, msg_id, tier, score, features, action: str, reason: str = None):
        record = {
            "msg_id": msg_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tier": tier.value if hasattr(tier, "value") else tier,
            "score": score,
            "features": dataclasses.asdict(features),
            "action": action,
            "reason": reason,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
