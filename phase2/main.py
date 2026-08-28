"""
Wires everything together and runs a few sample messages through the
full Phase 1 -> Phase 2 pipeline, including a multi-turn split case, so
you can see the whole thing work before plugging it into your actual
chat backend.

Requires: pip install redis rapidfuzz requests
Requires a running Redis instance (see section below) and a running
llama-server serving the quantized model (see the deploy section).
"""

import redis

from phase1_normalization import run_phase1_normalization
from redis_client import RedisWindowClient
from feature_extractor import FeatureExtractor
from scorer import RiskScorer
from llm_client import QwenClient
from logger import EvalLogger
from phase2_pipeline import InterventionHandler, run_phase2


def build_pipeline():
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    redis_client = RedisWindowClient(r)
    extractor = FeatureExtractor(redis_client)
    scorer = RiskScorer()
    llm = QwenClient(endpoint="http://localhost:8080/v1/chat/completions")
    logger = EvalLogger(path="phase2_events.jsonl")
    handler = InterventionHandler(redis_client, scorer, llm, logger)
    return redis_client, extractor, scorer, handler


def process_message(raw_text, room_id, sender_id, speaker, msg_id,
                     redis_client, extractor, scorer, handler):
    p1_result = run_phase1_normalization(raw_text, room_id, sender_id, msg_id)
    p2_result = run_phase2(p1_result, speaker, redis_client, extractor, scorer, handler)
    print(f"[{msg_id}] tier={p2_result['tier']:8s} score={p2_result['score']:3d}  "
          f"action={p2_result['action']}")
    if p2_result["action"] == "soft_mask":
        print(f"           -> delivered: {p2_result['text']}")
    elif p2_result["action"] == "llm_escalation":
        for i, opt in enumerate(p2_result.get("options", []), 1):
            print(f"           -> chip {i}: {opt}")
    elif p2_result["action"] == "block":
        print(f"           -> blocked: {p2_result['reason']}")
    elif p2_result["action"] == "pass":
        print(f"           -> delivered unchanged: {p2_result['text']}")
    return p2_result


if __name__ == "__main__":
    redis_client, extractor, scorer, handler = build_pipeline()

    room = "room_demo_1"

    # print("--- single-turn, obvious digit obfuscation ---")
    # process_message(
    #     "hey can you whatsapp me on nine eight",
    #     "room1", "sender_A", "guest", "msg_001",
    #     redis_client, extractor, scorer, handler,
    # )

    # print("\n--- clean control, should pass untouched ---")
    # process_message(
    #     "What's the price for 5 nights in December?",
    #     room, "sender_B", "guest", "msg_002",
    #     redis_client, extractor, scorer, handler,
    # )

    # print("\n--- multi-turn digit split from sender_z, same room ---")
    # for i, fragment in enumerate(["nine eight seven", "six five four", "three two one zero"], 1):
    #     process_message(
    #         fragment, "room3", "sender_z", "guest", f"msg_00{2+i}",
    #         redis_client, extractor, scorer, handler,
    #     )

    print("\n--- cross-turn handle: platform keyword then bare handle ---")
    process_message("here's my insta handle for the deposit screenshot: @traveler.k.29", room, "sender_C", "guest", "msg_010", redis_client, extractor, scorer, handler)
    # process_message("sunny_traveller99", room, "sender_C", "guest", "msg_011", redis_client, extractor, scorer, handler)
    # process_message("on telegram", room, "sender_C", "guest", "msg_011", redis_client, extractor, scorer, handler)
