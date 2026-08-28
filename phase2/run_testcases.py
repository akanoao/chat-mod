"""Run the JSON testcase suite through the Phase 1 -> Phase 2 pipeline."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import redis

from feature_extractor import FeatureExtractor
from llm_client import QwenClient
from logger import EvalLogger
from phase1_normalization import run_phase1_normalization
from phase2_pipeline import InterventionHandler, run_phase2
from redis_client import RedisWindowClient
from scorer import RiskScorer


def build_pipeline(redis_host: str, redis_port: int, log_path: str):
    connection = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=0,
        decode_responses=True,
    )
    redis_client = RedisWindowClient(connection)
    extractor = FeatureExtractor(redis_client)
    scorer = RiskScorer()
    llm = QwenClient(endpoint="http://localhost:8080/v1/chat/completions")
    logger = EvalLogger(path=log_path)
    handler = InterventionHandler(redis_client, scorer, llm, logger)
    return redis_client, extractor, scorer, handler


def run_testcase(testcase, redis_client, extractor, scorer, handler, output_log):
    testcase_id = testcase["id"]
    room_id = f"testcase_room_{testcase_id}"
    sender_id = f"testcase_sender_{testcase_id}"

    # Make repeated runs deterministic even when Redis still has old keys.
    redis_client.clear_digit_window(room_id, sender_id)
    redis_client.clear_conversation(room_id, sender_id)
    redis_client.reset_suspicion(sender_id)

    results = []
    print(f"\n{testcase_id} ({testcase['category']})")
    for turn_number, turn in enumerate(testcase["turns"], start=1):
        msg_id = f"{testcase_id}_turn_{turn_number}"
        phase1_result = run_phase1_normalization(
            turn["text"], room_id, sender_id, msg_id
        )
        result = run_phase2(
            phase1_result,
            turn["speaker"],
            redis_client,
            extractor,
            scorer,
            handler,
        )
        results.append(result)
        output_log.write(
            json.dumps(
                {
                    "testcase_id": testcase_id,
                    "category": testcase["category"],
                    "turn": turn_number,
                    "msg_id": msg_id,
                    "speaker": turn["speaker"],
                    "input_text": turn["text"],
                    "normalized_text": phase1_result["final_text"],
                    "output": result,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        output_log.flush()
        print(
            f"  msg={result['msg_id']} tier={result['tier']:8s} "
            f"score={result['score']:3d} action={result['action']}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "testcases.json",
        help="Path to the testcase JSON file",
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--log-path", default="phase2_testcase_events.jsonl")
    parser.add_argument(
        "--output-log",
        default="phase2_testcase_outputs.jsonl",
        help="JSONL file containing input and final pipeline output for every message",
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as testcase_file:
        testcases = json.load(testcase_file)

    redis_client, extractor, scorer, handler = build_pipeline(
        args.redis_host, args.redis_port, args.log_path
    )
    redis_client.r.ping()

    overall_tiers = Counter()
    overall_message_ids = defaultdict(list)
    testcase_summaries = []

    with open(args.output_log, "w", encoding="utf-8") as output_log:
        for testcase in testcases:
            results = run_testcase(
                testcase, redis_client, extractor, scorer, handler, output_log
            )
            tier_counts = Counter(result["tier"] for result in results)
            message_ids_by_tier = defaultdict(list)
            for result in results:
                message_ids_by_tier[result["tier"]].append(result["msg_id"])
            overall_tiers.update(tier_counts)
            for tier, message_ids in message_ids_by_tier.items():
                overall_message_ids[tier].extend(message_ids)
            testcase_summaries.append(
                (testcase["id"], tier_counts, message_ids_by_tier)
            )

    print("\n=== Moderation summary ===")
    for testcase_id, tier_counts, message_ids_by_tier in testcase_summaries:
        print(
            f"{testcase_id}: low={tier_counts['low']} "
            f"moderate={tier_counts['moderate']} high={tier_counts['high']}"
        )
        for tier in ("low", "moderate", "high"):
            print(f"  {tier}: {', '.join(message_ids_by_tier[tier]) or '-'}")
    print(
        f"TOTAL messages={sum(overall_tiers.values())} "
        f"low={overall_tiers['low']} moderate={overall_tiers['moderate']} "
        f"high={overall_tiers['high']}"
    )
    for tier in ("low", "moderate", "high"):
        print(f"TOTAL {tier}: {', '.join(overall_message_ids[tier]) or '-'}")


if __name__ == "__main__":
    main()