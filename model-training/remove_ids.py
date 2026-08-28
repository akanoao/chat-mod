import json
from pathlib import Path

INPUT_FILE = Path("synthetic_clean.jsonl")
TEMP_FILE = Path("synthetic_clean.tmp.jsonl")

IDS_TO_REMOVE = {
    "clean_digit_heavy_0376",
    "social_handle_0283",
    "clean_ordinary_0099",
    "clean_ordinary_0079",
    "clean_digit_heavy_0315",
    "clean_digit_heavy_0397",
    "clean_keyword_adjacent_0114",
    "clean_ordinary_0151",
    "social_handle_0120",
    "clean_ordinary_0078",
    "clean_digit_heavy_0235",
    "clean_ordinary_0139",
    "clean_ordinary_0107",
    "social_handle_0084",
    "clean_keyword_adjacent_0209",
    "clean_ordinary_0159",
}


def main():
    found_ids = set()
    kept_records = []

    with INPUT_FILE.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, 1):
            if not line.strip():
                continue

            record = json.loads(line)
            record_id = record.get("id")

            if record_id in IDS_TO_REMOVE:
                found_ids.add(record_id)
            else:
                kept_records.append(record)

    missing_ids = IDS_TO_REMOVE - found_ids

    with TEMP_FILE.open("w", encoding="utf-8") as outfile:
        for record in kept_records:
            outfile.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print(f"Removed: {len(found_ids)}")
    print(f"Remaining: {len(kept_records)}")

    if missing_ids:
        print("Not found:")
        for record_id in sorted(missing_ids):
            print(f"  {record_id}")

    TEMP_FILE.replace(INPUT_FILE)

    print(f"Replaced: {INPUT_FILE}")


if __name__ == "__main__":
    main()