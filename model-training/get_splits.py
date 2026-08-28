import json
import shutil
from pathlib import Path

INPUT_FILE = Path("synthetic_clean.jsonl")
TEMP_FILE = Path("synthetic_clean.tmp.jsonl")
BACKUP_FILE = Path("synthetic_clean.backup.jsonl")


def main():
    removed = 0
    kept = 0

    shutil.copy2(INPUT_FILE, BACKUP_FILE)

    with (
        INPUT_FILE.open("r", encoding="utf-8") as infile,
        TEMP_FILE.open("w", encoding="utf-8") as outfile,
    ):
        for line_number, line in enumerate(infile, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Kept invalid JSON on line {line_number}")
                outfile.write(line)
                kept += 1
                continue

            delivery = record.get("delivery", {})
            label = record.get("label", {})

            should_remove = (
                delivery.get("multiturn") is True
                and delivery.get("language_mix") is True
                and label.get("contains_violation") is True
            )

            if should_remove:
                removed += 1
            else:
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    TEMP_FILE.replace(INPUT_FILE)

    print(f"Removed: {removed}")
    print(f"Kept: {kept}")
    print(f"Backup written to: {BACKUP_FILE}")
    print(f"Updated: {INPUT_FILE}")


if __name__ == "__main__":
    main()