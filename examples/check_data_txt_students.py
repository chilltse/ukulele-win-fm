import argparse
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_data_txt(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    headers = lines[0::6]

    uids = []
    seq_lens = []
    bad_headers = 0

    for header in headers:
        parts = header.split(",")
        if len(parts) < 2:
            bad_headers += 1
            continue
        uid = parts[0].strip()
        seq_len_raw = parts[1].strip()
        try:
            seq_len = int(seq_len_raw)
        except ValueError:
            bad_headers += 1
            continue
        uids.append(uid)
        seq_lens.append(seq_len)

    return {
        "line_count": len(lines),
        "block_count": len(headers),
        "uids": uids,
        "seq_lens": seq_lens,
        "bad_headers": bad_headers,
    }


def parse_source_students(source_json: Path):
    records = json.loads(source_json.read_text(encoding="utf-8"))
    return {str(r.get("student_id")) for r in records if r.get("student_id") is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_txt",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dbe_kt22" / "data.txt"),
        help="Path to generated data.txt",
    )
    parser.add_argument(
        "--source_json",
        type=str,
        default=str(
            PROJECT_ROOT
            / "data"
            / "dbe_kt22"
            / "2_DBE_KT22_Practice_Sequences_100102_json"
            / "Practice_Sequences.json"
        ),
        help="Optional source JSON to compare unique student count",
    )
    parser.add_argument(
        "--output_ids",
        type=str,
        default=str(PROJECT_ROOT / "data" / "dbe_kt22" / "unique_student_ids.txt"),
        help="Path to save all unique student IDs (one per line)",
    )
    args = parser.parse_args()

    data_txt = Path(args.data_txt)
    stats = parse_data_txt(data_txt)

    uid_counter = Counter(stats["uids"])
    unique_uids = len(uid_counter)
    duplicate_uids = {uid: cnt for uid, cnt in uid_counter.items() if cnt > 1}
    unique_uid_list = sorted(uid_counter.keys(), key=lambda x: int(x) if x.isdigit() else x)

    print(f"data_txt: {data_txt.resolve()}")
    print(f"line_count: {stats['line_count']}")
    print(f"block_count(=student sequences): {stats['block_count']}")
    print(f"valid_headers: {len(stats['uids'])}")
    print(f"bad_headers: {stats['bad_headers']}")
    print(f"unique_students: {unique_uids}")
    print(f"duplicate_student_ids: {len(duplicate_uids)}")
    print(f"total_seq_len(sum of headers): {sum(stats['seq_lens'])}")

    output_ids = Path(args.output_ids)
    output_ids.parent.mkdir(parents=True, exist_ok=True)
    output_ids.write_text("\n".join(unique_uid_list) + "\n", encoding="utf-8")
    print(f"saved_unique_student_ids: {output_ids.resolve()}")

    if duplicate_uids:
        print("duplicate_examples(top10):")
        for uid, cnt in list(sorted(duplicate_uids.items(), key=lambda x: x[1], reverse=True))[:10]:
            print(f"  student_id={uid}, count={cnt}")

    source_json = Path(args.source_json)
    if source_json.exists():
        src_students = parse_source_students(source_json)
        print(f"source_unique_students: {len(src_students)}")
        print(f"missing_in_data_txt: {len(src_students - set(stats['uids']))}")
        print(f"extra_in_data_txt: {len(set(stats['uids']) - src_students)}")


if __name__ == "__main__":
    main()
