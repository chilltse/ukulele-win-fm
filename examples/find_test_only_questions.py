import argparse
import csv
import json
from pathlib import Path


def normalize_qid(token: str) -> str:
    s = str(token).strip()
    if not s or s.upper() == "NA" or s == "-1":
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


def _split_multi_kc_token(kc_token: str):
    raw = str(kc_token).strip()
    if not raw or raw.upper() == "NA" or raw == "-1":
        return []
    return [x.strip() for x in raw.split("_") if x.strip() and x.strip() != "-1"]


def extract_question_and_kcs(csv_path: Path):
    qids = set()
    qid_to_kcs = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "questions" not in (reader.fieldnames or []):
            raise ValueError(f"`questions` column not found in {csv_path}")
        has_concepts = "concepts" in (reader.fieldnames or [])
        has_selectmasks = "selectmasks" in (reader.fieldnames or [])

        for row in reader:
            q_tokens = str(row.get("questions", "")).split(",")
            c_tokens = str(row.get("concepts", "")).split(",") if has_concepts else []
            if has_selectmasks:
                m_tokens = str(row.get("selectmasks", "")).split(",")
            else:
                m_tokens = []

            for i, q in enumerate(q_tokens):
                if has_selectmasks and i < len(m_tokens):
                    if str(m_tokens[i]).strip() == "-1":
                        continue
                qid = normalize_qid(q)
                if qid:
                    qids.add(qid)
                    qid_to_kcs.setdefault(qid, set())
                    if has_concepts and i < len(c_tokens):
                        for kc in _split_multi_kc_token(c_tokens[i]):
                            qid_to_kcs[qid].add(kc)

    return qids, qid_to_kcs


def sorted_qids(qset):
    def key_fn(x):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)

    return sorted(qset, key=key_fn)


def load_internal_to_raw_kc_map(keyid2idx_path: Path):
    if not keyid2idx_path.exists():
        return {}
    with keyid2idx_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    concepts = obj.get("concepts", {}) if isinstance(obj, dict) else {}
    if not isinstance(concepts, dict):
        return {}
    internal_to_raw = {}
    for raw_kc, internal_idx in concepts.items():
        internal_to_raw[str(internal_idx)] = str(raw_kc)
    return internal_to_raw


def main():
    parser = argparse.ArgumentParser(
        description="Find question IDs that appear only in test set (not in train/valid)."
    )
    parser.add_argument(
        "--train-valid-csv",
        type=str,
        default="../data/xes3g5m/question_level/train_valid_sequences_quelevel.csv",
        help="Train+valid sequence CSV path.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="../data/xes3g5m/question_level/test_quelevel.csv",
        help="Test CSV path.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="../data/xes3g5m/question_level/test_only_question_ids.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-kc-csv",
        type=str,
        default="../data/xes3g5m/question_level/test_only_question_kcs.csv",
        help="Output CSV path with KC mapping for each test-only question.",
    )
    parser.add_argument(
        "--kc-map-json",
        type=str,
        default="../data/xes3g5m/metadata/kc_routes_map.json",
        help="Optional KC id -> KC name json path.",
    )
    parser.add_argument(
        "--keyid2idx-json",
        type=str,
        default="../data/xes3g5m/keyid2idx.json",
        help="Path to keyid2idx.json for internal concept-id -> raw KC-id mapping.",
    )
    args = parser.parse_args()

    train_valid_path = Path(args.train_valid_csv).resolve()
    test_path = Path(args.test_csv).resolve()
    output_path = Path(args.output_csv).resolve()
    output_kc_path = Path(args.output_kc_csv).resolve()
    kc_map_path = Path(args.kc_map_json).resolve()
    keyid2idx_path = Path(args.keyid2idx_json).resolve()

    if not train_valid_path.exists():
        raise FileNotFoundError(f"train/valid CSV not found: {train_valid_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test CSV not found: {test_path}")

    train_valid_qids, _ = extract_question_and_kcs(train_valid_path)
    test_qids, test_qid_to_kcs = extract_question_and_kcs(test_path)
    test_only_qids = sorted_qids(test_qids - train_valid_qids)

    kc_name_map = {}
    if kc_map_path.exists():
        with kc_map_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict):
                kc_name_map = {str(k): str(v) for k, v in obj.items()}
    internal_to_raw_kc = load_internal_to_raw_kc_map(keyid2idx_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id"])
        for qid in test_only_qids:
            writer.writerow([qid])

    output_kc_path.parent.mkdir(parents=True, exist_ok=True)
    with output_kc_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "kc_ids", "kc_names"])
        for qid in test_only_qids:
            internal_kcs = sorted_qids(test_qid_to_kcs.get(qid, set()))
            raw_kcs = [internal_to_raw_kc.get(str(kc), str(kc)) for kc in internal_kcs]
            raw_kcs = sorted_qids(set(raw_kcs))
            kc_names = [kc_name_map.get(str(kc), "") for kc in raw_kcs]
            writer.writerow([qid, "|".join(raw_kcs), "|".join(kc_names)])

    print(f"train_valid unique questions: {len(train_valid_qids)}")
    print(f"test unique questions: {len(test_qids)}")
    print(f"test-only questions: {len(test_only_qids)}")
    print(f"output: {output_path}")
    print(f"output with KCs: {output_kc_path}")
    if test_only_qids:
        print(f"first 20 test-only question ids: {test_only_qids[:20]}")


if __name__ == "__main__":
    main()
