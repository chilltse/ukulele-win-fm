import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_folds(folds_text: str) -> Set[int]:
    parts = [p.strip() for p in folds_text.split(",") if p.strip()]
    if not parts:
        raise ValueError("`--train-folds` is empty. Example: 1,2,3,4")
    return {int(p) for p in parts}


def normalize_token(token: object) -> str:
    return str(token).strip()


def numeric_sort_key(x: object):
    s = str(x)
    return (0, int(s)) if s.isdigit() else (1, s)


def is_pad_token(token: object, pad_val: int = -1) -> bool:
    token = normalize_token(token)
    if token == "":
        return True

    parts = [p.strip() for p in token.split("^")]
    if not parts:
        return True

    try:
        values = [int(p) for p in parts]
    except ValueError:
        return False

    return all(v == pad_val for v in values)


def split_seq(raw: object) -> List[str]:
    return [x.strip() for x in str(raw).split(",")]


def split_concept_token_to_discrete_kcs(token: str, pad_val: int = -1) -> List[str]:
    """
    Split packed KC tokens into discrete original KC ids.

    Example:
        "304_484_399" -> ["304", "484", "399"]
        "304" -> ["304"]
        "-1" -> []
        "-1_-1" -> []

    This function only returns discrete KCs.
    It will not keep packed KC strings such as "304_484_399".
    """
    token = normalize_token(token)

    if token == "" or is_pad_token(token, pad_val=pad_val):
        return []

    parts = [p.strip() for p in token.split("_") if p.strip()]

    discrete_kcs = []
    seen = set()

    for p in parts:
        if p == "" or p == str(pad_val):
            continue

        if p not in seen:
            discrete_kcs.append(p)
            seen.add(p)

    return discrete_kcs


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def build_idx2key_maps(keyid2idx_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """
    keyid2idx.json format:
        {
          "questions": {"3751": 0, "3752": 1, ...},
          "concepts": {"304": 526, ...},
          "uid": {...}
        }

    This function builds reverse maps:
        {
          "questions": {"0": "3751", "1": "3752", ...},
          "concepts": {"526": "304", ...},
          "uid": {...}
        }
    """
    if keyid2idx_path is None:
        return {}

    if not keyid2idx_path.exists():
        raise FileNotFoundError(f"keyid2idx file not found: {keyid2idx_path}")

    keyid2idx = load_json(keyid2idx_path)

    idx2key_maps: Dict[str, Dict[str, str]] = {}

    for field, mapping in keyid2idx.items():
        if not isinstance(mapping, dict):
            continue

        idx2key_maps[field] = {}

        for original_id, internal_id in mapping.items():
            internal_id_str = str(internal_id)
            original_id_str = str(original_id)

            if internal_id_str in idx2key_maps[field]:
                raise ValueError(
                    f"Duplicate internal id in keyid2idx[{field!r}]: {internal_id_str}. "
                    f"This would make reverse mapping ambiguous."
                )

            idx2key_maps[field][internal_id_str] = original_id_str

    return idx2key_maps


def restore_id_token(
    token: object,
    idx2key_map: Optional[Dict[str, str]],
    pad_val: int = -1,
) -> str:
    """
    Convert one internal id token back to original id.

    Example:
        token="526", idx2key_map["526"]="304" -> "304"

    Important:
    If the internal id represents a packed original concept:
        token="52", idx2key_map["52"]="304_484_399" -> "304_484_399"

    If exact token is not found, this function falls back to the original token.
    """
    token = normalize_token(token)

    if token == "" or token == str(pad_val):
        return token

    if not idx2key_map:
        return token

    if token in idx2key_map:
        return idx2key_map[token]

    # Robust fallback: handle rare cases like "12_13" where each part is an internal id.
    if "_" in token:
        parts = [p.strip() for p in token.split("_") if p.strip()]
        restored_parts = [idx2key_map.get(p, p) for p in parts]
        return "_".join(restored_parts)

    return token


def restore_sequence(
    raw_seq: object,
    idx2key_map: Optional[Dict[str, str]],
    pad_val: int = -1,
) -> str:
    return ",".join(
        restore_id_token(tok, idx2key_map=idx2key_map, pad_val=pad_val)
        for tok in split_seq(raw_seq)
    )


def choose_train_folds(
    df: pd.DataFrame,
    valid_fold: int,
    train_folds_text: Optional[str],
) -> Set[int]:
    all_folds = {int(x) for x in sorted(df["fold"].dropna().unique())}

    if train_folds_text:
        train_folds = parse_folds(train_folds_text)
        unknown = sorted(train_folds - all_folds)

        if unknown:
            raise ValueError(
                f"Unknown folds in --train-folds: {unknown}. "
                f"Available folds: {sorted(all_folds)}"
            )

        return train_folds

    if valid_fold not in all_folds:
        raise ValueError(f"--valid-fold={valid_fold} not in data folds {sorted(all_folds)}")

    train_folds = all_folds - {valid_fold}

    if not train_folds:
        raise ValueError("No train folds left after excluding valid fold.")

    return train_folds


def load_data_config(data_config_path: Path) -> Dict[str, object]:
    if not data_config_path.exists():
        raise FileNotFoundError(f"data_config.json not found: {data_config_path}")

    with data_config_path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def resolve_dataset_paths(
    dataset_name: str,
    data_config_path: Path,
    input_csv_arg: str,
    output_csv_arg: str,
    covered_questions_output_csv_arg: str,
    restored_train_csv_arg: str,
) -> Tuple[Path, Path, Path, Optional[Path]]:
    if input_csv_arg.strip():
        input_csv = Path(input_csv_arg)

        if output_csv_arg.strip():
            output_csv = Path(output_csv_arg)
        else:
            output_csv = input_csv.with_name("kc_train_stats_original_ids.csv")

        if covered_questions_output_csv_arg.strip():
            covered_questions_output_csv = Path(covered_questions_output_csv_arg)
        else:
            covered_questions_output_csv = input_csv.with_name("kc_train_covered_questions_original_ids.csv")

        restored_train_csv = Path(restored_train_csv_arg) if restored_train_csv_arg.strip() else None

        return input_csv, output_csv, covered_questions_output_csv, restored_train_csv

    config = load_data_config(data_config_path)

    if dataset_name not in config:
        raise ValueError(f"dataset_name={dataset_name!r} not found in {data_config_path}")

    dataset_cfg = config[dataset_name]

    if not isinstance(dataset_cfg, dict):
        raise ValueError(f"Invalid dataset config for {dataset_name}: expected dict")

    dpath_raw = dataset_cfg.get("dpath", "")

    if not dpath_raw:
        raise ValueError(f"Missing `dpath` in dataset config: {dataset_name}")

    dpath = (data_config_path.parent / str(dpath_raw)).resolve()

    train_valid_original_file_quelevel = dataset_cfg.get("train_valid_original_file_quelevel")

    if not train_valid_original_file_quelevel:
        raise ValueError(
            f"Missing train_valid_original_file_quelevel in dataset config: {dataset_name}"
        )

    input_csv = (dpath / str(train_valid_original_file_quelevel)).resolve()

    if output_csv_arg.strip():
        output_csv = Path(output_csv_arg)
    else:
        output_csv = (dpath / "kc_train_stats_original_ids.csv").resolve()

    if covered_questions_output_csv_arg.strip():
        covered_questions_output_csv = Path(covered_questions_output_csv_arg)
    else:
        covered_questions_output_csv = (dpath / "kc_train_covered_questions_original_ids.csv").resolve()

    restored_train_csv = Path(restored_train_csv_arg) if restored_train_csv_arg.strip() else None

    return input_csv, output_csv, covered_questions_output_csv, restored_train_csv


def restore_train_dataframe_ids(
    df_train: pd.DataFrame,
    idx2key_maps: Dict[str, Dict[str, str]],
    pad_val: int = -1,
) -> pd.DataFrame:
    """
    Return a copy of df_train where questions/concepts/uid are restored to original ids.

    This is mainly for debugging and inspection.
    """
    out = df_train.copy()

    if "questions" in out.columns:
        out["questions"] = out["questions"].apply(
            lambda x: restore_sequence(
                x,
                idx2key_map=idx2key_maps.get("questions"),
                pad_val=pad_val,
            )
        )

    if "concepts" in out.columns:
        out["concepts"] = out["concepts"].apply(
            lambda x: restore_sequence(
                x,
                idx2key_map=idx2key_maps.get("concepts"),
                pad_val=pad_val,
            )
        )

    if "uid" in out.columns:
        out["uid"] = out["uid"].apply(
            lambda x: restore_id_token(
                x,
                idx2key_map=idx2key_maps.get("uid"),
                pad_val=pad_val,
            )
        )

    return out


def analyze_kc_stats(
    df_train: pd.DataFrame,
    idx2key_maps: Dict[str, Dict[str, str]],
    pad_val: int = -1,
    min_practices: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    attempts: Dict[str, int] = defaultdict(int)
    correct_sum: Dict[str, int] = defaultdict(int)
    correct_list: Dict[str, List[int]] = defaultdict(list)

    covered_questions: Dict[str, Set[str]] = defaultdict(set)
    covered_single_kc_questions: Dict[str, Set[str]] = defaultdict(set)
    covered_multi_kc_questions: Dict[str, Set[str]] = defaultdict(set)

    covered_students: Dict[str, Set[str]] = defaultdict(set)
    repeat_sum: Dict[str, int] = defaultdict(int)

    # kc -> qid -> set({"single", "multi"})
    question_cover_modes: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    missing_triples = 0
    invalid_responses = 0
    packed_concept_tokens = 0

    unmapped_question_tokens: Set[str] = set()
    unmapped_concept_tokens: Set[str] = set()

    q_idx2key = idx2key_maps.get("questions")
    c_idx2key = idx2key_maps.get("concepts")
    u_idx2key = idx2key_maps.get("uid")

    for _, row in df_train.iterrows():
        raw_uid = str(row.get("uid", ""))
        uid = restore_id_token(raw_uid, idx2key_map=u_idx2key, pad_val=pad_val)

        q_seq_raw = split_seq(row["questions"])
        c_seq_raw = split_seq(row["concepts"])
        r_seq = split_seq(row["responses"])

        rep_seq: Optional[List[str]] = None
        if "is_repeat" in row.index:
            rep_seq = split_seq(row["is_repeat"])

        seq_len = min(len(q_seq_raw), len(c_seq_raw), len(r_seq))

        if seq_len == 0:
            continue

        if len(q_seq_raw) != len(c_seq_raw) or len(c_seq_raw) != len(r_seq):
            missing_triples += 1

        for i in range(seq_len):
            q_raw = normalize_token(q_seq_raw[i])
            c_raw = normalize_token(c_seq_raw[i])
            r_tok = normalize_token(r_seq[i])

            if q_raw == "" or q_raw == str(pad_val):
                continue

            if c_raw == "" or is_pad_token(c_raw, pad_val=pad_val):
                continue

            if r_tok in ("", str(pad_val)):
                continue

            qid = restore_id_token(q_raw, idx2key_map=q_idx2key, pad_val=pad_val)
            concept_token = restore_id_token(c_raw, idx2key_map=c_idx2key, pad_val=pad_val)

            if q_idx2key and q_raw not in q_idx2key and q_raw != str(pad_val):
                unmapped_question_tokens.add(q_raw)

            if c_idx2key and c_raw not in c_idx2key and c_raw != str(pad_val):
                unmapped_concept_tokens.add(c_raw)

            discrete_kcs = split_concept_token_to_discrete_kcs(concept_token, pad_val=pad_val)

            if not discrete_kcs:
                continue

            try:
                resp = int(r_tok)
            except ValueError:
                invalid_responses += 1
                continue

            if resp not in (0, 1):
                invalid_responses += 1
                continue

            is_multi_kc_token = len(discrete_kcs) > 1

            if is_multi_kc_token:
                packed_concept_tokens += 1
                cover_mode = "multi"
            else:
                cover_mode = "single"

            rep_val = 0

            if rep_seq is not None and i < len(rep_seq):
                try:
                    rep_val = int(rep_seq[i])
                except ValueError:
                    rep_val = 0

            for kc in discrete_kcs:
                attempts[kc] += 1
                correct_sum[kc] += resp
                correct_list[kc].append(resp)

                covered_questions[kc].add(qid)
                question_cover_modes[kc][qid].add(cover_mode)

                if is_multi_kc_token:
                    covered_multi_kc_questions[kc].add(qid)
                else:
                    covered_single_kc_questions[kc].add(qid)

                if uid != "":
                    covered_students[kc].add(uid)

                if rep_val > 0:
                    repeat_sum[kc] += 1

    rows = []
    detail_rows = []

    for kc, n_attempt in attempts.items():
        if n_attempt < min_practices:
            continue

        mean_acc = correct_sum[kc] / n_attempt
        acc_std = pd.Series(correct_list[kc], dtype="float64").std(ddof=0)
        repeat_ratio = repeat_sum[kc] / n_attempt if n_attempt > 0 else 0.0

        all_qids = sorted(covered_questions[kc], key=numeric_sort_key)
        single_qids = sorted(covered_single_kc_questions[kc], key=numeric_sort_key)
        multi_qids = sorted(covered_multi_kc_questions[kc], key=numeric_sort_key)

        rows.append(
            {
                "kc": kc,
                "practice_count": n_attempt,
                "mean_acc": mean_acc,
                "acc_std": float(acc_std) if pd.notna(acc_std) else 0.0,
                "covered_question_count": len(all_qids),
                "covered_single_kc_question_count": len(single_qids),
                "covered_multi_kc_question_count": len(multi_qids),
                "covered_question_ids": json.dumps(all_qids, ensure_ascii=False),
                "covered_single_kc_question_ids": json.dumps(single_qids, ensure_ascii=False),
                "covered_multi_kc_question_ids": json.dumps(multi_qids, ensure_ascii=False),
                "covered_student_count": len(covered_students[kc]),
                "repeat_practice_count": repeat_sum[kc],
                "repeat_ratio": repeat_ratio,
            }
        )

        for qid in all_qids:
            modes = sorted(question_cover_modes[kc][qid])
            if modes == ["multi"]:
                cover_type = "multi"
            elif modes == ["single"]:
                cover_type = "single"
            else:
                cover_type = "both"

            detail_rows.append(
                {
                    "kc": kc,
                    "question_id": qid,
                    "cover_type": cover_type,
                }
            )

    stats_df = pd.DataFrame(rows)

    if not stats_df.empty:
        stats_df = stats_df.sort_values(
            by=["practice_count", "mean_acc"],
            ascending=[False, False],
        ).reset_index(drop=True)

    detail_df = pd.DataFrame(detail_rows)

    if not detail_df.empty:
        detail_df = detail_df.sort_values(
            by=["kc", "question_id"],
            key=lambda s: s.map(lambda x: numeric_sort_key(x)),
        ).reset_index(drop=True)

    stats_df.attrs["missing_triples"] = missing_triples
    stats_df.attrs["invalid_responses"] = invalid_responses
    stats_df.attrs["packed_concept_tokens"] = packed_concept_tokens
    stats_df.attrs["unmapped_question_tokens"] = sorted(unmapped_question_tokens, key=numeric_sort_key)
    stats_df.attrs["unmapped_concept_tokens"] = sorted(unmapped_concept_tokens, key=numeric_sort_key)

    return stats_df, detail_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze KC-level statistics from train set only, with original ids restored from keyid2idx."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="yousician_fmkc",
        help="Dataset key in configs/data_config.json.",
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default="",
        help="Optional manual input csv path. If set, overrides dataset_name config path.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Stats output csv path. Default: <dataset dpath>/kc_train_stats_original_ids.csv.",
    )

    parser.add_argument(
        "--covered-questions-output-csv",
        type=str,
        default="",
        help="Long-format output csv. Each row is one KC-question coverage relation.",
    )

    parser.add_argument(
        "--restored-train-csv",
        type=str,
        default="",
        help="Optional debug output: train rows with questions/concepts/uid restored to original ids.",
    )

    parser.add_argument(
        "--keyid2idx",
        type=str,
        default="",
        help=(
            "Path to keyid2idx.json. "
            "If omitted, auto-search <dataset_dir>/keyid2idx.json "
            "(and one-level parent fallback for question_level layouts)."
        ),
    )

    parser.add_argument(
        "--data_config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "data_config.json"),
        help="Path to data_config.json used with --dataset_name.",
    )

    parser.add_argument(
        "--valid-fold",
        type=int,
        default=0,
        help="Validation fold to exclude from train set when --train-folds is not provided.",
    )

    parser.add_argument(
        "--train-folds",
        type=str,
        default="",
        help="Optional explicit train folds, comma-separated. Example: 1,2,3,4.",
    )

    parser.add_argument(
        "--pad-val",
        type=int,
        default=-1,
        help="Pad value used in sequences.",
    )

    parser.add_argument(
        "--min-practices",
        type=int,
        default=1,
        help="Filter out KC with practice_count < min_practices.",
    )

    parser.add_argument(
        "--show-topk",
        type=int,
        default=20,
        help="Print top-k KC rows by practice_count.",
    )

    args = parser.parse_args()

    input_csv, output_csv, covered_questions_output_csv, restored_train_csv = resolve_dataset_paths(
        dataset_name=args.dataset_name,
        data_config_path=Path(args.data_config),
        input_csv_arg=args.input_csv,
        output_csv_arg=args.output_csv,
        covered_questions_output_csv_arg=args.covered_questions_output_csv,
        restored_train_csv_arg=args.restored_train_csv,
    )

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    if args.keyid2idx.strip():
        keyid2idx_path = Path(args.keyid2idx)
    else:
        auto_candidates = [
            input_csv.parent / "keyid2idx.json",
            input_csv.parent.parent / "keyid2idx.json",
        ]
        keyid2idx_path = next((p for p in auto_candidates if p.exists()), None)

    idx2key_maps = build_idx2key_maps(keyid2idx_path)

    df = pd.read_csv(input_csv)

    required_cols = {"fold", "questions", "concepts", "responses"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing columns in input file: {sorted(missing_cols)}")

    train_folds = choose_train_folds(
        df=df,
        valid_fold=args.valid_fold,
        train_folds_text=args.train_folds,
    )

    df_train = df[df["fold"].isin(train_folds)].copy()

    if restored_train_csv is not None:
        restored_df = restore_train_dataframe_ids(
            df_train=df_train,
            idx2key_maps=idx2key_maps,
            pad_val=args.pad_val,
        )
        restored_train_csv.parent.mkdir(parents=True, exist_ok=True)
        restored_df.to_csv(restored_train_csv, index=False)

    stats_df, detail_df = analyze_kc_stats(
        df_train=df_train,
        idx2key_maps=idx2key_maps,
        pad_val=args.pad_val,
        min_practices=args.min_practices,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_csv, index=False)

    covered_questions_output_csv.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(covered_questions_output_csv, index=False)

    unmapped_question_tokens = stats_df.attrs.get("unmapped_question_tokens", [])
    unmapped_concept_tokens = stats_df.attrs.get("unmapped_concept_tokens", [])

    print(f"dataset_name: {args.dataset_name}")
    print(f"input_csv: {input_csv.resolve()}")
    print(f"keyid2idx: {keyid2idx_path.resolve() if keyid2idx_path else None}")
    print(f"all_folds: {sorted(set(df['fold'].dropna().astype(int).tolist()))}")
    print(f"train_folds_used: {sorted(train_folds)}")
    print(f"train_rows_used: {len(df_train)}")
    print(f"unique_original_discrete_kc: {len(stats_df)}")
    print(f"ignored_misaligned_sequences: {stats_df.attrs.get('missing_triples', 0)}")
    print(f"ignored_invalid_responses: {stats_df.attrs.get('invalid_responses', 0)}")
    print(f"packed_concept_tokens_after_restore: {stats_df.attrs.get('packed_concept_tokens', 0)}")
    print(f"unmapped_question_tokens_count: {len(unmapped_question_tokens)}")
    print(f"unmapped_concept_tokens_count: {len(unmapped_concept_tokens)}")

    if unmapped_question_tokens:
        print(f"unmapped_question_tokens_preview: {unmapped_question_tokens[:20]}")

    if unmapped_concept_tokens:
        print(f"unmapped_concept_tokens_preview: {unmapped_concept_tokens[:20]}")

    print(f"saved_stats_csv: {output_csv.resolve()}")
    print(f"saved_covered_questions_csv: {covered_questions_output_csv.resolve()}")

    if restored_train_csv is not None:
        print(f"saved_restored_train_csv: {restored_train_csv.resolve()}")

    if len(stats_df) > 0 and args.show_topk > 0:
        topk = min(args.show_topk, len(stats_df))
        print(f"\nTop-{topk} original discrete KC by practice_count:")
        print(stats_df.head(topk).to_string(index=False))


if __name__ == "__main__":
    # 拿到的是原始Id，即键
    # Example 1: use dataset config
    # python analyze_kc_train_stats.py --dataset_name {dataset_name} --valid-fold 0

    # Example 2: use manual csv path
    # python examples/analyze_kc_train_stats.py \
    #   --input_csv path/to/train_valid_quelevel.csv \
    #   --valid-fold 0 \
    #   --keyid2idx path/to/keyid2idx.json \
    #   --output_csv path/to/kc_train_stats_original_ids.csv \
    #   --covered-questions-output-csv path/to/kc_train_covered_questions_original_ids.csv \
    #   --restored-train-csv path/to/restored_train_original_ids.csv

    main()