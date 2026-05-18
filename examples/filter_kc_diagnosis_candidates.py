import argparse
from pathlib import Path

import pandas as pd


# ============================================================
# Global configuration
# 修改超参数只需要改这里
# ============================================================
CONFIG = {
    # input / output
    "input_csv": "",
    "output_dir": "",

    # candidate filtering thresholds
    "min_questions": 10,
    "min_valid_points": 100,
    "top_ratio": 0.05,
    "multi_kc_threshold": 0.3,

    # output
    "encoding": "utf-8-sig",
}


REQUIRED_COLUMNS = [
    "kc_id",
    "kc_name",
    "n_questions",
    "n_valid_points_total",
    "weighted_question_residual_var",
    "weighted_question_residual_std",
    "multi_kc_question_ratio",
]


def check_required_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in input CSV: "
            + ", ".join(missing)
            + "\nPlease make sure the input is *_kc_question_heterogeneity.csv."
        )


def add_diagnosis_flags(
    df: pd.DataFrame,
    min_questions: int,
    min_valid_points: int,
    top_ratio: float,
    multi_kc_threshold: float,
) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "n_questions",
        "n_valid_points_total",
        "weighted_question_residual_var",
        "weighted_question_residual_std",
        "multi_kc_question_ratio",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_enough_support"] = (
        (df["n_questions"] >= min_questions)
        & (df["n_valid_points_total"] >= min_valid_points)
    )

    supported = df[df["is_enough_support"]].copy()

    if supported.empty:
        df["residual_var_rank_desc"] = pd.NA
        df["residual_var_percentile_desc"] = pd.NA
        df["is_top_residual_var"] = False
        df["is_over_coarse_candidate"] = False
        df["is_combination_sensitive_candidate"] = False
        return df

    supported["residual_var_rank_desc"] = supported[
        "weighted_question_residual_var"
    ].rank(method="min", ascending=False)

    n_supported = len(supported)

    supported["residual_var_percentile_desc"] = (
        supported["residual_var_rank_desc"] / n_supported
    )

    supported["is_top_residual_var"] = (
        supported["residual_var_percentile_desc"] <= top_ratio
    )

    df = df.merge(
        supported[
            [
                "kc_id",
                "residual_var_rank_desc",
                "residual_var_percentile_desc",
                "is_top_residual_var",
            ]
        ],
        on="kc_id",
        how="left",
    )

    df["is_top_residual_var"] = df["is_top_residual_var"].fillna(False)

    df["is_over_coarse_candidate"] = (
        df["is_enough_support"]
        & df["is_top_residual_var"]
        & (df["multi_kc_question_ratio"] < multi_kc_threshold)
    )

    df["is_combination_sensitive_candidate"] = (
        df["is_enough_support"]
        & df["is_top_residual_var"]
        & (df["multi_kc_question_ratio"] >= multi_kc_threshold)
    )

    return df


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    first_cols = [
        "kc_id",
        "kc_name",
        "n_questions",
        "n_valid_points_total",
        "weighted_question_residual_var",
        "weighted_question_residual_std",
        "weighted_mean_question_residual",
        "weighted_mean_abs_question_residual",
        "weighted_question_acc_var",
        "weighted_question_acc_std",
        "weighted_question_pred_var",
        "weighted_question_rmse",
        "max_abs_question_residual",
        "mean_question_kc_count",
        "multi_kc_question_ratio",
        "is_enough_support",
        "residual_var_rank_desc",
        "residual_var_percentile_desc",
        "is_top_residual_var",
        "is_over_coarse_candidate",
        "is_combination_sensitive_candidate",
    ]

    existing_first_cols = [c for c in first_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_first_cols]

    return df[existing_first_cols + remaining_cols]


def filter_candidates(
    input_csv: str,
    output_dir: str,
    min_questions: int,
    min_valid_points: int,
    top_ratio: float,
    multi_kc_threshold: float,
    encoding: str,
):
    input_path = Path(input_csv).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    check_required_columns(df)

    result_df = add_diagnosis_flags(
        df=df,
        min_questions=min_questions,
        min_valid_points=min_valid_points,
        top_ratio=top_ratio,
        multi_kc_threshold=multi_kc_threshold,
    )

    result_df = select_output_columns(result_df)

    if output_dir is None or str(output_dir).strip() == "":
        output_path = input_path.parent
    else:
        output_path = Path(output_dir).resolve()

    output_path.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem

    all_flagged_path = output_path / f"{stem}_with_candidate_flags.csv"
    over_coarse_path = output_path / f"{stem}_over_coarse_candidates.csv"
    combination_sensitive_path = output_path / f"{stem}_combination_sensitive_candidates.csv"
    summary_path = output_path / f"{stem}_candidate_summary.txt"

    over_coarse_df = result_df[result_df["is_over_coarse_candidate"]].copy()
    combination_sensitive_df = result_df[
        result_df["is_combination_sensitive_candidate"]
    ].copy()

    sort_cols = [
        "weighted_question_residual_var",
        "n_questions",
        "n_valid_points_total",
    ]

    over_coarse_df = over_coarse_df.sort_values(
        sort_cols,
        ascending=[False, False, False],
    ).reset_index(drop=True)

    combination_sensitive_df = combination_sensitive_df.sort_values(
        sort_cols,
        ascending=[False, False, False],
    ).reset_index(drop=True)

    result_df = result_df.sort_values(
        [
            "is_top_residual_var",
            "weighted_question_residual_var",
            "n_questions",
            "n_valid_points_total",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    result_df.to_csv(all_flagged_path, index=False, encoding=encoding)
    over_coarse_df.to_csv(over_coarse_path, index=False, encoding=encoding)
    combination_sensitive_df.to_csv(
        combination_sensitive_path,
        index=False,
        encoding=encoding,
    )

    n_total = len(result_df)
    n_supported = int(result_df["is_enough_support"].sum())
    n_top = int(result_df["is_top_residual_var"].sum())
    n_over = len(over_coarse_df)
    n_combo = len(combination_sensitive_df)

    summary = f"""
KC diagnosis candidate filtering summary
=======================================

Input CSV:
{input_path}

Output directory:
{output_path}

Rules
-----
Enough support:
    n_questions >= {min_questions}
    n_valid_points_total >= {min_valid_points}

High residual heterogeneity:
    weighted_question_residual_var in top {top_ratio:.2%}
    among enough-support KCs only

Over-coarse candidate:
    enough support
    high residual heterogeneity
    multi_kc_question_ratio < {multi_kc_threshold}

Combination-sensitive candidate:
    enough support
    high residual heterogeneity
    multi_kc_question_ratio >= {multi_kc_threshold}

Counts
------
Total KCs: {n_total}
Enough-support KCs: {n_supported}
Top residual-var KCs: {n_top}
Over-coarse candidates: {n_over}
Combination-sensitive candidates: {n_combo}

Output files
------------
All KCs with flags:
{all_flagged_path}

Over-coarse candidates:
{over_coarse_path}

Combination-sensitive candidates:
{combination_sensitive_path}
""".strip()

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)

    if n_over > 0:
        print("\nTop over-coarse candidates:")
        print(
            over_coarse_df[
                [
                    "kc_id",
                    "kc_name",
                    "n_questions",
                    "n_valid_points_total",
                    "weighted_question_residual_var",
                    "weighted_question_residual_std",
                    "multi_kc_question_ratio",
                    "residual_var_rank_desc",
                    "residual_var_percentile_desc",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    if n_combo > 0:
        print("\nTop combination-sensitive candidates:")
        print(
            combination_sensitive_df[
                [
                    "kc_id",
                    "kc_name",
                    "n_questions",
                    "n_valid_points_total",
                    "weighted_question_residual_var",
                    "weighted_question_residual_std",
                    "multi_kc_question_ratio",
                    "residual_var_rank_desc",
                    "residual_var_percentile_desc",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Filter over-coarse and combination-sensitive KC candidates "
            "from *_kc_question_heterogeneity.csv."
        )
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default=CONFIG["input_csv"],
        help="Path to *_kc_question_heterogeneity.csv.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=CONFIG["output_dir"],
        help="Output directory. Default: same directory as input_csv.",
    )

    parser.add_argument(
        "--min_questions",
        type=int,
        default=CONFIG["min_questions"],
        help="Minimum number of questions covered by a KC.",
    )

    parser.add_argument(
        "--min_valid_points",
        type=int,
        default=CONFIG["min_valid_points"],
        help="Minimum total validation interactions for a KC.",
    )

    parser.add_argument(
        "--top_ratio",
        type=float,
        default=CONFIG["top_ratio"],
        help="Top ratio by weighted_question_residual_var among supported KCs.",
    )

    parser.add_argument(
        "--multi_kc_threshold",
        type=float,
        default=CONFIG["multi_kc_threshold"],
        help=(
            "Threshold for multi_kc_question_ratio. "
            "Below this is over-coarse; above/equal is combination-sensitive."
        ),
    )

    parser.add_argument(
        "--encoding",
        type=str,
        default=CONFIG["encoding"],
        help="CSV output encoding.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.input_csv is None or str(args.input_csv).strip() == "":
        raise ValueError(
            "input_csv is empty. Please either set CONFIG['input_csv'] "
            "at the top of the script or pass --input_csv."
        )

    filter_candidates(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        min_questions=args.min_questions,
        min_valid_points=args.min_valid_points,
        top_ratio=args.top_ratio,
        multi_kc_threshold=args.multi_kc_threshold,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()