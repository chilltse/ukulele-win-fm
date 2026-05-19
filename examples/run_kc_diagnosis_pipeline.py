import argparse

from pykt.diagnosis.pipeline import run_kc_diagnosis_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run full KC diagnosis pipeline: residual analysis + candidate filtering."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Trained run directory containing config.json and model checkpoint.",
    )
    parser.add_argument("--bz", type=int, default=32)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help=(
            "Optional. If omitted, use <save_dir>/d. "
            "If relative, use <save_dir>/<output_dir>. "
            "If absolute, use it directly."
        ),
    )
    parser.add_argument(
        "--signed_mode",
        type=str,
        default="true_minus_pred",
        choices=["true_minus_pred", "pred_minus_true"],
    )
    parser.add_argument("--save_interactions", action="store_true")
    parser.add_argument("--min_pair_questions", type=int, default=2)
    parser.add_argument(
        "--data_config_path",
        type=str,
        default="../configs/data_config.json",
        help=(
            "Path to data_config.json. Default: ../configs/data_config.json. "
            "If not found, fallback to configs/data_config.json."
        ),
    )
    parser.add_argument(
        "--long_prefix",
        action="store_true",
        help="Use long dataset_model_emb_fold prefix instead of short 'valid' prefix.",
    )

    parser.add_argument("--min_questions", type=int, default=8)
    parser.add_argument("--min_valid_points", type=int, default=800)
    parser.add_argument("--top_ratio", type=float, default=0.15)
    parser.add_argument("--multi_kc_threshold", type=float, default=0.4)
    parser.add_argument("--encoding", type=str, default="utf-8-sig")

    args = parser.parse_args()

    run_kc_diagnosis_pipeline(
        save_dir=args.save_dir,
        batch_size=args.bz,
        output_dir=args.output_dir,
        signed_mode=args.signed_mode,
        save_interactions=args.save_interactions,
        min_pair_questions=args.min_pair_questions,
        data_config_path=args.data_config_path,
        short_prefix=not args.long_prefix,
        min_questions=args.min_questions,
        min_valid_points=args.min_valid_points,
        top_ratio=args.top_ratio,
        multi_kc_threshold=args.multi_kc_threshold,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    # "saved_model\xes3g5m_dkt+_qid_f0_s42_c24dee95fa_551aa3ec-6240-46f7-b884-9aecba6d2426"
    main()
