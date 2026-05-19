from .candidate_filter import filter_candidates
from .residual_analyzer import analyze_valid_question_residuals


def run_kc_diagnosis_pipeline(
    save_dir: str,
    batch_size: int = 256,
    output_dir: str = "",
    signed_mode: str = "true_minus_pred",
    save_interactions: bool = False,
    min_pair_questions: int = 2,
    data_config_path: str = "../configs/data_config.json",
    short_prefix: bool = True,
    min_questions: int = 10,
    min_valid_points: int = 100,
    top_ratio: float = 0.1,
    multi_kc_threshold: float = 0.4,
    encoding: str = "utf-8-sig",
):
    """
    One-shot pipeline:
      1. Compute validation residual diagnosis tables.
      2. Filter over-coarse and combination-sensitive KC candidates.

    output_dir behavior:
      - empty: <save_dir>/d
      - relative: <save_dir>/<output_dir>
      - absolute: output_dir
    """
    residual_outputs = analyze_valid_question_residuals(
        save_dir=save_dir,
        batch_size=batch_size,
        output_dir=output_dir,
        signed_mode=signed_mode,
        save_interactions=save_interactions,
        min_pair_questions=min_pair_questions,
        data_config_path=data_config_path,
        short_prefix=short_prefix,
    )

    kc_question_csv = residual_outputs["kc_question_heterogeneity"]
    resolved_output_dir = residual_outputs["output_dir"]

    candidate_outputs = filter_candidates(
        input_csv=kc_question_csv,
        output_dir=resolved_output_dir,
        min_questions=min_questions,
        min_valid_points=min_valid_points,
        top_ratio=top_ratio,
        multi_kc_threshold=multi_kc_threshold,
        encoding=encoding,
    )

    return {
        "residual_outputs": residual_outputs,
        "candidate_outputs": candidate_outputs,
    }
