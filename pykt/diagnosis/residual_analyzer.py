import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch

from pykt.datasets import init_dataset4train
from pykt.models import load_model

from .io_utils import (
    clean_model_config,
    join_ids,
    join_names,
    load_id_name_maps,
    resolve_data_config_path,
    resolve_diagnosis_output_dir,
    safe_id_sort_key,
)
from .prediction_adapters import get_valid_predictions
from .stats import finalize_stat, new_stat, update_stat, weighted_mean, weighted_var


def analyze_valid_question_residuals(
    save_dir,
    batch_size=256,
    output_dir=None,
    signed_mode="true_minus_pred",
    save_interactions=False,
    min_pair_questions=2,
    data_config_path="../configs/data_config.json",
    short_prefix=True,
):
    """
    Compute validation residual diagnosis tables for a trained pyKT run.

    Parameters
    ----------
    save_dir:
        Trained run directory containing config.json and model checkpoint.
    batch_size:
        Validation batch size.
    output_dir:
        If empty, output is written to <save_dir>/d.
        If relative, output is written to <save_dir>/<output_dir>.
        If absolute, output is written there directly.
    signed_mode:
        true_minus_pred: positive residual means model under-predicts correctness.
        pred_minus_true: positive residual means model over-predicts correctness.
    save_interactions:
        Whether to save raw interaction-level residuals. Can be large.
    min_pair_questions:
        Minimum number of questions for a KC pair to appear in pair summary.
    data_config_path:
        Default ../configs/data_config.json.
        If not found, fallback to configs/data_config.json.
    short_prefix:
        If True, output files use prefix "valid" to avoid Windows path-too-long errors.
        If False, output files use dataset_model_emb_fold prefix.

    Returns
    -------
    dict of output file paths.
    """
    save_dir = Path(save_dir).resolve()
    cfg_path = save_dir / "config.json"

    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in save_dir: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        run_cfg = json.load(f)

    params = run_cfg["params"]
    model_name = params["model_name"]
    dataset_name = params["dataset_name"]
    emb_type = params["emb_type"]
    valid_fold = int(params["fold"])

    data_config_path = resolve_data_config_path(data_config_path)
    with data_config_path.open("r", encoding="utf-8") as f:
        full_data_config = json.load(f)

    data_cfg = full_data_config[dataset_name]
    data_cfg["dataset_name"] = dataset_name

    model_config = clean_model_config(model_name, run_cfg["model_config"])

    # Some pyKT models require seq_len in constructor config.
    if model_name in [
        "saint",
        "saint++",
        "sakt",
        "atdkt",
        "simplekt",
        "stablekt",
        "datakt",
        "folibikt",
    ]:
        model_config["seq_len"] = run_cfg["train_config"]["seq_len"]

    _, valid_loader, *_ = init_dataset4train(
        dataset_name=dataset_name,
        model_name=model_name,
        data_config=full_data_config,
        i=valid_fold,
        batch_size=batch_size,
    )

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    model = load_model(model_name, model_config, data_cfg, emb_type, str(save_dir))
    model.to(device)
    model.eval()

    concept_name_map, question_name_map = load_id_name_maps(data_cfg)

    question_stats = defaultdict(new_stat)
    kc_stats = defaultdict(new_stat)
    qkc_stats = defaultdict(new_stat)
    q_to_kcs = defaultdict(set)
    interaction_rows = []

    with torch.no_grad():
        for dcur in valid_loader:
            q_arr, kc_arr, true_arr, pred_arr = get_valid_predictions(
                model=model,
                model_name=model_name,
                dcur=dcur,
                device=device,
            )

            for qid, kc, true_value, pred_value in zip(q_arr, kc_arr, true_arr, pred_arr):
                qid = int(qid)
                kc = int(kc)
                t = float(true_value)
                p = float(pred_value)

                if signed_mode == "pred_minus_true":
                    residual = p - t
                elif signed_mode == "true_minus_pred":
                    residual = t - p
                else:
                    raise ValueError(
                        "signed_mode must be one of: true_minus_pred, pred_minus_true"
                    )

                update_stat(question_stats[qid], t, p, residual)
                update_stat(kc_stats[kc], t, p, residual)
                update_stat(qkc_stats[(qid, kc)], t, p, residual)

                q_to_kcs[qid].add(kc)

                if save_interactions:
                    interaction_rows.append(
                        {
                            "question_id": qid,
                            "question_name": question_name_map.get(qid, ""),
                            "kc_id": kc,
                            "kc_name": concept_name_map.get(kc, ""),
                            "true": t,
                            "pred": p,
                            "signed_residual": residual,
                            "abs_residual": abs(residual),
                            "sq_residual": residual * residual,
                        }
                    )

    # ------------------------------------------------------------
    # 1. Question-level residual table
    # ------------------------------------------------------------
    question_rows = []
    for qid, st in question_stats.items():
        fs = finalize_stat(st)
        kcs = q_to_kcs.get(qid, set())

        question_rows.append(
            {
                "question_id": qid,
                "question_name": question_name_map.get(qid, ""),
                "kc_ids": join_ids(kcs),
                "kc_names": join_names(kcs, concept_name_map),
                "kc_count": len(kcs),
                "n_valid_points": fs["n"],
                "mean_true": fs["mean_true"],
                "mean_pred": fs["mean_pred"],
                "mean_signed_residual": fs["mean_signed_residual"],
                "abs_mean_signed_residual": abs(fs["mean_signed_residual"]),
                "mean_abs_residual": fs["mean_abs_residual"],
                "rmse": fs["rmse"],
                "residual_var": fs["residual_var"],
                "residual_std": fs["residual_std"],
                "residual_se": fs["residual_se"],
                "residual_ci95_low": fs["residual_ci95_low"],
                "residual_ci95_high": fs["residual_ci95_high"],
                "pred_std": fs["pred_std"],
                "true_std": fs["true_std"],
            }
        )

    question_df = pd.DataFrame(question_rows)
    if not question_df.empty:
        question_df = question_df.sort_values(
            ["abs_mean_signed_residual", "n_valid_points"],
            ascending=[False, False],
        ).reset_index(drop=True)

    # ------------------------------------------------------------
    # 2. KC-level residual table
    # ------------------------------------------------------------
    kc_rows = []
    for kc, st in kc_stats.items():
        fs = finalize_stat(st)
        kc_rows.append(
            {
                "kc_id": kc,
                "kc_name": concept_name_map.get(kc, ""),
                "n_valid_points": fs["n"],
                "mean_true": fs["mean_true"],
                "mean_pred": fs["mean_pred"],
                "mean_signed_residual": fs["mean_signed_residual"],
                "abs_mean_signed_residual": abs(fs["mean_signed_residual"]),
                "mean_abs_residual": fs["mean_abs_residual"],
                "rmse": fs["rmse"],
                "residual_var": fs["residual_var"],
                "residual_std": fs["residual_std"],
                "residual_se": fs["residual_se"],
                "residual_ci95_low": fs["residual_ci95_low"],
                "residual_ci95_high": fs["residual_ci95_high"],
            }
        )

    kc_df = pd.DataFrame(kc_rows)
    if not kc_df.empty:
        kc_df = kc_df.sort_values(
            ["n_valid_points", "abs_mean_signed_residual"],
            ascending=[False, False],
        ).reset_index(drop=True)

    # ------------------------------------------------------------
    # 3. Question-KC edge residual table
    # ------------------------------------------------------------
    qkc_rows = []
    for (qid, kc), st in qkc_stats.items():
        fs = finalize_stat(st)
        all_kcs = q_to_kcs.get(qid, set())
        qkc_rows.append(
            {
                "question_id": qid,
                "question_name": question_name_map.get(qid, ""),
                "kc_id": kc,
                "kc_name": concept_name_map.get(kc, ""),
                "question_all_kc_ids": join_ids(all_kcs),
                "question_all_kc_names": join_names(all_kcs, concept_name_map),
                "question_kc_count": len(all_kcs),
                "n_valid_points": fs["n"],
                "mean_true": fs["mean_true"],
                "mean_pred": fs["mean_pred"],
                "mean_signed_residual": fs["mean_signed_residual"],
                "abs_mean_signed_residual": abs(fs["mean_signed_residual"]),
                "mean_abs_residual": fs["mean_abs_residual"],
                "rmse": fs["rmse"],
                "residual_var": fs["residual_var"],
                "residual_std": fs["residual_std"],
            }
        )

    qkc_df = pd.DataFrame(qkc_rows)
    if not qkc_df.empty:
        qkc_df = qkc_df.sort_values(
            ["kc_id", "abs_mean_signed_residual"],
            ascending=[True, False],
        ).reset_index(drop=True)

    # ------------------------------------------------------------
    # 4. KC internal question-level heterogeneity
    # ------------------------------------------------------------
    kc_to_q_items = defaultdict(list)

    for _, row in question_df.iterrows():
        qid = int(row["question_id"])
        kcs = q_to_kcs.get(qid, set())
        w = float(row["n_valid_points"])

        for kc in kcs:
            kc_to_q_items[kc].append(
                {
                    "question_id": qid,
                    "weight": w,
                    "q_mean_true": float(row["mean_true"]),
                    "q_mean_pred": float(row["mean_pred"]),
                    "q_mean_signed_residual": float(row["mean_signed_residual"]),
                    "q_abs_mean_signed_residual": float(row["abs_mean_signed_residual"]),
                    "q_rmse": float(row["rmse"]),
                    "q_residual_var": float(row["residual_var"]),
                    "q_kc_count": int(row["kc_count"]),
                }
            )

    kc_q_rows = []
    for kc, items in kc_to_q_items.items():
        weights = [it["weight"] for it in items]

        q_residuals = [it["q_mean_signed_residual"] for it in items]
        q_abs_residuals = [it["q_abs_mean_signed_residual"] for it in items]
        q_accs = [it["q_mean_true"] for it in items]
        q_preds = [it["q_mean_pred"] for it in items]
        q_rmses = [it["q_rmse"] for it in items]
        q_kc_counts = [it["q_kc_count"] for it in items]

        q_res_var_w = weighted_var(q_residuals, weights)
        q_acc_var_w = weighted_var(q_accs, weights)
        q_pred_var_w = weighted_var(q_preds, weights)

        kc_q_rows.append(
            {
                "kc_id": kc,
                "kc_name": concept_name_map.get(kc, ""),
                "n_questions": len(items),
                "n_valid_points_total": sum(weights),
                "weighted_mean_question_residual": weighted_mean(q_residuals, weights),
                "weighted_mean_abs_question_residual": weighted_mean(q_abs_residuals, weights),
                "weighted_question_residual_var": q_res_var_w,
                "weighted_question_residual_std": math.sqrt(q_res_var_w)
                if not math.isnan(q_res_var_w)
                else float("nan"),
                "weighted_question_acc_var": q_acc_var_w,
                "weighted_question_acc_std": math.sqrt(q_acc_var_w)
                if not math.isnan(q_acc_var_w)
                else float("nan"),
                "weighted_question_pred_var": q_pred_var_w,
                "weighted_question_rmse": weighted_mean(q_rmses, weights),
                "max_abs_question_residual": max(q_abs_residuals)
                if q_abs_residuals
                else float("nan"),
                "mean_question_kc_count": sum(q_kc_counts) / len(q_kc_counts)
                if q_kc_counts
                else float("nan"),
                "multi_kc_question_ratio": sum(1 for x in q_kc_counts if x > 1)
                / len(q_kc_counts)
                if q_kc_counts
                else float("nan"),
            }
        )

    kc_question_df = pd.DataFrame(kc_q_rows)
    if not kc_question_df.empty:
        kc_question_df = kc_question_df.sort_values(
            ["weighted_question_residual_var", "n_questions"],
            ascending=[False, False],
        ).reset_index(drop=True)

    # ------------------------------------------------------------
    # 5. KC-pair combination residual summary
    # ------------------------------------------------------------
    pair_stats = defaultdict(new_stat)
    pair_question_count = defaultdict(int)

    for _, row in question_df.iterrows():
        qid = int(row["question_id"])
        kcs = sorted(q_to_kcs.get(qid, set()), key=safe_id_sort_key)

        if len(kcs) < 2:
            continue

        t = float(row["mean_true"])
        p = float(row["mean_pred"])
        residual = float(row["mean_signed_residual"])
        n_points = int(row["n_valid_points"])

        for a, b in combinations(kcs, 2):
            pair = (int(a), int(b))
            pair_question_count[pair] += 1

            st = pair_stats[pair]
            st["n"] += n_points
            st["sum_true"] += t * n_points
            st["sum_true2"] += t * t * n_points
            st["sum_pred"] += p * n_points
            st["sum_pred2"] += p * p * n_points
            st["sum_res"] += residual * n_points
            st["sum_abs_res"] += abs(residual) * n_points
            st["sum_sq_res"] += residual * residual * n_points
            st["sum_res2"] += residual * residual * n_points

    pair_rows = []
    for (a, b), st in pair_stats.items():
        n_q = pair_question_count[(a, b)]
        if n_q < min_pair_questions:
            continue

        fs = finalize_stat(st)
        pair_rows.append(
            {
                "kc_a": a,
                "kc_a_name": concept_name_map.get(a, ""),
                "kc_b": b,
                "kc_b_name": concept_name_map.get(b, ""),
                "n_questions": n_q,
                "n_valid_points_total": fs["n"],
                "mean_true": fs["mean_true"],
                "mean_pred": fs["mean_pred"],
                "mean_signed_residual": fs["mean_signed_residual"],
                "abs_mean_signed_residual": abs(fs["mean_signed_residual"]),
                "mean_abs_residual": fs["mean_abs_residual"],
                "rmse": fs["rmse"],
                "residual_var": fs["residual_var"],
                "residual_std": fs["residual_std"],
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(
            ["abs_mean_signed_residual", "n_questions"],
            ascending=[False, False],
        ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    output_dir = resolve_diagnosis_output_dir(save_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if short_prefix:
        prefix = "valid"
    else:
        prefix = f"{dataset_name}_{model_name}_{emb_type}_fold{valid_fold}"

    question_path = output_dir / f"{prefix}_question_level_residuals.csv"
    kc_path = output_dir / f"{prefix}_kc_level_residuals.csv"
    qkc_path = output_dir / f"{prefix}_question_kc_edge_residuals.csv"
    kc_question_path = output_dir / f"{prefix}_kc_question_heterogeneity.csv"
    pair_path = output_dir / f"{prefix}_kc_pair_residuals.csv"

    question_df.to_csv(question_path, index=False, encoding="utf-8-sig")
    kc_df.to_csv(kc_path, index=False, encoding="utf-8-sig")
    qkc_df.to_csv(qkc_path, index=False, encoding="utf-8-sig")
    kc_question_df.to_csv(kc_question_path, index=False, encoding="utf-8-sig")
    pair_df.to_csv(pair_path, index=False, encoding="utf-8-sig")

    interaction_path = None
    if save_interactions:
        interaction_df = pd.DataFrame(interaction_rows)
        interaction_path = output_dir / f"{prefix}_interaction_residuals.csv"
        interaction_df.to_csv(interaction_path, index=False, encoding="utf-8-sig")

    print("\n=== VALID question-level residual diagnosis done ===")
    print(f"save_dir: {save_dir}")
    print(f"dataset_name: {dataset_name}")
    print(f"model_name: {model_name}")
    print(f"emb_type: {emb_type}")
    print(f"valid_fold: {valid_fold}")
    print(f"signed_mode: {signed_mode}")
    print(f"data_config_path: {data_config_path}")
    print(f"output_dir: {output_dir}")

    print("\nSaved files:")
    print(f"1. question-level residuals:      {question_path}")
    print(f"2. KC-level residuals:            {kc_path}")
    print(f"3. question-KC edge residuals:    {qkc_path}")
    print(f"4. KC question heterogeneity:     {kc_question_path}")
    print(f"5. KC-pair residuals:             {pair_path}")
    if interaction_path is not None:
        print(f"6. raw interaction residuals:     {interaction_path}")

    if not question_df.empty:
        print("\nTop 20 questions by abs mean signed residual:")
        cols = [
            "question_id",
            "question_name",
            "kc_ids",
            "kc_count",
            "n_valid_points",
            "mean_true",
            "mean_pred",
            "mean_signed_residual",
            "abs_mean_signed_residual",
            "rmse",
        ]
        available_cols = [c for c in cols if c in question_df.columns]
        print(question_df.head(20)[available_cols].to_string(index=False))

    if not kc_question_df.empty:
        print("\nTop 20 KCs by within-KC question residual variance:")
        cols = [
            "kc_id",
            "kc_name",
            "n_questions",
            "n_valid_points_total",
            "weighted_question_residual_var",
            "weighted_question_residual_std",
            "multi_kc_question_ratio",
            "weighted_mean_abs_question_residual",
        ]
        available_cols = [c for c in cols if c in kc_question_df.columns]
        print(kc_question_df.head(20)[available_cols].to_string(index=False))

    return {
        "output_dir": str(output_dir),
        "question_level_residuals": str(question_path),
        "kc_level_residuals": str(kc_path),
        "question_kc_edge_residuals": str(qkc_path),
        "kc_question_heterogeneity": str(kc_question_path),
        "kc_pair_residuals": str(pair_path),
        "interaction_residuals": str(interaction_path) if interaction_path else None,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "emb_type": emb_type,
        "fold": valid_fold,
    }
