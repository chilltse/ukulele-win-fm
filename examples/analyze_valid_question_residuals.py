import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch
from torch.nn.functional import one_hot

from pykt.datasets import init_dataset4train
from pykt.models import load_model


device = "cpu" if not torch.cuda.is_available() else "cuda"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _to_number(x):
    s = str(x)
    return int(s) if s.isdigit() else s


def _safe_id_sort_key(x):
    s = str(x)
    if s.isdigit():
        return (0, int(s))
    return (1, s)


def _load_id_name_maps(data_config):
    """
    Read keyid2idx.json and invert:
        original_id -> internal_id
    into:
        internal_id -> original_id

    Returns:
        concept_name_map: dict[int, str]
        question_name_map: dict[int, str]
    """
    keyid2idx_path = Path(data_config["dpath"]) / "keyid2idx.json"
    concept_name_map = {}
    question_name_map = {}

    if not keyid2idx_path.exists():
        return concept_name_map, question_name_map

    try:
        with keyid2idx_path.open("r", encoding="utf-8") as f:
            keyid2idx = json.load(f)

        concept_map = keyid2idx.get("concepts", {})
        question_map = keyid2idx.get("questions", {})

        concept_name_map = {int(v): str(k) for k, v in concept_map.items()}
        question_name_map = {int(v): str(k) for k, v in question_map.items()}

    except Exception as e:
        print(f"[warn] failed to parse {keyid2idx_path}: {e}")

    return concept_name_map, question_name_map


def _clean_model_config(model_name, cfg):
    model_config = dict(cfg)

    # These are training/logging-only configs, not constructor args for most pyKT models.
    for k in [
        "use_wandb",
        "learning_rate",
        "add_uuid",
        "l2",
        "tree_pred_decay_lr_mult",
    ]:
        if k in model_config:
            del model_config[k]

    return model_config


def _new_stat():
    return {
        "n": 0,
        "sum_true": 0.0,
        "sum_true2": 0.0,
        "sum_pred": 0.0,
        "sum_pred2": 0.0,
        "sum_res": 0.0,
        "sum_abs_res": 0.0,
        "sum_sq_res": 0.0,
        "sum_res2": 0.0,
    }


def _update_stat(st, true_value, pred_value, residual):
    t = float(true_value)
    p = float(pred_value)
    r = float(residual)

    st["n"] += 1
    st["sum_true"] += t
    st["sum_true2"] += t * t
    st["sum_pred"] += p
    st["sum_pred2"] += p * p
    st["sum_res"] += r
    st["sum_abs_res"] += abs(r)
    st["sum_sq_res"] += r * r
    st["sum_res2"] += r * r


def _finalize_stat(st):
    n = st["n"]
    if n <= 0:
        return {
            "n": 0,
            "mean_true": float("nan"),
            "mean_pred": float("nan"),
            "mean_signed_residual": float("nan"),
            "mean_abs_residual": float("nan"),
            "rmse": float("nan"),
            "residual_var": float("nan"),
            "residual_std": float("nan"),
            "residual_se": float("nan"),
            "residual_ci95_low": float("nan"),
            "residual_ci95_high": float("nan"),
            "pred_var": float("nan"),
            "pred_std": float("nan"),
            "true_var": float("nan"),
            "true_std": float("nan"),
        }

    mean_true = st["sum_true"] / n
    mean_pred = st["sum_pred"] / n
    mean_res = st["sum_res"] / n
    mean_abs_res = st["sum_abs_res"] / n
    rmse = math.sqrt(st["sum_sq_res"] / n)

    residual_var = max(st["sum_res2"] / n - mean_res * mean_res, 0.0)
    residual_std = math.sqrt(residual_var)
    residual_se = residual_std / math.sqrt(n) if n > 1 else float("nan")

    pred_var = max(st["sum_pred2"] / n - mean_pred * mean_pred, 0.0)
    true_var = max(st["sum_true2"] / n - mean_true * mean_true, 0.0)

    if math.isnan(residual_se):
        ci_low = float("nan")
        ci_high = float("nan")
    else:
        ci_low = mean_res - 1.96 * residual_se
        ci_high = mean_res + 1.96 * residual_se

    return {
        "n": n,
        "mean_true": mean_true,
        "mean_pred": mean_pred,
        "mean_signed_residual": mean_res,
        "mean_abs_residual": mean_abs_res,
        "rmse": rmse,
        "residual_var": residual_var,
        "residual_std": residual_std,
        "residual_se": residual_se,
        "residual_ci95_low": ci_low,
        "residual_ci95_high": ci_high,
        "pred_var": pred_var,
        "pred_std": math.sqrt(pred_var),
        "true_var": true_var,
        "true_std": math.sqrt(true_var),
    }


def _weighted_mean(values, weights):
    if len(values) == 0:
        return float("nan")
    sw = sum(weights)
    if sw <= 0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights)) / sw


def _weighted_var(values, weights):
    if len(values) == 0:
        return float("nan")
    sw = sum(weights)
    if sw <= 0:
        return float("nan")
    m = _weighted_mean(values, weights)
    return sum(w * (v - m) ** 2 for v, w in zip(values, weights)) / sw


def _join_ids(ids):
    return "|".join(str(x) for x in sorted(ids, key=_safe_id_sort_key))


def _join_names(ids, name_map):
    names = [name_map.get(int(x), "") for x in sorted(ids, key=_safe_id_sort_key)]
    return "|".join(names)


def analyze_valid_question_residuals(
    save_dir,
    batch_size=256,
    output_dir=None,
    signed_mode="true_minus_pred",
    save_interactions=False,
    min_pair_questions=2,
):
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

    if model_name != "dkt":
        raise ValueError(
            f"This script currently supports model_name='dkt' only, got '{model_name}'."
        )

    with (PROJECT_ROOT / "configs" / "data_config.json").open("r", encoding="utf-8") as f:
        full_data_config = json.load(f)

    data_cfg = full_data_config[dataset_name]
    data_cfg["dataset_name"] = dataset_name

    model_config = _clean_model_config(model_name, run_cfg["model_config"])

    # If later you extend to SAKT/SimpleKT/etc, seq_len injection may be needed.
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

    model = load_model(model_name, model_config, data_cfg, emb_type, str(save_dir))
    model.eval()

    concept_name_map, question_name_map = _load_id_name_maps(data_cfg)

    # Main aggregations
    question_stats = defaultdict(_new_stat)
    kc_stats = defaultdict(_new_stat)
    qkc_stats = defaultdict(_new_stat)

    # question -> observed KCs
    q_to_kcs = defaultdict(set)

    # Optional raw interaction rows
    interaction_rows = []

    with torch.no_grad():
        for dcur in valid_loader:
            required_keys = [
                "cseqs",
                "rseqs",
                "shft_cseqs",
                "shft_rseqs",
                "smasks",
            ]
            for key in required_keys:
                if key not in dcur:
                    raise KeyError(f"Missing required key in batch: {key}")

            if "shft_qseqs" not in dcur:
                raise KeyError(
                    "Missing 'shft_qseqs' in batch. "
                    "You need question-level sequences. "
                    "Please check whether your dataset was preprocessed with question IDs."
                )

            c = dcur["cseqs"].to(device)
            r = dcur["rseqs"].to(device)
            cshft = dcur["shft_cseqs"].to(device)
            rshft = dcur["shft_rseqs"].to(device)
            qshft = dcur["shft_qseqs"].to(device)
            sm = dcur["smasks"].to(device)

            # DKT forward:
            # y_full shape usually: [batch, seq_len, num_c]
            y_full = model(c.long(), r.long())

            # Select prediction corresponding to next concept cshft.
            y = (y_full * one_hot(cshft.long(), model.num_c)).sum(-1)

            pred_arr = torch.masked_select(y, sm).detach().cpu().numpy()
            true_arr = torch.masked_select(rshft, sm).detach().cpu().numpy()
            kc_arr = torch.masked_select(cshft, sm).detach().cpu().numpy()
            q_arr = torch.masked_select(qshft, sm).detach().cpu().numpy()

            for qid, kc, true_value, pred_value in zip(q_arr, kc_arr, true_arr, pred_arr):
                qid = int(qid)
                kc = int(kc)
                t = float(true_value)
                p = float(pred_value)

                if signed_mode == "pred_minus_true":
                    residual = p - t
                else:
                    residual = t - p

                question_stats[qid]
                kc_stats[kc]
                qkc_stats[(qid, kc)]

                _update_stat(question_stats[qid], t, p, residual)
                _update_stat(kc_stats[kc], t, p, residual)
                _update_stat(qkc_stats[(qid, kc)], t, p, residual)

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
        fs = _finalize_stat(st)
        kcs = q_to_kcs.get(qid, set())

        question_rows.append(
            {
                "question_id": qid,
                "question_name": question_name_map.get(qid, ""),
                "kc_ids": _join_ids(kcs),
                "kc_names": _join_names(kcs, concept_name_map),
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
    # 2. KC-level residual table, interaction-level aggregation
    # ------------------------------------------------------------
    kc_rows = []
    for kc, st in kc_stats.items():
        fs = _finalize_stat(st)
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
    #    Useful when one question has multiple KCs.
    # ------------------------------------------------------------
    qkc_rows = []
    for (qid, kc), st in qkc_stats.items():
        fs = _finalize_stat(st)
        qkc_rows.append(
            {
                "question_id": qid,
                "question_name": question_name_map.get(qid, ""),
                "kc_id": kc,
                "kc_name": concept_name_map.get(kc, ""),
                "question_all_kc_ids": _join_ids(q_to_kcs.get(qid, set())),
                "question_all_kc_names": _join_names(q_to_kcs.get(qid, set()), concept_name_map),
                "question_kc_count": len(q_to_kcs.get(qid, set())),
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
    #
    # This is important for your diagnosis:
    # For each KC, look at the distribution of question-level residuals
    # among questions containing this KC.
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

        q_res_var_w = _weighted_var(q_residuals, weights)
        q_acc_var_w = _weighted_var(q_accs, weights)
        q_pred_var_w = _weighted_var(q_preds, weights)

        kc_q_rows.append(
            {
                "kc_id": kc,
                "kc_name": concept_name_map.get(kc, ""),
                "n_questions": len(items),
                "n_valid_points_total": sum(weights),
                "weighted_mean_question_residual": _weighted_mean(q_residuals, weights),
                "weighted_mean_abs_question_residual": _weighted_mean(q_abs_residuals, weights),
                "weighted_question_residual_var": q_res_var_w,
                "weighted_question_residual_std": math.sqrt(q_res_var_w)
                if not math.isnan(q_res_var_w)
                else float("nan"),
                "weighted_question_acc_var": q_acc_var_w,
                "weighted_question_acc_std": math.sqrt(q_acc_var_w)
                if not math.isnan(q_acc_var_w)
                else float("nan"),
                "weighted_question_pred_var": q_pred_var_w,
                "weighted_question_rmse": _weighted_mean(q_rmses, weights),
                "max_abs_question_residual": max(q_abs_residuals) if q_abs_residuals else float("nan"),
                "mean_question_kc_count": sum(q_kc_counts) / len(q_kc_counts)
                if q_kc_counts
                else float("nan"),
                "multi_kc_question_ratio": sum(1 for x in q_kc_counts if x > 1) / len(q_kc_counts)
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
    #
    # For every question with multiple observed KCs, assign the question-level
    # residual to every pair in that question.
    # This is a cheap signal for combination-sensitive KC pairs.
    # ------------------------------------------------------------
    pair_stats = defaultdict(_new_stat)
    pair_question_count = defaultdict(int)

    for _, row in question_df.iterrows():
        qid = int(row["question_id"])
        kcs = sorted(q_to_kcs.get(qid, set()), key=_safe_id_sort_key)

        if len(kcs) < 2:
            continue

        # Use question-level values as the unit.
        # Weight is approximated by repeating via n_valid_points in _update_stat
        # only once here, so later we separately record n_points.
        t = float(row["mean_true"])
        p = float(row["mean_pred"])
        residual = float(row["mean_signed_residual"])
        n_points = int(row["n_valid_points"])

        for a, b in combinations(kcs, 2):
            pair = (int(a), int(b))
            pair_question_count[pair] += 1

            # Store weighted effect manually by expanding through sums.
            # This avoids appending the same question n_points times.
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

        fs = _finalize_stat(st)
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
    if output_dir is None or str(output_dir).strip() == "":
        output_dir = save_dir / f"{emb_type}_valid_residual_diagnosis"
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

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
        print(question_df.head(20)[cols].to_string(index=False))

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
        print(kc_question_df.head(20)[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Compute question-level residuals and diagnosis tables on VALID split."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Trained run directory containing config.json and model checkpoint.",
    )
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument(
        "--signed_mode",
        type=str,
        default="true_minus_pred",
        choices=["true_minus_pred", "pred_minus_true"],
        help=(
            "Definition of signed residual. "
            "true_minus_pred means positive residual = model under-predicts correctness."
        ),
    )
    parser.add_argument(
        "--save_interactions",
        action="store_true",
        help="Save raw interaction-level residuals. This file can be large.",
    )
    parser.add_argument(
        "--min_pair_questions",
        type=int,
        default=2,
        help="Minimum number of questions for a KC pair to appear in pair summary.",
    )

    args = parser.parse_args()

    analyze_valid_question_residuals(
        save_dir=args.save_dir,
        batch_size=args.bz,
        output_dir=args.output_dir,
        signed_mode=args.signed_mode,
        save_interactions=args.save_interactions,
        min_pair_questions=args.min_pair_questions,
    )


if __name__ == "__main__":
    # Example:
    # python analyze_valid_question_residuals.py --save_dir "saved_model/xxx" --bz 256
    main()