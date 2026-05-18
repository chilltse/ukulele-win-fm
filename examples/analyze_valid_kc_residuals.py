import argparse
import json
import math
import os
from collections import defaultdict
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


def _load_concept_name_map(data_config):
    keyid2idx_path = Path(data_config["dpath"]) / "keyid2idx.json"
    if not keyid2idx_path.exists():
        return {}
    try:
        with keyid2idx_path.open("r", encoding="utf-8") as f:
            keyid2idx = json.load(f)
        concept_map = keyid2idx.get("concepts", {})
        # keyid2idx: original_id -> internal_id
        # invert to internal_id(int) -> original_id(str)
        return {int(v): str(k) for k, v in concept_map.items()}
    except Exception as e:
        print(f"[warn] failed to parse {keyid2idx_path}: {e}")
        return {}


def _clean_model_config(model_name, cfg):
    model_config = dict(cfg)
    for k in ["use_wandb", "learning_rate", "add_uuid", "l2", "tree_pred_decay_lr_mult"]:
        if k in model_config:
            del model_config[k]
    if model_name in ["saint", "saint++", "sakt", "atdkt", "simplekt", "stablekt", "datakt", "folibikt"]:
        # seq_len will be injected from train_config by caller.
        pass
    return model_config


def analyze_valid_kc_residuals(save_dir, batch_size=256, output_csv=None, signed_mode="true_minus_pred"):
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
        raise ValueError(f"This script currently supports model_name='dkt' only, got '{model_name}'.")

    with (PROJECT_ROOT / "configs" / "data_config.json").open("r", encoding="utf-8") as f:
        full_data_config = json.load(f)
    data_cfg = full_data_config[dataset_name]
    data_cfg["dataset_name"] = dataset_name

    model_config = _clean_model_config(model_name, run_cfg["model_config"])

    if model_name in ["saint", "saint++", "sakt", "atdkt", "simplekt", "stablekt", "datakt", "folibikt"]:
        model_config["seq_len"] = run_cfg["train_config"]["seq_len"]

    # IMPORTANT: this only reads train_valid_file and keeps fold==valid_fold.
    _, valid_loader, *_ = init_dataset4train(
        dataset_name=dataset_name,
        model_name=model_name,
        data_config=full_data_config,
        i=valid_fold,
        batch_size=batch_size,
    )

    model = load_model(model_name, model_config, data_cfg, emb_type, str(save_dir))
    model.eval()

    # kc -> aggregate sums
    agg = defaultdict(lambda: {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0, "sum_signed": 0.0, "sum_true": 0.0, "sum_pred": 0.0})

    with torch.no_grad():
        for dcur in valid_loader:
            c, r = dcur["cseqs"], dcur["rseqs"]
            cshft, rshft = dcur["shft_cseqs"], dcur["shft_rseqs"]
            sm = dcur["smasks"]

            c = c.to(device)
            r = r.to(device)
            cshft = cshft.to(device)
            rshft = rshft.to(device)
            sm = sm.to(device)

            # simple DKT: forward(q, r) with concept ids as q
            y_full = model(c.long(), r.long())
            y = (y_full * one_hot(cshft.long(), model.num_c)).sum(-1)

            pred = torch.masked_select(y, sm).detach().cpu().numpy()
            true = torch.masked_select(rshft, sm).detach().cpu().numpy()
            kc_ids = torch.masked_select(cshft, sm).detach().cpu().numpy()

            for kc, t, p in zip(kc_ids, true, pred):
                kc = int(kc)
                t = float(t)
                p = float(p)

                if signed_mode == "pred_minus_true":
                    signed = p - t
                else:
                    signed = t - p

                st = agg[kc]
                st["n"] += 1
                st["sum_abs"] += abs(signed)
                st["sum_sq"] += signed * signed
                st["sum_signed"] += signed
                st["sum_true"] += t
                st["sum_pred"] += p

    concept_name_map = _load_concept_name_map(data_cfg)
    rows = []
    for kc, st in agg.items():
        n = st["n"]
        if n <= 0:
            continue
        rows.append(
            {
                "kc_id": kc,
                "kc_name": concept_name_map.get(kc, ""),
                "n_valid_points": n,
                "mean_abs_residual": st["sum_abs"] / n,
                "rmse": math.sqrt(st["sum_sq"] / n),
                "mean_signed_residual": st["sum_signed"] / n,
                "mean_true": st["sum_true"] / n,
                "mean_pred": st["sum_pred"] / n,
            }
        )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values("kc_id", key=lambda s: s.map(_to_number)).reset_index(drop=True)

    if output_csv is None or str(output_csv).strip() == "":
        output_csv = save_dir / f"{emb_type}_valid_kc_residuals.csv"
    else:
        output_csv = Path(output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("\n=== VALID KC residuals done ===")
    print(f"save_dir: {save_dir}")
    print(f"dataset_name: {dataset_name}, valid_fold: {valid_fold}, emb_type: {emb_type}")
    print(f"signed_mode: {signed_mode}")
    print(f"rows: {len(out_df)}")
    print(f"output_csv: {output_csv}")

    if not out_df.empty:
        print("\nTop 20 by n_valid_points:")
        print(
            out_df.sort_values("n_valid_points", ascending=False)
            .head(20)[
                [
                    "kc_id",
                    "kc_name",
                    "n_valid_points",
                    "mean_abs_residual",
                    "rmse",
                    "mean_signed_residual",
                    "mean_true",
                    "mean_pred",
                ]
            ]
            .to_string(index=False)
        )


def main():
    parser = argparse.ArgumentParser(description="Compute per-KC residual metrics on VALID split only.")
    parser.add_argument("--save_dir", type=str, required=True, help="Trained run directory containing config.json and *_model.ckpt")
    parser.add_argument("--bz", type=int, default=256)
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument(
        "--signed_mode",
        type=str,
        default="true_minus_pred",
        choices=["true_minus_pred", "pred_minus_true"],
        help="Definition of signed residual.",
    )
    args = parser.parse_args()

    analyze_valid_kc_residuals(
        save_dir=args.save_dir,
        batch_size=args.bz,
        output_csv=args.output_csv,
        signed_mode=args.signed_mode,
    )


if __name__ == "__main__":
    # example:
    # python analyze_valid_kc_residuals.py --save_dir "saved_model/xxx"
    main()
