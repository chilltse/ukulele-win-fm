#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import json
import wandb
import numpy as np
import pandas as pd


# =========================
# 1. Default configuration
# =========================

DEFAULT_USER = "chilltse808-anu"
DEFAULT_PROJECT_NAME = "ukulele-win-fm-examples"
DEFAULT_DATASET_NAME = "xes3g5m"
DEFAULT_MODEL_NAME = "simplekt"
DEFAULT_EMB_TYPE = "qid"

DEFAULT_USE_CACHE = False
DEFAULT_CACHE_DIR = "results/wandb_result"
DEFAULT_PRINT_STD = True
DEFAULT_ONLY_FINISHED = True
DEFAULT_MIN_FINISHED_RUNS = 5

# Optional: read W&B API key from ../configs/wandb.json if available.
# If you have already run `wandb login`, this part is not strictly necessary.
DEFAULT_WANDB_CONFIG_PATH = "../configs/wandb.json"


# =========================
# 2. CLI helpers
# =========================

def str2bool(value):
    """
    Parse common command-line boolean formats.

    Examples:
        --use_cache true
        --use_cache False
        --only_finished 1
        --print_std no
    """
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    if value in {"true", "t", "1", "yes", "y"}:
        return True
    if value in {"false", "f", "0", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_wandb_api_key(config_path):
    """
    Load W&B API key from a JSON config file if it exists.

    Expected format:
        {
          "api_key": "..."
        }
    """
    if not config_path:
        return

    if not os.path.exists(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        wandb_config = json.load(f)

    api_key = wandb_config.get("api_key")
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Summarize finished W&B prediction sweep results."
    )

    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--project_name", "--project-name", default=DEFAULT_PROJECT_NAME)
    parser.add_argument("--dataset_name", "--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--model_name", "--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--emb_type", "--emb-type", default=DEFAULT_EMB_TYPE)

    parser.add_argument("--print_std", "--print-std", type=str2bool, default=DEFAULT_PRINT_STD)
    parser.add_argument("--use_cache", "--use-cache", type=str2bool, default=DEFAULT_USE_CACHE)
    parser.add_argument("--cache_dir", "--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--only_finished", "--only-finished", type=str2bool, default=DEFAULT_ONLY_FINISHED)

    parser.add_argument(
        "--min_finished_runs",
        "--min-finished-runs",
        type=int,
        default=DEFAULT_MIN_FINISHED_RUNS,
        help="Minimum number of unique finished prediction runs expected. Usually equals the fold count.",
    )
    parser.add_argument(
        "--wandb_config_path",
        "--wandb-config-path",
        default=DEFAULT_WANDB_CONFIG_PATH,
        help="Optional JSON file containing {'api_key': '...'}.",
    )

    return parser


# =========================
# 3. Helper functions
# =========================

def get_runs_result(runs):
    """
    Convert W&B sweep runs into a pandas DataFrame.
    This collects:
      - run.summary metrics, e.g. testauc, testacc
      - run.config parameters, e.g. save_dir
      - run name, path id, and state
    """
    result_list = []

    for run in runs:
        result = {}

        # Metrics stored in W&B summary
        result.update(run.summary._json_dict)

        # Hyperparameters / config values
        model_config = {
            k: v for k, v in run.config.items()
            if not k.startswith("_") and type(v) not in [list, dict]
        }
        result.update(model_config)

        result["name"] = run.name
        result["path_id"] = run.path[-1]
        result["state"] = run.state

        result_list.append(result)

    runs_df = pd.DataFrame(result_list)

    if "_timestamp" in runs_df.columns:
        runs_df["create_time"] = runs_df["_timestamp"]

    return runs_df


def get_sweep_dict(api, project):
    """
    Get all sweeps from a W&B project.

    Return:
        {
            sweep_name: sweep_id
        }

    If the same sweep name appears multiple times, this script keeps the latest one
    by default instead of crashing.
    """
    raw = {}

    for sweep in project.sweeps():
        raw.setdefault(sweep.name, [])
        raw[sweep.name].append(sweep.id)

    sweep_dict = {}

    for name, ids in raw.items():
        if len(ids) > 1:
            print(f"[Warning] Duplicate sweep name found: {name}")
            print(f"          Sweep ids: {ids}")
            print(f"          This script will use the last one: {ids[-1]}")
            sweep_dict[name] = ids[-1]
        else:
            sweep_dict[name] = ids[0]

    return sweep_dict


def get_df(
    api,
    user,
    project_name,
    sweep_dict,
    sweep_name,
    use_cache=False,
    cache_dir="results/wandb_result",
    only_finished=True,
):
    """
    Load one sweep's run results as a DataFrame.
    """
    os.makedirs(cache_dir, exist_ok=True)

    if sweep_name not in sweep_dict:
        print(f"[Error] Cannot find sweep name: {sweep_name}")
        print("\nAvailable similar sweep names:")

        # Use the requested sweep name itself instead of global constants, so this
        # error message remains correct when arguments come from CLI.
        parts = [p for p in sweep_name.replace("/", "_").split("_") if p]
        for key in sorted(sweep_dict.keys()):
            if any(part in key for part in parts):
                print("  ", key)

        raise KeyError(f"Sweep name not found: {sweep_name}")

    sweep_id = sweep_dict[sweep_name]
    df_cache_path = os.path.join(cache_dir, f"{sweep_id}.csv")

    if use_cache and os.path.exists(df_cache_path):
        print(f"[Info] Loading cached results from: {df_cache_path}")
        df = pd.read_csv(df_cache_path)
    else:
        print(f"[Info] Fetching sweep from W&B: {user}/{project_name}/{sweep_id}")
        sweep = api.sweep(f"{user}/{project_name}/{sweep_id}")
        df = get_runs_result(sweep.runs)

        df.to_csv(df_cache_path, index=False)
        print(f"[Info] Saved cache to: {df_cache_path}")

    if df.empty:
        raise ValueError(f"No runs found in sweep: {sweep_name}")

    if only_finished and "state" in df.columns:
        before = len(df)
        df = df[df["state"] == "finished"].copy()
        after = len(df)
        print(f"[Info] Keep finished runs only: {before} -> {after}")

    if "_timestamp" in df.columns:
        df["create_time"] = df["_timestamp"].apply(int)
        df = df.sort_values("create_time")

    df["run_index"] = range(len(df))
    df.index = range(len(df))

    return df


def format_mean_std(values, print_std=True):
    """
    Compute mean/std and format as string.
    """
    values = np.array(values, dtype=float)
    mean = np.mean(values)
    std = np.std(values, ddof=0)

    if print_std:
        return f"{mean:.4f}±{std:.4f}"
    else:
        return f"{mean:.4f}"


def summarize_metric_group(key, group_name, all_res, metric_names, print_std=True):
    """
    Print one group of metrics and return summary rows.

    Important:
    Do NOT use np.unique() here.
    Each row/run should be treated as one experimental result.
    """
    missing = [m for m in metric_names if m not in all_res.columns]

    if missing:
        print(f"{key}_{group_name}: missing columns {missing}, skip.")
        return []

    outputs = []
    rows = []

    for metric in metric_names:
        values = pd.to_numeric(all_res[metric], errors="coerce").dropna().values

        if len(values) == 0:
            outputs.append("nan")
            rows.append(
                {
                    "key": key,
                    "group": group_name,
                    "metric": metric,
                    "num_runs": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "mean_std": "nan",
                }
            )
            continue

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))

        if print_std:
            formatted = f"{mean:.4f}±{std:.4f}"
        else:
            formatted = f"{mean:.4f}"

        outputs.append(formatted)

        rows.append(
            {
                "key": key,
                "group": group_name,
                "metric": metric,
                "num_runs": int(len(values)),
                "mean": mean,
                "std": std,
                "mean_std": formatted,
            }
        )

    print(f"{key}_{group_name}: " + ",".join(outputs))
    return rows


def extract_prediction_results(
    user,
    project_name,
    dataset_name,
    model_name,
    emb_type="qid",
    print_std=True,
    use_cache=False,
    cache_dir="results/wandb_result",
    only_finished=True,
    min_finished_runs=5,
):
    """
    Standalone version of:

        WandbUtils.extract_prediction_results(dataset_name, model_name, emb_type, print_std=True)

    It reads finished prediction runs from W&B and prints average/std results.
    It does NOT run prediction jobs.
    """
    api = wandb.Api(timeout=180)
    project = api.project(name=project_name)

    sweep_dict = get_sweep_dict(api, project)

    prediction_sweep_name = f"pred_wandbs/{dataset_name}_{model_name}_{emb_type}"

    print("=" * 80)
    print(f"Project: {user}/{project_name}")
    print(f"Prediction sweep name: {prediction_sweep_name}")
    print("=" * 80)

    all_res = get_df(
        api=api,
        user=user,
        project_name=project_name,
        sweep_dict=sweep_dict,
        sweep_name=prediction_sweep_name,
        use_cache=use_cache,
        cache_dir=cache_dir,
        only_finished=only_finished,
    )

    if "save_dir" not in all_res.columns:
        print("[Error] Column `save_dir` not found.")
        print("Available columns:")
        print(list(all_res.columns))
        return

    before = len(all_res)
    all_res = all_res.drop_duplicates(["save_dir"])
    after = len(all_res)

    print(f"[Info] Drop duplicate save_dir: {before} -> {after}")

    if len(all_res) < min_finished_runs:
        print("Failure running exists, please check!!!")
        print(
            f"Only found {len(all_res)} unique finished prediction runs. "
            f"Expected at least {min_finished_runs}."
        )
        print("\nCurrent runs:")
        cols = [c for c in ["name", "state", "save_dir", "testauc", "testacc"] if c in all_res.columns]
        print(all_res[cols].to_string(index=False))
        return

    key = dataset_name + "_" + model_name

    summary_rows = []

    # repeated metrics
    summary_rows.extend(
        summarize_metric_group(
            key=key,
            group_name="repeated",
            all_res=all_res,
            metric_names=[
                "testauc",
                "testacc",
                "window_testauc",
                "window_testacc",
            ],
            print_std=print_std,
        )
    )

    # concept-level metrics
    summary_rows.extend(
        summarize_metric_group(
            key=key,
            group_name="concepts",
            all_res=all_res,
            metric_names=[
                "oriaucconcepts",
                "oriaccconcepts",
                "windowaucconcepts",
                "windowaccconcepts",
            ],
            print_std=print_std,
        )
    )

    # early fusion metrics
    summary_rows.extend(
        summarize_metric_group(
            key=key,
            group_name="early",
            all_res=all_res,
            metric_names=[
                "oriaucearly_preds",
                "oriaccearly_preds",
                "windowaucearly_preds",
                "windowaccearly_preds",
            ],
            print_std=print_std,
        )
    )

    # late mean metrics
    summary_rows.extend(
        summarize_metric_group(
            key=key,
            group_name="latemean",
            all_res=all_res,
            metric_names=[
                "oriauclate_mean",
                "oriacclate_mean",
                "windowauclate_mean",
                "windowacclate_mean",
            ],
            print_std=print_std,
        )
    )

    # late vote metrics
    summary_rows.extend(
        summarize_metric_group(
            key=key,
            group_name="latevote",
            all_res=all_res,
            metric_names=[
                "oriauclate_vote",
                "oriacclate_vote",
                "windowauclate_vote",
                "windowacclate_vote",
            ],
            print_std=print_std,
        )
    )

    # late all metrics
    summary_rows.extend(
        summarize_metric_group(
            key=key,
            group_name="lateall",
            all_res=all_res,
            metric_names=[
                "oriauclate_all",
                "oriacclate_all",
                "windowauclate_all",
                "windowacclate_all",
            ],
            print_std=print_std,
        )
    )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(
            cache_dir, f"{dataset_name}_{model_name}_{emb_type}_summary.csv"
        )
        summary_df.to_csv(summary_path, index=False)
        print(f"[Info] Saved summary with std to: {summary_path}")


def main():
    args = build_arg_parser().parse_args()

    load_wandb_api_key(args.wandb_config_path)

    extract_prediction_results(
        user=args.user,
        project_name=args.project_name,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        emb_type=args.emb_type,
        print_std=args.print_std,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir,
        only_finished=args.only_finished,
        min_finished_runs=args.min_finished_runs,
    )


# =========================
# 4. Main entry
# =========================

if __name__ == "__main__":
    main()
