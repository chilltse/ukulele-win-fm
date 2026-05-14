import argparse
import csv
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# Utility functions
# ============================================================

def run_cmd(cmd: List[str], cwd: Path, log_file: Path) -> Tuple[int, str, str]:
    """
    Run a command and append stdout/stderr to log file.
    """
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"[{datetime.now().isoformat()}] CWD={cwd}\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.write("-" * 80 + "\n")
        f.write("--- STDOUT ---\n")
        f.write(proc.stdout or "")
        f.write("\n--- STDERR ---\n")
        f.write(proc.stderr or "")
        f.write("\n")
        f.write(f"EXIT_CODE={proc.returncode}\n")

    return proc.returncode, proc.stdout, proc.stderr


def parse_train_metrics(text: str) -> Dict[str, str]:
    """
    Parse metrics printed by wandb_*_train.py.

    Expected final row format:
        fold modelname embtype testauc testacc window_testauc window_testacc validauc validacc best_epoch

    Example:
        0 akt qid_fmkc 0.7234 0.6812 0.7123 0.6744 0.7301 0.6902 18

    Also supports fallback key-value logs like:
        testauc: 0.7234, testacc: 0.6812, validauc: 0.7301
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # First try to parse the final summary row.
    for ln in reversed(lines):
        parts = ln.split()

        if len(parts) >= 10 and parts[0].isdigit():
            return {
                "fold": parts[0],
                "modelname": parts[1],
                "embtype": parts[2],
                "testauc": parts[3],
                "testacc": parts[4],
                "window_testauc": parts[5],
                "window_testacc": parts[6],
                "validauc": parts[7],
                "validacc": parts[8],
                "best_epoch": parts[9],
            }

    # Fallback: parse key-value style logs.
    metric_names = [
        "testauc",
        "testacc",
        "window_testauc",
        "window_testacc",
        "validauc",
        "validacc",
    ]

    out: Dict[str, str] = {}

    for name in metric_names:
        pattern = rf"\b{name}\s*[:=]\s*(-?\d+(?:\.\d+)?)"
        matches = re.findall(pattern, text)
        if matches:
            out[name] = matches[-1]

    best_epoch_patterns = [
        r"\bbest_epoch\s*[:=]\s*(\d+)",
        r"\bbest epoch\s*[:=]\s*(\d+)",
    ]

    for pattern in best_epoch_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            out["best_epoch"] = matches[-1]
            break

    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def assert_file_exists(path: Path, message: str = "") -> None:
    if not path.exists():
        if message:
            raise FileNotFoundError(f"{message}: {path}")
        raise FileNotFoundError(f"File not found: {path}")


def assert_dir_exists(path: Path, message: str = "") -> None:
    if not path.exists() or not path.is_dir():
        if message:
            raise FileNotFoundError(f"{message}: {path}")
        raise FileNotFoundError(f"Directory not found: {path}")


def clear_pkl_files(dataset_dir: Path, log_file: Optional[Path] = None) -> None:
    """
    Delete old preprocessed cache files.

    This is important because every union run overwrites the same JSON file:
        yousician_ukulele.json

    If old .pkl cache files remain, training may accidentally use stale data.
    """
    if not dataset_dir.exists():
        return

    deleted = []

    for pattern in ["*.pkl", "*.pkl.*"]:
        for p in dataset_dir.rglob(pattern):
            try:
                p.unlink()
                deleted.append(str(p))
            except Exception as ex:
                if log_file is not None:
                    with log_file.open("a", encoding="utf-8") as f:
                        f.write(f"\n[WARNING] Failed to delete cache file: {p}\n")
                        f.write(f"Reason: {ex}\n")

    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as f:
            f.write("\n[CACHE CLEANUP]\n")
            if deleted:
                for p in deleted:
                    f.write(f"Deleted: {p}\n")
            else:
                f.write("No .pkl cache files found.\n")


def backup_file(path: Path) -> Optional[Path]:
    """
    Backup original JSON before overwriting it.
    """
    if not path.exists():
        return None

    backup_path = path.with_name(path.name + ".backup_before_union_grid")

    if not backup_path.exists():
        shutil.copy2(path, backup_path)

    return backup_path


def restore_file_from_backup(target_path: Path, backup_path: Optional[Path]) -> None:
    """
    Restore JSON from backup.
    """
    if backup_path is None:
        return

    if backup_path.exists():
        shutil.copy2(backup_path, target_path)


def parse_int_list(raw_values: Optional[List[str]]) -> List[int]:
    """
    Support both:
        --union_levels 5 4 3 2 1
    and:
        --union_levels 5,4,3,2,1
    """
    if not raw_values:
        return []

    values: List[int] = []

    for item in raw_values:
        for part in item.split(","):
            part = part.strip()
            if part:
                values.append(int(part))

    return values


def parse_str_list(raw_values: Optional[List[str]]) -> List[str]:
    """
    Support both:
        --models dkt akt dkt_plus
    and:
        --models dkt,akt,dkt_plus
    """
    if not raw_values:
        return []

    values: List[str] = []

    for item in raw_values:
        for part in item.split(","):
            part = part.strip()
            if part:
                values.append(part)

    return values


def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in s)


def find_repo_root(start: Path) -> Path:
    """
    Find repo root by searching upward for both examples/ and data/.
    """
    start = start.resolve()

    if start.is_file():
        start = start.parent

    for p in [start] + list(start.parents):
        if (p / "examples").is_dir() and (p / "data").is_dir():
            return p

    raise FileNotFoundError(
        "Cannot automatically find repo root. "
        "Please pass --repo_root G:\\ANU_course\\pykt-ukulele-May-WIN\\pykt-ukelele"
    )


def resolve_train_script(examples_dir: Path, model_name: str, train_script_dir: str = "") -> Path:
    """
    Resolve model-specific training script.

    Examples:
        dkt       -> wandb_dkt_train.py
        akt       -> wandb_akt_train.py
        dkt_plus  -> wandb_dkt_plus_train.py
        sakt      -> wandb_sakt_train.py
        simplekt  -> wandb_simplekt_train.py
    """
    script_name = f"wandb_{model_name}_train.py"

    if train_script_dir:
        p = Path(train_script_dir)
        if not p.is_absolute():
            p = examples_dir / p
        return p / script_name

    return examples_dir / script_name


def build_preprocess_cmd(
    preprocess_script: Path,
    dataset_name: str,
    preprocess_arg_name: str,
) -> List[str]:
    return [
        sys.executable,
        str(preprocess_script),
        preprocess_arg_name,
        dataset_name,
    ]


def build_train_cmd(
    args: argparse.Namespace,
    train_script: Path,
    dataset_name: str,
    emb_type: str,
    fold: int,
) -> List[str]:
    cmd = [
        sys.executable,
        str(train_script),
        "--dataset_name",
        dataset_name,
        "--emb_type",
        emb_type,
    ]

    if not args.no_pass_seed:
        cmd.extend(["--seed", str(args.seed)])

    if not args.no_pass_fold:
        cmd.extend(["--fold", str(fold)])

    if args.extra_train_args:
        cmd.extend(args.extra_train_args)

    return cmd


def write_summary(summary_csv: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "union_level",
        "union_file",
        "model_name_requested",
        "train_script",
        "dataset_name",
        "dataset_variant",
        "status",
        "error_stage",
        "error_message",
        "fold",
        "modelname",
        "embtype",
        "testauc",
        "testacc",
        "window_testauc",
        "window_testacc",
        "validauc",
        "validacc",
        "best_epoch",
    ]

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Yousician union-level grid pipeline for qid baseline and qid_fmkc."
    )

    parser.add_argument(
        "--repo_root",
        type=str,
        default="",
        help="Repo root. Default: auto-detect from current script location.",
    )

    parser.add_argument(
        "--union_levels",
        nargs="+",
        default=["5,4,3,2,1"],
        help="Union levels to run. Default: 5,4,3,2,1",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["dkt,akt,dkt_plus,sakt,simplekt"],
        help="Models to run. Default: dkt,akt,dkt_plus,sakt,simplekt",
    )

    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0"],
        help="Training folds. Example: --folds 0 1 2 3 4 or --folds 0,1,2,3,4",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed.",
    )

    parser.add_argument(
        "--source_dir",
        type=str,
        default="",
        help=(
            "Source JSON directory. Default: "
            "data/yousician/split_raw_json"
        ),
    )

    parser.add_argument(
        "--source_json_pattern",
        type=str,
        default="yousician_ukulele_union_level_{level}.json",
        help="Source JSON filename pattern. Must contain {level}.",
    )

    parser.add_argument(
        "--target_json_name",
        type=str,
        default="yousician_ukulele.json",
        help="Target JSON filename copied into yousician and yousician_fmkc.",
    )

    parser.add_argument(
        "--baseline_dataset",
        type=str,
        default="yousician",
        help="Baseline dataset name.",
    )

    parser.add_argument(
        "--fmkc_dataset",
        type=str,
        default="yousician_fmkc",
        help="FMKC dataset name.",
    )

    parser.add_argument(
        "--preprocess_arg_name",
        type=str,
        default="--dataset_name",
        help=(
            "Argument name used by data_preprocess.py. "
            "If your script uses -d, pass: --preprocess_arg_name -d"
        ),
    )

    parser.add_argument(
        "--train_script_dir",
        type=str,
        default="",
        help="Optional directory containing wandb_<model>_train.py. Default: examples directory.",
    )

    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Skip data_preprocess.py.",
    )

    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training; only copy JSON and preprocess.",
    )

    parser.add_argument(
        "--no_restore_original_json",
        action="store_true",
        help="Do not restore original yousician_ukulele.json files at the end.",
    )

    parser.add_argument(
        "--no_pass_seed",
        action="store_true",
        help="Do not pass --seed to training scripts.",
    )

    parser.add_argument(
        "--no_pass_fold",
        action="store_true",
        help="Do not pass --fold to training scripts.",
    )

    parser.add_argument(
        "--extra_train_args",
        nargs=argparse.REMAINDER,
        default=None,
        help="Extra arguments appended to every training command. Put this at the end.",
    )

    args = parser.parse_args()

    union_levels = parse_int_list(args.union_levels)
    models = parse_str_list(args.models)
    folds = parse_int_list(args.folds)

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = find_repo_root(Path(__file__).resolve())

    examples_dir = repo_root / "examples"
    data_root = repo_root / "data"

    baseline_dir = data_root / args.baseline_dataset
    fmkc_dir = data_root / args.fmkc_dataset

    if args.source_dir:
        source_dir = Path(args.source_dir)
        if not source_dir.is_absolute():
            source_dir = repo_root / source_dir
    else:
        source_dir = baseline_dir / "split_raw_json"

    preprocess_script = examples_dir / "data_preprocess.py"

    baseline_target_json = baseline_dir / args.target_json_name
    fmkc_target_json = fmkc_dir / args.target_json_name

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = examples_dir / "grid_runs" / f"yousician_union_grid_{run_stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    summary_csv = output_root / "summary.csv"

    # ------------------------------------------------------------
    # Pre-check
    # ------------------------------------------------------------
    assert_dir_exists(repo_root, "Repository root not found")
    assert_dir_exists(examples_dir, "examples directory not found")
    assert_dir_exists(data_root, "data directory not found")
    assert_dir_exists(baseline_dir, f"{args.baseline_dataset} directory not found")
    assert_dir_exists(fmkc_dir, f"{args.fmkc_dataset} directory not found")
    assert_dir_exists(source_dir, "Source JSON directory not found")
    assert_file_exists(preprocess_script, "data_preprocess.py not found")

    for model_name in models:
        train_script = resolve_train_script(examples_dir, model_name, args.train_script_dir)
        assert_file_exists(train_script, f"Training script not found for model={model_name}")

    for level in union_levels:
        source_json = source_dir / args.source_json_pattern.format(level=level)
        assert_file_exists(source_json, f"Source union JSON not found for union_level={level}")

    ensure_parent(baseline_target_json)
    ensure_parent(fmkc_target_json)

    # ------------------------------------------------------------
    # Backup original JSONs
    # ------------------------------------------------------------
    baseline_backup = backup_file(baseline_target_json)
    fmkc_backup = backup_file(fmkc_target_json)

    print("=" * 80)
    print("Yousician union-level grid pipeline")
    print("=" * 80)
    print(f"repo_root: {repo_root}")
    print(f"examples_dir: {examples_dir}")
    print(f"source_dir: {source_dir}")
    print(f"output_root: {output_root}")
    print(f"union_levels: {union_levels}")
    print(f"models: {models}")
    print(f"folds: {folds}")
    print(f"baseline dataset: {args.baseline_dataset} / emb_type=qid")
    print(f"FMKC dataset: {args.fmkc_dataset} / emb_type=qid_fmkc")
    print(f"skip_preprocess: {args.skip_preprocess}")
    print(f"skip_train: {args.skip_train}")
    print(f"restore_original_json: {not args.no_restore_original_json}")
    print("=" * 80)

    rows: List[Dict[str, str]] = []

    try:
        # ========================================================
        # Union-level loop: union5 -> union1 by default
        # ========================================================
        for level in union_levels:
            tag = f"union{level}"
            run_dir = output_root / tag
            run_dir.mkdir(parents=True, exist_ok=True)

            log_file = run_dir / "pipeline.log"

            source_json = source_dir / args.source_json_pattern.format(level=level)

            print("\n" + "=" * 80)
            print(f"Running {tag}")
            print("=" * 80)

            row_base: Dict[str, str] = {
                "union_level": str(level),
                "union_file": source_json.name,
                "status": "ok",
                "error_stage": "",
                "error_message": "",
            }

            try:
                # ------------------------------------------------
                # Step 1: copy union JSON to both datasets
                # ------------------------------------------------
                print(f"[{tag}] Step 1: copy JSON to dataset directories")

                shutil.copy2(source_json, baseline_target_json)
                shutil.copy2(source_json, fmkc_target_json)

                with log_file.open("a", encoding="utf-8") as f:
                    f.write("\n[COPY UNION JSON]\n")
                    f.write(f"Copied {source_json} -> {baseline_target_json}\n")
                    f.write(f"Copied {source_json} -> {fmkc_target_json}\n")

                # ------------------------------------------------
                # Step 2: clear old cache
                # ------------------------------------------------
                print(f"[{tag}] Step 2: clear old .pkl cache")

                clear_pkl_files(baseline_dir, log_file)
                clear_pkl_files(fmkc_dir, log_file)

                # ------------------------------------------------
                # Step 3: preprocess
                # ------------------------------------------------
                if not args.skip_preprocess:
                    print(f"[{tag}] Step 3: run data_preprocess.py")

                    for ds_name in [args.baseline_dataset, args.fmkc_dataset]:
                        rc, out, err = run_cmd(
                            build_preprocess_cmd(
                                preprocess_script=preprocess_script,
                                dataset_name=ds_name,
                                preprocess_arg_name=args.preprocess_arg_name,
                            ),
                            cwd=examples_dir,
                            log_file=log_file,
                        )

                        if rc != 0:
                            raise RuntimeError(
                                f"data_preprocess.py failed for {ds_name} with exit code {rc}"
                            )
                else:
                    print(f"[{tag}] Step 3: preprocess skipped")

                # ------------------------------------------------
                # Step 4: train
                # ------------------------------------------------
                if args.skip_train:
                    skip_row = dict(row_base)
                    skip_row.update({
                        "status": "ok_skip_train",
                        "dataset_name": "",
                        "dataset_variant": "",
                        "model_name_requested": "",
                        "train_script": "",
                        "fold": "",
                        "embtype": "",
                    })
                    rows.append(skip_row)
                    continue

                train_jobs = [
                    {
                        "dataset_name": args.baseline_dataset,
                        "dataset_variant": "baseline_qid",
                        "emb_type": "qid",
                    },
                    {
                        "dataset_name": args.fmkc_dataset,
                        "dataset_variant": "fmkc",
                        "emb_type": "qid_fmkc",
                    },
                ]

                for model_name in models:
                    train_script = resolve_train_script(
                        examples_dir=examples_dir,
                        model_name=model_name,
                        train_script_dir=args.train_script_dir,
                    )

                    for train_job in train_jobs:
                        dataset_name = train_job["dataset_name"]
                        dataset_variant = train_job["dataset_variant"]
                        emb_type = train_job["emb_type"]

                        for fold in folds:
                            print(
                                f"[{tag}] Step 4: train "
                                f"model={model_name}, dataset={dataset_name}, "
                                f"emb_type={emb_type}, fold={fold}"
                            )

                            train_log_file = (
                                run_dir
                                / (
                                    f"train_{safe_filename(model_name)}_"
                                    f"{safe_filename(dataset_name)}_"
                                    f"{safe_filename(emb_type)}_"
                                    f"fold{fold}.log"
                                )
                            )

                            train_row = dict(row_base)
                            train_row.update({
                                "model_name_requested": model_name,
                                "train_script": train_script.name,
                                "dataset_name": dataset_name,
                                "dataset_variant": dataset_variant,
                                "embtype": emb_type,
                                "fold": str(fold),
                            })

                            try:
                                cmd = build_train_cmd(
                                    args=args,
                                    train_script=train_script,
                                    dataset_name=dataset_name,
                                    emb_type=emb_type,
                                    fold=fold,
                                )

                                rc, out, err = run_cmd(
                                    cmd,
                                    cwd=examples_dir,
                                    log_file=train_log_file,
                                )

                                if rc != 0:
                                    raise RuntimeError(
                                        f"{train_script.name} failed with exit code {rc}"
                                    )

                                metrics = parse_train_metrics(out + "\n" + err)

                                if metrics:
                                    train_row.update(metrics)
                                    train_row["status"] = "ok"
                                else:
                                    train_row["status"] = "ok_but_metrics_not_parsed"
                                    train_row["error_stage"] = "parse_metrics"
                                    train_row["error_message"] = (
                                        "Training finished, but metrics could not be parsed."
                                    )

                            except Exception as train_ex:
                                train_row["status"] = "failed"
                                train_row["error_stage"] = "train"
                                train_row["error_message"] = str(train_ex)

                                with train_log_file.open("a", encoding="utf-8") as f:
                                    f.write("\n[TRAIN ERROR]\n")
                                    f.write(str(train_ex) + "\n")

                            rows.append(train_row)

                # Write intermediate summary after each union level.
                write_summary(summary_csv, rows)

            except Exception as ex:
                failed_row = dict(row_base)
                failed_row["status"] = "failed"
                failed_row["error_stage"] = "pipeline"
                failed_row["error_message"] = str(ex)
                rows.append(failed_row)

                with log_file.open("a", encoding="utf-8") as f:
                    f.write("\n[PIPELINE ERROR]\n")
                    f.write(str(ex) + "\n")

                print(f"[{tag}] Failed: {ex}")

                # Still write partial summary.
                write_summary(summary_csv, rows)

        # ========================================================
        # Final summary
        # ========================================================
        write_summary(summary_csv, rows)

        print("\n" + "=" * 80)
        print("Pipeline finished.")
        print(f"Summary saved to: {summary_csv}")
        print("=" * 80)

        for r in rows:
            print(
                f"union={r.get('union_level', '')} "
                f"model={r.get('model_name_requested', '')} "
                f"dataset={r.get('dataset_name', '')} "
                f"variant={r.get('dataset_variant', '')} "
                f"fold={r.get('fold', '')} "
                f"status={r.get('status', '')} "
                f"testauc={r.get('testauc', '')} "
                f"testacc={r.get('testacc', '')} "
                f"validauc={r.get('validauc', '')} "
                f"validacc={r.get('validacc', '')} "
                f"error={r.get('error_message', '')}"
            )

    finally:
        # ========================================================
        # Restore original JSONs
        # ========================================================
        if not args.no_restore_original_json:
            print("\nRestoring original dataset JSON files...")

            restore_file_from_backup(baseline_target_json, baseline_backup)
            restore_file_from_backup(fmkc_target_json, fmkc_backup)

            print(f"Restored: {baseline_target_json}")
            print(f"Restored: {fmkc_target_json}")
        else:
            print("\nOriginal JSON restore skipped because --no_restore_original_json was set.")


if __name__ == "__main__":
    # Example:
    #
    # python .\yousician_union_grid_pipeline.py ^
    #   --union_levels 5,4,3,2,1 ^
    #   --models dkt,akt,dkt_plus,sakt,simplekt ^
    #   --folds 0
    #
    # If your data_preprocess.py only supports -d:
    #
    # python .\yousician_union_grid_pipeline.py ^
    #   --preprocess_arg_name -d ^
    #   --union_levels 5,4,3,2,1 ^
    #   --models dkt,akt,dkt_plus,sakt,simplekt ^
    #   --folds 0
    #
    main()