# python .\run_group_dkvmn_pipeline.py --groups_dir ..\data\assist2017_tree\ds_split\question_coverage_5groups\filtered_csv --target_ds ..\data\assist2017_tree --examples_dir . --mode grid
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


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_")


def parse_size_ms(size_ms_str: str) -> List[int]:
    values = []
    for x in size_ms_str.split(","):
        x = x.strip()
        if not x:
            continue
        values.append(int(x))
    if not values:
        raise ValueError("--size_ms cannot be empty.")
    return values


def find_group_csvs(groups_dir: Path, pattern: str) -> List[Path]:
    files = sorted(groups_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No group CSV files found in {groups_dir} with pattern '{pattern}'."
        )
    return files


def build_run_plan(
    group_files: List[Path],
    size_ms: List[int],
    mode: str,
) -> List[Tuple[Path, int]]:
    """
    mode='paired':
        group_01 -> size_m[0], group_02 -> size_m[1], ...
        This gives 5 runs if there are 5 groups and 5 size_m values.

    mode='grid':
        every group is run with every size_m.
        This gives 25 runs if there are 5 groups and 5 size_m values.
    """
    if mode == "paired":
        if len(group_files) != len(size_ms):
            raise ValueError(
                "In paired mode, number of group files must equal number of size_m values.\n"
                f"Found group_files={len(group_files)}, size_ms={len(size_ms)}.\n"
                "Use --mode grid if you want every group to run with every size_m."
            )
        return list(zip(group_files, size_ms))

    if mode == "grid":
        return [(group_file, size_m) for group_file in group_files for size_m in size_ms]

    raise ValueError(f"Unknown mode: {mode}")


def copy_group_to_target(
    group_csv: Path,
    target_ds: Path,
    target_filename: str,
) -> Path:
    target_csv = target_ds / target_filename
    target_ds.mkdir(parents=True, exist_ok=True)
    shutil.copy2(group_csv, target_csv)
    return target_csv


def replace_group_csv_in_target(
    group_csv: Path,
    target_ds: Path,
    target_filename: str,
) -> Tuple[Path, bool]:
    """
    Replace target CSV with the given group CSV.
    Returns:
        (target_csv_path, deleted_original_before_copy)
    """
    target_csv = target_ds / target_filename
    target_ds.mkdir(parents=True, exist_ok=True)
    deleted_original = False
    if target_csv.exists():
        target_csv.unlink()
        deleted_original = True
    shutil.copy2(group_csv, target_csv)
    return target_csv, deleted_original


def backup_original_target_csv(
    target_ds: Path,
    target_filename: str,
    log_dir: Path,
) -> Optional[Path]:
    target_csv = target_ds / target_filename
    if not target_csv.exists():
        return None

    backup_dir = log_dir / "backup_original_dataset"
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_path = backup_dir / target_filename
    shutil.copy2(target_csv, backup_path)
    return backup_path


def delete_pkl_files(target_ds: Path, recursive: bool = True) -> List[Path]:
    pattern_iter = target_ds.rglob("*.pkl") if recursive else target_ds.glob("*.pkl")
    deleted = []

    for pkl_path in sorted(pattern_iter):
        if pkl_path.is_file():
            pkl_path.unlink()
            deleted.append(pkl_path)

    return deleted


def run_command_live(
    cmd: List[str],
    cwd: Path,
    log_file: Path,
    dry_run: bool = False,
) -> Tuple[int, str]:
    """
    Run command and stream output to both console and log file.
    Return (returncode, combined_output_text).
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("w", encoding="utf-8", errors="replace") as f:
        f.write("=" * 100 + "\n")
        f.write(f"Start time: {now_str()}\n")
        f.write(f"CWD: {cwd}\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.write("=" * 100 + "\n\n")

        if dry_run:
            f.write("[DRY RUN] Command not executed.\n")
            return 0, "[DRY RUN] Command not executed."

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        output_parts = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            output_parts.append(line)

        proc.wait()
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"End time: {now_str()}\n")
        f.write(f"Return code: {proc.returncode}\n")

    return proc.returncode, "".join(output_parts)


# ============================================================
# Metric parsing
# ============================================================


def _last_float_pair(text: str, pattern: str) -> Tuple[Optional[float], Optional[float]]:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None, None

    for a, b in reversed(matches):
        try:
            fa, fb = float(a), float(b)
            return fa, fb
        except ValueError:
            continue

    return None, None


def _last_non_negative_float_pair(text: str, pattern: str) -> Tuple[Optional[float], Optional[float]]:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None, None

    for a, b in reversed(matches):
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            continue
        if fa >= 0 and fb >= 0:
            return fa, fb

    return None, None


def extract_metrics(output_text: str) -> Dict[str, Optional[float]]:
    """
    Extract common pyKT metrics from train logs.

    Supported examples:
        validauc: 0.5412, validacc: 0.6461
        best auc: 0.5412
        testauc: 0.5321, testacc: 0.6210
        window_testauc: 0.5333, window_testacc: 0.6201

    If testauc/testacc are -1, this function still records them as last_testauc_raw,
    but preferred_auc/preferred_acc falls back to window_testauc or validauc.
    """
    validauc, validacc = _last_float_pair(
        output_text,
        r"validauc\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*validacc\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    )

    testauc_raw, testacc_raw = _last_float_pair(
        output_text,
        r"testauc\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*testacc\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    )

    testauc, testacc = _last_non_negative_float_pair(
        output_text,
        r"testauc\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*testacc\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    )

    window_testauc, window_testacc = _last_non_negative_float_pair(
        output_text,
        r"window_testauc\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*window_testacc\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    )

    best_validauc = None
    best_matches = re.findall(
        r"best\s+auc\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        output_text,
        flags=re.IGNORECASE,
    )
    if best_matches:
        try:
            best_validauc = float(best_matches[-1])
        except ValueError:
            best_validauc = None

    preferred_auc = testauc
    preferred_acc = testacc
    preferred_source = "test"

    if preferred_auc is None or preferred_acc is None:
        preferred_auc = window_testauc
        preferred_acc = window_testacc
        preferred_source = "window_test"

    if preferred_auc is None or preferred_acc is None:
        preferred_auc = validauc
        preferred_acc = validacc
        preferred_source = "valid"

    return {
        "preferred_auc": preferred_auc,
        "preferred_acc": preferred_acc,
        "preferred_metric_source": preferred_source if preferred_auc is not None else None,
        "validauc": validauc,
        "validacc": validacc,
        "best_validauc": best_validauc,
        "testauc": testauc,
        "testacc": testacc,
        "last_testauc_raw": testauc_raw,
        "last_testacc_raw": testacc_raw,
        "window_testauc": window_testauc,
        "window_testacc": window_testacc,
    }


# ============================================================
# Main pipeline
# ============================================================


def write_results_csv(results: List[Dict[str, object]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id",
        "group_name",
        "group_csv",
        "size_m",
        "dataset_name",
        "emb_type",
        "target_csv",
        "deleted_original_csv",
        "deleted_pkl_count",
        "preprocess_returncode",
        "preprocess_log_file",
        "returncode",
        "success",
        "preferred_auc",
        "preferred_acc",
        "preferred_metric_source",
        "validauc",
        "validacc",
        "best_validauc",
        "testauc",
        "testacc",
        "last_testauc_raw",
        "last_testacc_raw",
        "window_testauc",
        "window_testacc",
        "log_file",
        "command",
        "start_time",
        "end_time",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy each generated group CSV into a target dataset directory, "
            "replace anonymized_full_release_competition_dataset.csv, delete pkl caches, "
            "run data_preprocess.py and wandb_dkvmn_train.py with different size_m values, "
            "and collect AUC/ACC."
        )
    )

    parser.add_argument(
        "--groups_dir",
        required=True,
        help="Directory containing generated group CSV files, e.g. question_coverage_5groups/filtered_csv.",
    )
    parser.add_argument(
        "--target_ds",
        required=True,
        help="Target dataset directory, e.g. ../data/assist2017_tree.",
    )
    parser.add_argument(
        "--examples_dir",
        default=".",
        help="Directory where wandb_dkvmn_train.py is located. Default: current directory.",
    )
    parser.add_argument(
        "--group_pattern",
        default="group_*_qcov.csv",
        help="Glob pattern for group CSVs. Default: group_*_qcov.csv.",
    )
    parser.add_argument(
        "--target_filename",
        default="anonymized_full_release_competition_dataset.csv",
        help="Filename to replace under target_ds.",
    )
    parser.add_argument(
        "--train_script",
        default="wandb_dkvmn_train.py",
        help="Training script name. Default: wandb_dkvmn_train.py.",
    )
    parser.add_argument(
        "--preprocess_script",
        default="data_preprocess.py",
        help="Preprocess script name. Default: data_preprocess.py.",
    )
    parser.add_argument(
        "--python_cmd",
        default=sys.executable,
        help="Python command to use. Default: current Python executable.",
    )
    parser.add_argument(
        "--dataset_name",
        default="assist2017_tree",
        help="Dataset name passed to training script.",
    )
    parser.add_argument(
        "--emb_type",
        default="qid",
        help="Embedding type passed to training script.",
    )
    parser.add_argument(
        "--size_ms",
        default="10,20,30,40,50",
        help="Comma-separated size_m values. Default: 10,20,30,40,50.",
    )
    parser.add_argument(
        "--mode",
        choices=["paired", "grid"],
        default="grid",
        help=(
            "paired: group_01->10, group_02->20, ...; "
            "grid: every group runs every size_m. Default: grid."
        ),
    )
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Directory for logs and result CSV. Default: target_ds/run_logs/dkvmn_group_runs_<timestamp>.",
    )
    parser.add_argument(
        "--no_recursive_pkl_delete",
        action="store_true",
        help="Only delete *.pkl directly under target_ds instead of recursively.",
    )
    parser.add_argument(
        "--no_backup_original",
        action="store_true",
        help="Do not backup original target CSV before the first replacement.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print/capture planned commands without actually copying/deleting/running.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue remaining runs even if one training command fails.",
    )
    parser.add_argument(
        "--extra_train_args",
        nargs=argparse.REMAINDER,
        help=(
            "Extra arguments passed to training script. Put this at the end. "
            "Example: --extra_train_args --seed 42 --fold 0"
        ),
    )

    args = parser.parse_args()

    groups_dir = Path(args.groups_dir).resolve()
    target_ds = Path(args.target_ds).resolve()
    examples_dir = Path(args.examples_dir).resolve()
    train_script_path = examples_dir / args.train_script
    preprocess_script_path = examples_dir / args.preprocess_script
    size_ms = parse_size_ms(args.size_ms)

    if not groups_dir.exists():
        raise FileNotFoundError(f"groups_dir does not exist: {groups_dir}")
    if not target_ds.exists():
        raise FileNotFoundError(f"target_ds does not exist: {target_ds}")
    if not examples_dir.exists():
        raise FileNotFoundError(f"examples_dir does not exist: {examples_dir}")
    if not train_script_path.exists():
        raise FileNotFoundError(f"train script not found: {train_script_path}")
    if not preprocess_script_path.exists():
        raise FileNotFoundError(f"preprocess script not found: {preprocess_script_path}")

    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = target_ds / "run_logs" / f"dkvmn_group_runs_{timestamp}"
    else:
        log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    group_files = find_group_csvs(groups_dir, args.group_pattern)
    run_plan = build_run_plan(group_files, size_ms, args.mode)

    print("========== Pipeline Config ==========")
    print(f"groups_dir:       {groups_dir}")
    print(f"target_ds:        {target_ds}")
    print(f"examples_dir:     {examples_dir}")
    print(f"train_script:     {train_script_path}")
    print(f"preprocess_script:{preprocess_script_path}")
    print(f"target_filename:  {args.target_filename}")
    print(f"group_pattern:    {args.group_pattern}")
    print(f"found_groups:     {len(group_files)}")
    print(f"size_ms:          {size_ms}")
    print(f"mode:             {args.mode}")
    print(f"total_runs:       {len(run_plan)}")
    print(f"log_dir:          {log_dir}")
    print()

    if not args.no_backup_original and not args.dry_run:
        backup_path = backup_original_target_csv(target_ds, args.target_filename, log_dir)
        if backup_path is not None:
            print(f"[Backup] Original target CSV backed up to: {backup_path}")
        else:
            print("[Backup] No existing target CSV found. Backup skipped.")
        print()

    results: List[Dict[str, object]] = []
    results_csv = log_dir / "dkvmn_group_size_m_results.csv"

    extra_train_args = args.extra_train_args or []

    for run_idx, (group_csv, size_m) in enumerate(run_plan, start=1):
        group_name = safe_name(group_csv)
        run_id = f"run_{run_idx:02d}_{group_name}_size_m_{size_m}"
        log_file = log_dir / f"{run_id}.log"
        start_time = now_str()

        print("\n" + "=" * 100)
        print(f"[{run_idx}/{len(run_plan)}] {run_id}")
        print(f"Group CSV: {group_csv}")
        print(f"size_m:    {size_m}")
        print("=" * 100)

        target_csv = target_ds / args.target_filename
        deleted_pkl_count = 0
        deleted_original_csv = False
        preprocess_returncode = -1
        preprocess_log_file = log_dir / f"{run_id}.preprocess.log"
        returncode = -1
        output_text = ""

        try:
            if args.dry_run:
                print(f"[DRY RUN] Would delete existing target CSV if present: {target_csv}")
                print(f"[DRY RUN] Would copy {group_csv} -> {target_csv}")
                print(f"[DRY RUN] Would delete pkl files under {target_ds}")
                print(f"[DRY RUN] Would run preprocess: {preprocess_script_path} --dataset_name {args.dataset_name}")
            else:
                target_csv, deleted_original_csv = replace_group_csv_in_target(
                    group_csv, target_ds, args.target_filename
                )
                if deleted_original_csv:
                    print(f"[Delete CSV] removed old target CSV: {target_csv}")
                print(f"[Copy] {group_csv} -> {target_csv}")

                deleted_pkl = delete_pkl_files(
                    target_ds,
                    recursive=not args.no_recursive_pkl_delete,
                )
                deleted_pkl_count = len(deleted_pkl)
                print(f"[Delete PKL] deleted {deleted_pkl_count} .pkl files")

            preprocess_cmd = [
                args.python_cmd,
                str(preprocess_script_path),
                "--dataset_name",
                args.dataset_name,
            ]
            print("[Preprocess] " + " ".join(preprocess_cmd))
            preprocess_returncode, _ = run_command_live(
                cmd=preprocess_cmd,
                cwd=examples_dir,
                log_file=preprocess_log_file,
                dry_run=args.dry_run,
            )
            if preprocess_returncode != 0:
                raise RuntimeError(f"Preprocess failed with return code {preprocess_returncode}")

            cmd = [
                args.python_cmd,
                str(train_script_path),
                "--dataset_name", args.dataset_name,
                "--emb_type", args.emb_type,
                "--dkvmn_use_question", "1",
                "--size_m", str(size_m),
            ]
            cmd.extend(extra_train_args)

            print("[Run] " + " ".join(cmd))
            returncode, output_text = run_command_live(
                cmd=cmd,
                cwd=examples_dir,
                log_file=log_file,
                dry_run=args.dry_run,
            )

            metrics = extract_metrics(output_text)
            success = returncode == 0

            end_time = now_str()
            row: Dict[str, object] = {
                "run_id": run_id,
                "group_name": group_name,
                "group_csv": str(group_csv),
                "size_m": size_m,
                "dataset_name": args.dataset_name,
                "emb_type": args.emb_type,
                "target_csv": str(target_csv),
                "deleted_original_csv": deleted_original_csv,
                "deleted_pkl_count": deleted_pkl_count,
                "preprocess_returncode": preprocess_returncode,
                "preprocess_log_file": str(preprocess_log_file),
                "returncode": returncode,
                "success": success,
                "log_file": str(log_file),
                "command": " ".join(cmd),
                "start_time": start_time,
                "end_time": end_time,
            }
            row.update(metrics)
            results.append(row)
            write_results_csv(results, results_csv)

            print("\n[Metrics]")
            print(f"preferred_auc:  {metrics.get('preferred_auc')}")
            print(f"preferred_acc:  {metrics.get('preferred_acc')}")
            print(f"metric_source:  {metrics.get('preferred_metric_source')}")
            print(f"results_csv:    {results_csv}")

            if returncode != 0 and not args.continue_on_error:
                print("\n[Stop] Training command failed. Use --continue_on_error to keep running remaining groups.")
                break

        except Exception as e:
            end_time = now_str()
            print(f"[Error] {run_id}: {e}")

            row = {
                "run_id": run_id,
                "group_name": group_name,
                "group_csv": str(group_csv),
                "size_m": size_m,
                "dataset_name": args.dataset_name,
                "emb_type": args.emb_type,
                "target_csv": str(target_csv),
                "deleted_original_csv": deleted_original_csv,
                "deleted_pkl_count": deleted_pkl_count,
                "preprocess_returncode": preprocess_returncode,
                "preprocess_log_file": str(preprocess_log_file),
                "returncode": returncode,
                "success": False,
                "preferred_auc": None,
                "preferred_acc": None,
                "preferred_metric_source": None,
                "validauc": None,
                "validacc": None,
                "best_validauc": None,
                "testauc": None,
                "testacc": None,
                "last_testauc_raw": None,
                "last_testacc_raw": None,
                "window_testauc": None,
                "window_testacc": None,
                "log_file": str(log_file),
                "command": "",
                "start_time": start_time,
                "end_time": end_time,
            }
            results.append(row)
            write_results_csv(results, results_csv)

            if not args.continue_on_error:
                raise

    print("\n========== Finished ==========")
    print(f"Result CSV: {results_csv}")

    if results:
        print("\n========== Summary ==========")
        for row in results:
            print(
                f"{row.get('run_id')}: "
                f"auc={row.get('preferred_auc')}, "
                f"acc={row.get('preferred_acc')}, "
                f"source={row.get('preferred_metric_source')}, "
                f"success={row.get('success')}"
            )


if __name__ == "__main__":
    main()
