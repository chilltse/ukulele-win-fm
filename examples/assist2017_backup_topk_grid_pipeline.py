import argparse
import csv
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
    Parse the final row printed by wandb_dkt_train.py / wandb_train.py.

    Expected format is usually:
        fold modelname embtype testauc testacc window_testauc window_testacc validauc validacc best_epoch

    Example:
        0 dkt qid_tree 0.7234 0.6812 0.7123 0.6744 0.7301 0.6902 18
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for ln in reversed(lines):
        parts = ln.split()

        if len(parts) >= 10 and parts[0].isdigit():
            # Usually parts[1] should be dkt.
            # We keep this check loose in case the script prints model names differently.
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

    return {}


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

    This is very important because each topK run overwrites the same CSV file.
    If old .pkl files remain, training may accidentally use old cached data.
    """
    if not dataset_dir.exists():
        return

    deleted = []

    for pattern in ["*.pkl", "*.pkl.*"]:
        for p in dataset_dir.glob(pattern):
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
    Backup original dataset CSV before overwriting it.
    """
    if not path.exists():
        return None

    backup_path = path.with_name(path.name + ".backup_before_topk_grid")

    if not backup_path.exists():
        shutil.copy2(path, backup_path)

    return backup_path


def restore_file_from_backup(target_path: Path, backup_path: Optional[Path]) -> None:
    """
    Restore dataset CSV from backup.
    """
    if backup_path is None:
        return

    if backup_path.exists():
        shutil.copy2(backup_path, target_path)


def parse_folds(raw_folds: List[str]) -> List[int]:
    """
    Support both:
        --folds 0 1 2 3 4
    and:
        --folds 0,1,2,3,4
    """
    folds: List[int] = []

    for item in raw_folds:
        for part in item.split(","):
            part = part.strip()
            if part:
                folds.append(int(part))

    return folds


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-run assist2017 top-k KC coverage split/preprocess/train pipeline."
    )

    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=[10, 20, 50, 100, 200],
        help="Grid of top-k student counts.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed.",
    )

    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0"],
        help="Training folds. Example: --folds 0 1 2 3 4 or --folds 0,1,2,3,4",
    )

    parser.add_argument(
        "--emb_type",
        type=str,
        default="qid_tree",
        help="Embedding type for wandb_dkt_train.py.",
    )

    parser.add_argument(
        "--run_preprocess_for_assist2017",
        action="store_true",
        help="Also run data_preprocess.py for assist2017. Default only runs assist2017_tree.",
    )

    parser.add_argument(
        "--also_train_assist2017_qid",
        action="store_true",
        help="Also train baseline assist2017 + qid.",
    )

    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training; only run split/filter/copy/preprocess.",
    )

    parser.add_argument(
        "--no_restore_original_csv",
        action="store_true",
        help="Do not restore original dataset CSV files at the end.",
    )

    args = parser.parse_args()

    folds = parse_folds(args.folds)

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"

    data_root = repo_root / "data"
    assist2017_dir = data_root / "assist2017"
    assist2017_tree_dir = data_root / "assist2017_tree"
    ds_split_dir = assist2017_tree_dir / "ds_split"

    split_script = ds_split_dir / "1_split_by_kc_coverage.py"
    filter_script = ds_split_dir / "2_filter_csv_by_group_students.py"
    preprocess_script = examples_dir / "data_preprocess.py"
    train_script = examples_dir / "wandb_dkt_train.py"

    source_txt = assist2017_tree_dir / "backup_ds_files" / "data.txt"
    source_csv = (
        assist2017_tree_dir
        / "backup_ds_files"
        / "original_anonymized_full_release_competition_dataset.csv"
    )

    target_csv_name = "anonymized_full_release_competition_dataset.csv"

    assist2017_csv = assist2017_dir / target_csv_name
    assist2017_tree_csv = assist2017_tree_dir / target_csv_name

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = ds_split_dir / "grid_runs" / f"run_{run_stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    summary_csv = output_root / "summary.csv"

    # ------------------------------------------------------------
    # Pre-check
    # ------------------------------------------------------------
    assert_dir_exists(repo_root, "Repository root not found")
    assert_dir_exists(examples_dir, "examples directory not found")
    assert_dir_exists(ds_split_dir, "ds_split directory not found")

    assert_file_exists(split_script, "Split script not found")
    assert_file_exists(filter_script, "Filter script not found")
    assert_file_exists(preprocess_script, "Preprocess script not found")
    assert_file_exists(train_script, "Train script not found")
    assert_file_exists(source_txt, "Source data.txt not found")
    assert_file_exists(source_csv, "Original CSV not found")

    ensure_parent(assist2017_csv)
    ensure_parent(assist2017_tree_csv)

    # ------------------------------------------------------------
    # Backup original CSVs
    # ------------------------------------------------------------
    assist2017_backup = backup_file(assist2017_csv)
    assist2017_tree_backup = backup_file(assist2017_tree_csv)

    print("=" * 80)
    print("assist2017 topK grid pipeline")
    print("=" * 80)
    print(f"repo_root: {repo_root}")
    print(f"output_root: {output_root}")
    print(f"topk: {args.topk}")
    print(f"folds: {folds}")
    print(f"emb_type: {args.emb_type}")
    print(f"skip_train: {args.skip_train}")
    print(f"restore_original_csv: {not args.no_restore_original_csv}")
    print("=" * 80)

    rows: List[Dict[str, str]] = []

    try:
        # ========================================================
        # TopK loop
        # ========================================================
        for k in args.topk:
            tag = f"top{k}"
            run_dir = output_root / tag
            run_dir.mkdir(parents=True, exist_ok=True)

            log_file = run_dir / "pipeline.log"

            print("\n" + "=" * 80)
            print(f"Running {tag}")
            print("=" * 80)

            row_base: Dict[str, str] = {
                "topk": str(k),
                "status": "ok",
                "error_stage": "",
                "error_message": "",
            }

            try:
                # ------------------------------------------------
                # Step 1: split top-K students by max KC coverage
                # ------------------------------------------------
                out_dir = run_dir / f"size_{k}"
                out_dir.mkdir(parents=True, exist_ok=True)

                print(f"[{tag}] Step 1: split by KC coverage")

                rc, out, err = run_cmd(
                    [
                        sys.executable,
                        str(split_script),
                        "--input",
                        str(source_txt),
                        "--out_dir",
                        str(out_dir),
                        "--group_size",
                        str(k),
                    ],
                    cwd=repo_root,
                    log_file=log_file,
                )

                if rc != 0:
                    raise RuntimeError(f"split_by_kc_coverage failed with exit code {rc}")

                group_summary_csv = out_dir / "group_summary.csv"
                assert_file_exists(
                    group_summary_csv,
                    "group_summary.csv was not created by split script",
                )

                # ------------------------------------------------
                # Step 2: filter original CSV by selected group
                # ------------------------------------------------
                print(f"[{tag}] Step 2: filter original CSV")

                filtered_csv = out_dir / "max_kc_coverage_students.csv"

                if filtered_csv.exists():
                    filtered_csv.unlink()

                rc, out, err = run_cmd(
                    [
                        sys.executable,
                        str(filter_script),
                        "--csv",
                        str(source_csv),
                        "--group_summary",
                        str(group_summary_csv),
                        "--out",
                        str(filtered_csv),
                        "--group",
                        "max_kc_coverage",
                    ],
                    cwd=repo_root,
                    log_file=log_file,
                )

                if rc != 0:
                    raise RuntimeError(f"filter_csv_by_group_students failed with exit code {rc}")

                assert_file_exists(
                    filtered_csv,
                    "Filtered CSV was not created by filter script",
                )

                # ------------------------------------------------
                # Step 3: overwrite dataset CSVs
                # ------------------------------------------------
                print(f"[{tag}] Step 3: copy filtered CSV to assist2017 and assist2017_tree")

                shutil.copy2(filtered_csv, assist2017_csv)
                shutil.copy2(filtered_csv, assist2017_tree_csv)

                with log_file.open("a", encoding="utf-8") as f:
                    f.write("\n[COPY FILTERED CSV]\n")
                    f.write(f"Copied {filtered_csv} -> {assist2017_csv}\n")
                    f.write(f"Copied {filtered_csv} -> {assist2017_tree_csv}\n")

                # ------------------------------------------------
                # Step 4: clear old pkl cache
                # ------------------------------------------------
                print(f"[{tag}] Step 4: clear old .pkl cache")

                clear_pkl_files(assist2017_dir, log_file)
                clear_pkl_files(assist2017_tree_dir, log_file)

                # ------------------------------------------------
                # Step 5: preprocess
                # ------------------------------------------------
                print(f"[{tag}] Step 5: run data_preprocess.py")

                preprocess_datasets = ["assist2017_tree"]

                if args.run_preprocess_for_assist2017 or args.also_train_assist2017_qid:
                    preprocess_datasets.insert(0, "assist2017")

                for ds_name in preprocess_datasets:
                    rc, out, err = run_cmd(
                        [
                            sys.executable,
                            str(preprocess_script),
                            "-d",
                            ds_name,
                        ],
                        cwd=examples_dir,
                        log_file=log_file,
                    )

                    if rc != 0:
                        raise RuntimeError(
                            f"data_preprocess.py failed for {ds_name} with exit code {rc}"
                        )

                # ------------------------------------------------
                # Step 6: train
                # ------------------------------------------------
                if args.skip_train:
                    skip_row = dict(row_base)
                    skip_row.update({
                        "dataset_name": "",
                        "embtype": "",
                        "fold": "",
                        "status": "ok_skip_train",
                    })
                    rows.append(skip_row)
                    continue

                train_jobs = [
                    {
                        "dataset_name": "assist2017_tree",
                        "emb_type": args.emb_type,
                    }
                ]

                if args.also_train_assist2017_qid:
                    train_jobs.insert(
                        0,
                        {
                            "dataset_name": "assist2017",
                            "emb_type": "qid",
                        }
                    )

                for train_job in train_jobs:
                    dataset_name = train_job["dataset_name"]
                    emb_type = train_job["emb_type"]

                    for fold in folds:
                        print(
                            f"[{tag}] Step 6: train dataset={dataset_name}, "
                            f"emb_type={emb_type}, fold={fold}"
                        )

                        train_log_file = run_dir / f"train_{dataset_name}_{emb_type}_fold{fold}.log"

                        train_row = dict(row_base)
                        train_row.update({
                            "dataset_name": dataset_name,
                            "embtype": emb_type,
                            "fold": str(fold),
                        })

                        try:
                            rc, out, err = run_cmd(
                                [
                                    sys.executable,
                                    str(train_script),
                                    "--dataset_name",
                                    dataset_name,
                                    "--model_name",
                                    "dkt",
                                    "--emb_type",
                                    emb_type,
                                    "--seed",
                                    str(args.seed),
                                    "--fold",
                                    str(fold),
                                ],
                                cwd=examples_dir,
                                log_file=train_log_file,
                            )

                            if rc != 0:
                                raise RuntimeError(
                                    f"wandb_dkt_train.py failed with exit code {rc}"
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

        # ========================================================
        # Write summary
        # ========================================================
        fieldnames = [
            "topk",
            "dataset_name",
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
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        print("\n" + "=" * 80)
        print("Pipeline finished.")
        print(f"Summary saved to: {summary_csv}")
        print("=" * 80)

        for r in rows:
            print(
                f"topk={r.get('topk', '')} "
                f"dataset={r.get('dataset_name', '')} "
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
        # Restore original CSVs
        # ========================================================
        if not args.no_restore_original_csv:
            print("\nRestoring original dataset CSV files...")

            restore_file_from_backup(assist2017_csv, assist2017_backup)
            restore_file_from_backup(assist2017_tree_csv, assist2017_tree_backup)

            print(f"Restored: {assist2017_csv}")
            print(f"Restored: {assist2017_tree_csv}")
        else:
            print("\nOriginal CSV restore skipped because --no_restore_original_csv was set.")


if __name__ == "__main__":
    main()