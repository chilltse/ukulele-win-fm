#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract all questions and image references for candidate KCs.

Workflow:
1) Read candidate CSV, take values from --kc-col, usually `kc_name`.
2) Treat each value as a key in decoded_kc_routes_map.json, and get its KC leaf name.
3) Search decoded_questions.json for questions whose kc_routes leaf equals that KC leaf name.
4) Export full question information and all image tokens such as `question_5464-image_0`.

Example:
python extract_kc_candidate_questions_images.py \
  --candidate-csv xes3g5m_tree_manual_split_dkt_qid_fold0_kc_question_heterogeneity_over_coarse_candidates.csv \
  --kc-map-json decoded_kc_routes_map.json \
  --questions-json decoded_questions.json \
  --output-dir out_candidate_questions
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


IMAGE_TOKEN_RE = re.compile(r"(?<![\w-])([A-Za-z]+_\d+-image_\d+)(?![\w-])")
DEFAULT_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_kc_key(value: Any) -> str:
    """Convert CSV values like 402, 402.0, '402' into the same JSON key string '402'."""
    if value is None:
        return ""

    # pandas missing values
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if isinstance(value, bool):
        return str(value).strip()

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    s = str(value).strip()
    # handle strings like '402.0'
    if re.fullmatch(r"\d+\.0", s):
        return s[:-2]
    return s


def route_leaf(route: Any, sep: str = "----", strip: bool = True) -> str:
    if route is None:
        return ""
    leaf = str(route).split(sep)[-1]
    return leaf.strip() if strip else leaf


def normalize_leaf_name(name: Any, strip: bool = True) -> str:
    if name is None:
        return ""
    s = str(name)
    return s.strip() if strip else s


def iter_text_fields(obj: Any, prefix: str = "") -> Iterable[Tuple[str, str]]:
    """Yield all string fields recursively with their JSON path."""
    if isinstance(obj, str):
        yield prefix or "$", obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            yield from iter_text_fields(v, child_prefix)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            child_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from iter_text_fields(v, child_prefix)


def extract_image_refs(question_obj: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract unique image tokens from all string fields: content, analysis, options, etc."""
    refs: List[Dict[str, str]] = []
    seen = set()
    for field_path, text in iter_text_fields(question_obj):
        for m in IMAGE_TOKEN_RE.finditer(text):
            token = m.group(1)
            key = (token, field_path)
            if key not in seen:
                seen.add(key)
                refs.append({"image_token": token, "source_field": field_path})
    return refs


def find_image_file(image_root: Optional[Path], token: str, exts: List[str]) -> Optional[Path]:
    """Try to locate token image under image_root. Returns first matching file if found."""
    if image_root is None:
        return None
    if not image_root.exists():
        return None

    # First try direct child names, then recursive search.
    for ext in exts:
        p = image_root / f"{token}{ext}"
        if p.exists():
            return p

    # Recursive search can be slower, but useful for metadata/images-like directories.
    for ext in exts:
        matches = list(image_root.rglob(f"{token}{ext}"))
        if matches:
            return matches[0]
    return None


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract candidate KC question details and image references from decoded XES3G5M files."
    )
    parser.add_argument("--candidate-csv", required=True, type=Path,
                        help="CSV containing candidate KCs, e.g. over_coarse_candidates.csv")
    parser.add_argument("--kc-map-json", required=True, type=Path,
                        help="decoded_kc_routes_map.json; keys are KC ids, values are decoded KC leaf names")
    parser.add_argument("--questions-json", required=True, type=Path,
                        help="decoded_questions.json; question_id -> full question object")
    parser.add_argument("--output-dir", default=Path("kc_candidate_question_outputs"), type=Path,
                        help="Output directory")
    parser.add_argument("--kc-col", default="kc_name",
                        help="Column in candidate CSV used as key into kc-map-json. Default: kc_name")
    parser.add_argument("--route-sep", default="----",
                        help="Separator used in kc_routes. Default: ----")
    parser.add_argument("--no-strip-leaf", action="store_true",
                        help="Do not strip whitespace when comparing route leaf names. Default: strip whitespace")
    parser.add_argument("--image-root", default=None, type=Path,
                        help="Optional local directory containing image files. If given, script will try to resolve image_token to real file path")
    parser.add_argument("--copy-images", action="store_true",
                        help="If --image-root is provided, copy found images into output-dir/images")
    parser.add_argument("--image-exts", default=",".join(DEFAULT_IMAGE_EXTS),
                        help="Comma-separated image extensions to search. Default: common image extensions")
    args = parser.parse_args()

    strip_leaf = not args.no_strip_leaf
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_df = pd.read_csv(args.candidate_csv)
    if args.kc_col not in candidate_df.columns:
        raise ValueError(
            f"Column {args.kc_col!r} not found in candidate CSV. "
            f"Available columns: {list(candidate_df.columns)}"
        )

    kc_map: Dict[str, Any] = read_json(args.kc_map_json)
    questions: Dict[str, Dict[str, Any]] = read_json(args.questions_json)
    image_exts = [ext.strip() for ext in args.image_exts.split(",") if ext.strip()]
    image_exts = [ext if ext.startswith(".") else "." + ext for ext in image_exts]

    # Build leaf -> questions index once. This is much faster than scanning all questions for every KC.
    leaf_to_question_ids: Dict[str, List[str]] = {}
    question_leaf_routes: Dict[str, Dict[str, List[str]]] = {}

    for qid, qobj in questions.items():
        routes = qobj.get("kc_routes", [])
        if not isinstance(routes, list):
            routes = []
        q_leaf_map: Dict[str, List[str]] = {}
        for route in routes:
            leaf = normalize_leaf_name(route_leaf(route, sep=args.route_sep, strip=strip_leaf), strip=strip_leaf)
            if not leaf:
                continue
            q_leaf_map.setdefault(leaf, []).append(str(route))
            leaf_to_question_ids.setdefault(leaf, []).append(str(qid))
        question_leaf_routes[str(qid)] = q_leaf_map

    # Preserve CSV order, but de-duplicate target KC keys.
    candidate_records = candidate_df.to_dict(orient="records")
    target_keys: List[str] = []
    key_to_csv_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in candidate_records:
        key = normalize_kc_key(row.get(args.kc_col))
        if not key:
            continue
        if key not in key_to_csv_rows:
            target_keys.append(key)
        # Convert NaN to None for JSON cleanliness.
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                clean_row[k] = None
            else:
                clean_row[k] = v
        key_to_csv_rows.setdefault(key, []).append(clean_row)

    result: Dict[str, Any] = {
        "metadata": {
            "candidate_csv": str(args.candidate_csv),
            "kc_map_json": str(args.kc_map_json),
            "questions_json": str(args.questions_json),
            "kc_col": args.kc_col,
            "route_sep": args.route_sep,
            "strip_leaf_for_matching": strip_leaf,
        },
        "targets": {},
        "unresolved_kc_keys": [],
    }

    flat_rows: List[Dict[str, Any]] = []
    image_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    copied_tokens = set()
    image_out_dir = out_dir / "images"
    if args.copy_images:
        image_out_dir.mkdir(parents=True, exist_ok=True)

    for key in target_keys:
        raw_kc_value = kc_map.get(key)

        # Fallback: if CSV already contains the decoded KC name instead of the key.
        if raw_kc_value is None:
            fallback_name = key
            target_leaf = normalize_leaf_name(fallback_name, strip=strip_leaf)
            result["unresolved_kc_keys"].append(key)
            resolved_from_map = False
        else:
            target_leaf = normalize_leaf_name(raw_kc_value, strip=strip_leaf)
            resolved_from_map = True

        matched_qids = sorted(set(leaf_to_question_ids.get(target_leaf, [])), key=lambda x: int(x) if x.isdigit() else x)

        target_obj: Dict[str, Any] = {
            "kc_key": key,
            "kc_name": raw_kc_value if raw_kc_value is not None else key,
            "normalized_kc_name": target_leaf,
            "resolved_from_kc_map": resolved_from_map,
            "csv_rows": key_to_csv_rows.get(key, []),
            "n_questions": len(matched_qids),
            "questions": {},
        }

        target_image_tokens = set()
        for qid in matched_qids:
            qobj = questions[str(qid)]
            matched_routes = question_leaf_routes.get(str(qid), {}).get(target_leaf, [])
            image_refs = extract_image_refs(qobj)
            image_tokens = []
            for ref in image_refs:
                token = ref["image_token"]
                image_tokens.append(token)
                target_image_tokens.add(token)
                found = find_image_file(args.image_root, token, image_exts)
                copied_to = ""
                if args.copy_images and found is not None:
                    dst = image_out_dir / found.name
                    if token not in copied_tokens:
                        shutil.copy2(found, dst)
                        copied_tokens.add(token)
                    copied_to = str(dst)

                image_rows.append({
                    "candidate_kc_key": key,
                    "candidate_kc_name": target_leaf,
                    "question_id": qid,
                    "image_token": token,
                    "source_field": ref["source_field"],
                    "resolved_image_path": str(found) if found is not None else "",
                    "copied_to": copied_to,
                })

            enriched = dict(qobj)
            enriched["_matched_candidate_kc_key"] = key
            enriched["_matched_candidate_kc_name"] = target_leaf
            enriched["_matched_routes"] = matched_routes
            enriched["_image_refs"] = image_refs
            target_obj["questions"][qid] = enriched

            flat_rows.append({
                "candidate_kc_key": key,
                "candidate_kc_name": target_leaf,
                "question_id": qid,
                "matched_routes": safe_json_dumps(matched_routes),
                "image_tokens": safe_json_dumps(sorted(set(image_tokens))),
                "content": qobj.get("content", ""),
                "answer": safe_json_dumps(qobj.get("answer", [])),
                "analysis": qobj.get("analysis", ""),
                "type": qobj.get("type", ""),
                "options": safe_json_dumps(qobj.get("options", {})),
                "all_kc_routes": safe_json_dumps(qobj.get("kc_routes", [])),
            })

        result["targets"][key] = target_obj
        summary_rows.append({
            "candidate_kc_key": key,
            "candidate_kc_name": target_leaf,
            "resolved_from_kc_map": resolved_from_map,
            "n_questions_found": len(matched_qids),
            "n_unique_image_tokens": len(target_image_tokens),
            "question_ids": safe_json_dumps(matched_qids),
            "image_tokens": safe_json_dumps(sorted(target_image_tokens)),
        })

    # Write outputs.
    details_json = out_dir / "candidate_kc_question_details.json"
    flat_csv = out_dir / "candidate_kc_question_details_flat.csv"
    images_csv = out_dir / "candidate_kc_image_manifest.csv"
    summary_csv = out_dir / "candidate_kc_summary.csv"

    write_json(details_json, result)
    pd.DataFrame(flat_rows).to_csv(flat_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(image_rows).to_csv(images_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    print("Done.")
    print(f"Targets processed: {len(target_keys)}")
    print(f"Unresolved KC keys: {len(result['unresolved_kc_keys'])}")
    print(f"Question rows exported: {len(flat_rows)}")
    print(f"Image reference rows exported: {len(image_rows)}")
    print(f"Details JSON: {details_json}")
    print(f"Flat question CSV: {flat_csv}")
    print(f"Image manifest CSV: {images_csv}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    # python extract_kc_candidate_questions_images.py --candidate-csv xes3g5m_tree_manual_split_dkt_qid_fold0_kc_question_heterogeneity_over_coarse_candidates.csv --kc-map-json decoded_kc_routes_map.json --questions-json decoded_questions.json --image-root images --copy-images --output-dir kc_candidate_question_outputs
    # python extract_kc_candidate_questions_images.py --candidate-csv "saved_model\xes3g5m_dkt+_qid_f0_s42_c24dee95fa_551aa3ec-6240-46f7-b884-9aecba6d2426\d\valid_kc_question_heterogeneity_over_coarse_candidates.csv" --kc-map-json ../data\xes3g5m\metadata\kc_routes_map.json --questions-json ../data\xes3g5m\metadata\questions.json --output-dir kc_candidate_question_outputs --image-root ../data\xes3g5m\metadata\images --copy-images 
    main()
