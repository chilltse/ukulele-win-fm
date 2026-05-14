# import json
# import os
# from pathlib import Path

# import pandas as pd
# from .utils import write_txt


# KEYS = ["uid", "concepts", "questions"]


# def _split_seq(x):
#     if pd.isna(x):
#         return []
#     s = str(x).strip()
#     if s == "":
#         return []
#     return [t.strip() for t in s.split(",")]


# def _valid_response(v):
#     try:
#         iv = int(v)
#         return iv in (0, 1)
#     except Exception:
#         return False


# def _ts_sort_value(ts):
#     if ts is None:
#         return 10**30
#     s = str(ts).strip()
#     if s == "" or s.upper() == "NA" or s == "-1":
#         return 10**30
#     try:
#         return int(float(s))
#     except Exception:
#         return 10**30


# def _leaf_from_route(route):
#     """Take the last KC name from a route like A----B----leaf."""
#     return str(route).split("----")[-1].strip()


# def _dedup_keep_order(items):
#     seen = set()
#     out = []
#     for x in items:
#         sx = str(x).strip()
#         if sx == "" or sx in seen:
#             continue
#         seen.add(sx)
#         out.append(sx)
#     return out


# def _find_default_file(read_file, write_file, candidates):
#     """Try to find helper files near the source csv or output data.txt."""
#     base_dirs = []
#     for p in [read_file, write_file]:
#         if p is None:
#             continue
#         pp = Path(p).resolve()
#         base_dirs.extend([pp.parent, pp.parent.parent])
#     # remove duplicates while keeping order
#     seen = set()
#     for d in base_dirs:
#         if d in seen:
#             continue
#         seen.add(d)
#         for name in candidates:
#             fp = d / name
#             if fp.exists():
#                 return str(fp)
#     return None


# def _build_question_new_concepts_map(split_questions_file, kc_routes_map_file):
#     """
#     Build question_id -> new concept token.

#     Important:
#     - This is question-specific, not old-concept-specific.
#     - If a question has multiple KC routes, output is joined by '_', e.g. '12_34'.
#     """
#     if not split_questions_file or not kc_routes_map_file:
#         return {}, {"enabled": False, "reason": "missing split_questions_file or kc_routes_map_file"}

#     with open(split_questions_file, "r", encoding="utf-8") as f:
#         questions_json = json.load(f)
#     with open(kc_routes_map_file, "r", encoding="utf-8") as f:
#         kc_map = json.load(f)

#     # kc_routes_map is expected to be {"0": "KC name", ...}; invert it.
#     name2id = {}
#     duplicate_names = []
#     for kid, name in kc_map.items():
#         key = str(name).strip()
#         if key in name2id and name2id[key] != str(kid):
#             duplicate_names.append(key)
#         name2id[key] = str(kid)

#     qid2concepts = {}
#     missing_leaf_names = set()
#     questions_without_routes = 0

#     for qid, item in questions_json.items():
#         routes = item.get("kc_routes", []) if isinstance(item, dict) else []
#         if not routes:
#             questions_without_routes += 1
#             continue

#         concept_ids = []
#         for route in routes:
#             leaf = _leaf_from_route(route)
#             if leaf in name2id:
#                 concept_ids.append(name2id[leaf])
#             else:
#                 missing_leaf_names.add(leaf)

#         concept_ids = _dedup_keep_order(concept_ids)
#         if concept_ids:
#             qid2concepts[str(qid)] = "_".join(concept_ids)

#     info = {
#         "enabled": True,
#         "split_questions_file": str(split_questions_file),
#         "kc_routes_map_file": str(kc_routes_map_file),
#         "question_mappings": len(qid2concepts),
#         "missing_leaf_names": sorted(missing_leaf_names),
#         "missing_leaf_count": len(missing_leaf_names),
#         "duplicate_kc_names": sorted(set(duplicate_names)),
#         "duplicate_kc_name_count": len(set(duplicate_names)),
#         "questions_without_routes": questions_without_routes,
#     }
#     return qid2concepts, info


# def read_data_from_csv(
#     read_file,
#     write_file,
#     split_questions_file=None,
#     kc_routes_map_file=None,
# ):
#     """
#     XES3G5M preprocessing (typically from question_level/train_valid_sequences_quelevel.csv):
#     - remove padded positions by selectmasks == -1
#     - keep only valid responses in {0, 1}
#     - optionally replace old leaf KC ids with the newly split KC leaf ids
#     - write pyKT standard 6-line block format to data.txt

#     How to enable split-KC replacement:
#     1. Pass split_questions_file and kc_routes_map_file directly, or
#     2. Set environment variables:
#        SPLIT_QUESTIONS_JSON=/path/to/decoded_questions_kc_routes_split.json
#        KC_ROUTES_MAP_JSON=/path/to/kc_routes_map_with_split_changes.json
#     3. Or put those two files near read_file/write_file with the same filenames.
#     """
#     # Build question -> updated leaf KC id(s) mapping.
#     split_questions_file = (
#         split_questions_file
#         or os.environ.get("SPLIT_QUESTIONS_JSON")
#         or _find_default_file(
#             read_file,
#             write_file,
#             [r"G:\ANU_course\pykt-ukulele-May-WIN\pykt-ukelele\data\xes3g5m_tree\metadata\questions.json"],
#         )
#     )
#     kc_routes_map_file = (
#         kc_routes_map_file
#         or os.environ.get("KC_ROUTES_MAP_JSON")
#         or _find_default_file(
#             read_file,
#             write_file,
#             [r"G:\ANU_course\pykt-ukulele-May-WIN\pykt-ukelele\data\xes3g5m_tree\metadata\kc_routes_map.json"],
#         )
#     )
#     qid2new_concepts, split_info = _build_question_new_concepts_map(
#         split_questions_file, kc_routes_map_file
#     )
#     if split_info.get("enabled"):
#         print(
#             "split KC replacement enabled, "
#             f"question mappings: {split_info['question_mappings']}, "
#             f"missing leaf names: {split_info['missing_leaf_count']}, "
#             f"duplicate KC names: {split_info['duplicate_kc_name_count']}"
#         )
#         if split_info["missing_leaf_count"] > 0:
#             print("WARNING missing leaf names:", split_info["missing_leaf_names"][:20])
#     else:
#         print("split KC replacement disabled:", split_info.get("reason"))

#     # Build a unified raw pool before re-splitting:
#     # include both train_valid_sequences_quelevel.csv and test_quelevel.csv if available.
#     read_files = [read_file]
#     norm_path = str(read_file).replace("\\", "/")
#     if norm_path.endswith("/question_level/train_valid_sequences_quelevel.csv"):
#         qlevel_dir = norm_path.rsplit("/", 1)[0]
#         test_path = f"{qlevel_dir}/test_quelevel.csv"
#         if test_path != read_file and os.path.exists(test_path):
#             read_files.append(test_path)

#     dfs = [pd.read_csv(fp, encoding="utf-8", low_memory=False) for fp in read_files]
#     df = pd.concat(dfs, ignore_index=True)

#     required_cols = {"uid", "questions", "concepts", "responses"}
#     missing = required_cols - set(df.columns)
#     if missing:
#         raise ValueError(f"Missing required columns in source files {read_files}: {sorted(missing)}")

#     has_timestamps = "timestamps" in df.columns
#     has_selectmasks = "selectmasks" in df.columns

#     user_inter = []
#     total_interactions = 0
#     valid_rows = 0

#     uniq_users = set()
#     uniq_questions = set()
#     uniq_concepts_raw = set()
#     uniq_concepts_leaf = set()
#     user_chunks = dict()

#     replaced_positions = 0
#     unchanged_positions = 0
#     questions_not_in_split_map = set()
#     changed_examples = []

#     for ridx, row in df.iterrows():
#         uid = str(row["uid"])
#         questions = _split_seq(row["questions"])
#         concepts = _split_seq(row["concepts"])
#         responses = _split_seq(row["responses"])
#         timestamps = _split_seq(row["timestamps"]) if has_timestamps else []
#         selectmasks = _split_seq(row["selectmasks"]) if has_selectmasks else []

#         n = min(len(questions), len(concepts), len(responses))
#         if n == 0:
#             continue

#         q2, c2, r2, t2 = [], [], [], []
#         for i in range(n):
#             if has_selectmasks and i < len(selectmasks) and selectmasks[i] == "-1":
#                 continue
#             if not _valid_response(responses[i]):
#                 continue

#             qid = str(questions[i]).strip()
#             old_concept = str(concepts[i]).strip()
#             new_concept = qid2new_concepts.get(qid, old_concept)

#             if qid2new_concepts:
#                 if qid in qid2new_concepts:
#                     if new_concept != old_concept:
#                         replaced_positions += 1
#                         if len(changed_examples) < 10:
#                             changed_examples.append((qid, old_concept, new_concept))
#                     else:
#                         unchanged_positions += 1
#                 else:
#                     questions_not_in_split_map.add(qid)

#             q2.append(qid)
#             c2.append(new_concept)
#             r2.append(str(int(responses[i])))
#             if has_timestamps:
#                 t2.append(timestamps[i] if i < len(timestamps) else "NA")

#         if len(q2) == 0:
#             continue

#         valid_rows += 1

#         uniq_users.add(uid)
#         uniq_questions.update(q2)
#         uniq_concepts_raw.update(c2)
#         for c in c2:
#             for leaf in str(c).split("_"):
#                 leaf = leaf.strip()
#                 if leaf and leaf not in {"-1", "NA"}:
#                     uniq_concepts_leaf.add(leaf)
#         row_ts_key = _ts_sort_value(t2[0]) if (has_timestamps and len(t2) > 0) else ridx
#         user_chunks.setdefault(uid, []).append((row_ts_key, q2, c2, r2, t2))

#     # Merge repeated uid rows by timestamp order.
#     for uid, chunks in user_chunks.items():
#         chunks = sorted(chunks, key=lambda x: x[0])
#         merged_q, merged_c, merged_r, merged_t = [], [], [], []
#         for _, q2, c2, r2, t2 in chunks:
#             merged_q.extend(q2)
#             merged_c.extend(c2)
#             merged_r.extend(r2)
#             if has_timestamps:
#                 merged_t.extend(t2)

#         seq_len = len(merged_q)
#         if seq_len == 0:
#             continue
#         total_interactions += seq_len
#         user_inter.append(
#             [
#                 [uid, str(seq_len)],
#                 [str(x) for x in merged_q],
#                 [str(x) for x in merged_c],
#                 [str(x) for x in merged_r],
#                 [str(x) for x in merged_t] if has_timestamps else ["NA"],
#                 ["NA"],
#             ]
#         )

#     write_txt(write_file, user_inter)

#     avg_ins = round(total_interactions / len(uniq_users), 4) if uniq_users else 0.0
#     print(
#         "after xes3g5m preprocess, "
#         f"source files: {len(read_files)}, "
#         f"interaction num: {total_interactions}, "
#         f"user num: {len(uniq_users)}, "
#         f"question num: {len(uniq_questions)}, "
#         f"concept num (leaf): {len(uniq_concepts_leaf)}, "
#         f"concept num (raw token): {len(uniq_concepts_raw)}, "
#         f"avg(ins) per s: {avg_ins}, "
#         f"seq rows kept: {valid_rows}, "
#         f"uid merged sequences: {len(user_inter)}"
#     )

#     if qid2new_concepts:
#         print(
#             "split KC replacement summary, "
#             f"replaced positions: {replaced_positions}, "
#             f"same-as-before positions: {unchanged_positions}, "
#             f"questions not in split map: {len(questions_not_in_split_map)}"
#         )
#         if replaced_positions == 0:
#             print(
#                 "WARNING: no concept positions were replaced. "
#                 "Check whether CSV questions ids match decoded_questions JSON keys."
#             )
#         if changed_examples:
#             print("split KC replacement examples:")
#             for qid, old_c, new_c in changed_examples:
#                 print(f"  qid={qid}: {old_c} -> {new_c}")






import json
import os
import re
from pathlib import Path

import pandas as pd
from .utils import write_txt


KEYS = ["uid", "concepts", "questions"]


# ============================================================
# Basic utilities
# ============================================================

def _split_seq(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "":
        return []
    return [t.strip() for t in s.split(",")]


def _valid_response(v):
    try:
        iv = int(v)
        return iv in (0, 1)
    except Exception:
        return False


def _ts_sort_value(ts):
    if ts is None:
        return 10**30
    s = str(ts).strip()
    if s == "" or s.upper() == "NA" or s == "-1":
        return 10**30
    try:
        return int(float(s))
    except Exception:
        return 10**30


def _leaf_from_route(route):
    """Take the last KC name from a route like A----B----leaf."""
    return str(route).split("----")[-1].strip()


def _dedup_keep_order(items):
    seen = set()
    out = []
    for x in items:
        sx = str(x).strip()
        if sx == "" or sx in seen:
            continue
        seen.add(sx)
        out.append(sx)
    return out


def _is_windows_abs_path(s):
    """Return True for paths like G:\\data\\file.json even on non-Windows systems."""
    return bool(re.match(r"^[A-Za-z]:[\\/]", str(s)))


def _existing_path(path_str):
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return str(p)
    return None


def _find_default_file(read_file, write_file, candidate_names):
    """
    Find helper files near the source csv or output data.txt.

    Important:
    - candidate_names should be filenames, not old metadata/questions.json.
    - absolute paths are also supported.
    """
    # 1) Try candidate absolute paths directly.
    for name in candidate_names:
        if Path(name).is_absolute() or _is_windows_abs_path(name):
            fp = _existing_path(name)
            if fp:
                return fp

    # 2) Search nearby directories.
    base_dirs = []
    for p in [read_file, write_file]:
        if p is None:
            continue
        pp = Path(p).resolve()
        base_dirs.extend([
            pp.parent,
            pp.parent.parent,
            pp.parent.parent.parent,
        ])

    # remove duplicates while keeping order
    seen = set()
    for d in base_dirs:
        if d in seen:
            continue
        seen.add(d)
        for name in candidate_names:
            # skip absolute-like candidates here; already tried above
            if Path(name).is_absolute() or _is_windows_abs_path(name):
                continue
            fp = d / name
            if fp.exists():
                return str(fp)

    return None


def _qid_variants(qid):
    """
    Generate robust variants for question ids.

    Examples:
    - '8' -> ['8']
    - '8.0' -> ['8.0', '8']
    - 'question_8' -> ['question_8', '8']
    """
    s = str(qid).strip()
    variants = []
    if s:
        variants.append(s)

    # numeric float-like id, e.g. 8.0 -> 8
    try:
        f = float(s)
        if f.is_integer():
            variants.append(str(int(f)))
    except Exception:
        pass

    # question_123 / q_123 / q123 -> 123
    m = re.search(r"(?:question[_-]?|q[_-]?)(\d+)$", s, flags=re.IGNORECASE)
    if m:
        variants.append(m.group(1))

    return _dedup_keep_order(variants)


# ============================================================
# Split-KC mapping utilities
# ============================================================

def _build_kc_name_index(kc_map):
    """
    Build KC-name -> KC-id index.

    kc_routes_map is expected to be {"0": "KC name", ...}.

    We keep:
    1. exact_name2id: exact string, no stripping.
    2. stripped_name2ids: stripped string -> all ids.

    For lookup, exact name is preferred. If exact lookup fails and stripped
    lookup has duplicates, we choose the smallest numeric id and report it.
    """
    exact_name2id = {}
    stripped_name2ids = {}

    for kid, name in kc_map.items():
        kid_s = str(kid).strip()
        name_exact = str(name)
        name_stripped = name_exact.strip()

        if name_exact not in exact_name2id:
            exact_name2id[name_exact] = kid_s

        stripped_name2ids.setdefault(name_stripped, [])
        if kid_s not in stripped_name2ids[name_stripped]:
            stripped_name2ids[name_stripped].append(kid_s)

    duplicate_stripped_names = {
        name: ids
        for name, ids in stripped_name2ids.items()
        if len(ids) > 1
    }

    return exact_name2id, stripped_name2ids, duplicate_stripped_names


def _sort_kc_ids(ids):
    def key_fn(x):
        try:
            return (0, int(x))
        except Exception:
            return (1, str(x))
    return sorted(ids, key=key_fn)


def _lookup_kc_id(leaf_name, exact_name2id, stripped_name2ids, duplicate_lookup_records):
    """
    Convert leaf KC name to KC id.

    exact name is preferred. If only stripped duplicate candidates exist,
    choose the smallest numeric id and record the ambiguity.
    """
    leaf_exact = str(leaf_name)
    leaf_stripped = leaf_exact.strip()

    if leaf_exact in exact_name2id:
        return exact_name2id[leaf_exact]

    ids = stripped_name2ids.get(leaf_stripped, [])
    if len(ids) == 1:
        return ids[0]
    if len(ids) > 1:
        chosen = _sort_kc_ids(ids)[0]
        duplicate_lookup_records.append({
            "leaf_name": leaf_stripped,
            "candidate_ids": _sort_kc_ids(ids),
            "chosen_id": chosen,
        })
        return chosen

    return None


def _build_question_new_concepts_map(split_questions_file, kc_routes_map_file):
    """
    Build question_id -> new concept token.

    Critical design:
    - This is question-specific, not old-concept-specific.
    - The old concepts column in the sequence CSV is ignored when a question id
      exists in this map.
    - If a question has multiple KC routes, the output is joined by '_',
      e.g. '12_34'.
    """
    if not split_questions_file or not kc_routes_map_file:
        return {}, {
            "enabled": False,
            "reason": "missing split_questions_file or kc_routes_map_file",
            "split_questions_file": split_questions_file,
            "kc_routes_map_file": kc_routes_map_file,
        }

    with open(split_questions_file, "r", encoding="utf-8") as f:
        questions_json = json.load(f)
    with open(kc_routes_map_file, "r", encoding="utf-8") as f:
        kc_map = json.load(f)

    exact_name2id, stripped_name2ids, duplicate_stripped_names = _build_kc_name_index(kc_map)

    qid2concepts = {}
    missing_leaf_names = set()
    questions_without_routes = 0
    qid_variant_collisions = []
    duplicate_lookup_records = []
    mapped_concept_ids = set()

    for raw_qid, item in questions_json.items():
        routes = item.get("kc_routes", []) if isinstance(item, dict) else []
        if not routes:
            questions_without_routes += 1
            continue

        concept_ids = []
        for route in routes:
            leaf = _leaf_from_route(route)
            kid = _lookup_kc_id(
                leaf,
                exact_name2id,
                stripped_name2ids,
                duplicate_lookup_records,
            )
            if kid is None:
                missing_leaf_names.add(leaf)
            else:
                concept_ids.append(kid)
                mapped_concept_ids.add(str(kid))

        concept_ids = _dedup_keep_order(concept_ids)
        if not concept_ids:
            continue

        concept_token = "_".join(concept_ids)

        # Main key and robust variants.
        possible_qids = _qid_variants(raw_qid)

        # If JSON item also has explicit id/question_id fields, include them.
        if isinstance(item, dict):
            for k in ["question_id", "qid", "id"]:
                if k in item:
                    possible_qids.extend(_qid_variants(item[k]))

        for qid_key in _dedup_keep_order(possible_qids):
            old = qid2concepts.get(qid_key)
            if old is not None and old != concept_token:
                qid_variant_collisions.append({
                    "qid": qid_key,
                    "old_concept": old,
                    "new_concept": concept_token,
                    "raw_qid": raw_qid,
                })
                # Keep the first mapping to avoid unstable overwriting.
                continue
            qid2concepts[qid_key] = concept_token

    def _to_int_or_none(x):
        try:
            return int(str(x))
        except Exception:
            return None

    mapped_int_ids = [x for x in (_to_int_or_none(c) for c in mapped_concept_ids) if x is not None]

    info = {
        "enabled": True,
        "split_questions_file": str(split_questions_file),
        "kc_routes_map_file": str(kc_routes_map_file),
        "question_mappings": len(qid2concepts),
        "missing_leaf_names": sorted(missing_leaf_names),
        "missing_leaf_count": len(missing_leaf_names),
        "duplicate_kc_names": {
            name: ids
            for name, ids in duplicate_stripped_names.items()
        },
        "duplicate_kc_name_count": len(duplicate_stripped_names),
        "duplicate_lookup_records": duplicate_lookup_records[:50],
        "duplicate_lookup_record_count": len(duplicate_lookup_records),
        "questions_without_routes": questions_without_routes,
        "qid_variant_collision_count": len(qid_variant_collisions),
        "qid_variant_collisions": qid_variant_collisions[:50],
        "mapped_unique_concept_count": len(mapped_concept_ids),
        "mapped_min_concept_id": min(mapped_int_ids) if mapped_int_ids else None,
        "mapped_max_concept_id": max(mapped_int_ids) if mapped_int_ids else None,
    }
    return qid2concepts, info


def _parse_int_env(name, default=None):
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _parse_bool_env(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() not in {"0", "false", "no", "off"}


def _print_split_info(split_info, qid2new_concepts, debug_qids=None):
    if not split_info.get("enabled"):
        print("split KC replacement disabled:", split_info.get("reason"))
        print("SPLIT_QUESTIONS_JSON =", split_info.get("split_questions_file"))
        print("KC_ROUTES_MAP_JSON   =", split_info.get("kc_routes_map_file"))
        return

    print(
        "split KC replacement enabled, "
        f"question mappings: {split_info['question_mappings']}, "
        f"mapped unique concept ids: {split_info['mapped_unique_concept_count']}, "
        f"mapped id range: {split_info['mapped_min_concept_id']}..{split_info['mapped_max_concept_id']}, "
        f"missing leaf names: {split_info['missing_leaf_count']}, "
        f"duplicate KC names: {split_info['duplicate_kc_name_count']}"
    )
    print("SPLIT_QUESTIONS_JSON =", split_info["split_questions_file"])
    print("KC_ROUTES_MAP_JSON   =", split_info["kc_routes_map_file"])

    if split_info["missing_leaf_count"] > 0:
        print("WARNING missing leaf names:", split_info["missing_leaf_names"][:20])

    if split_info["qid_variant_collision_count"] > 0:
        print(
            "WARNING qid variant collisions:",
            split_info["qid_variant_collisions"][:5],
        )

    if split_info["duplicate_lookup_record_count"] > 0:
        print(
            "WARNING duplicate stripped KC-name lookup used:",
            split_info["duplicate_lookup_records"][:5],
        )

    # Print a few qid -> concept mappings to confirm we are reading the right JSON.
    if debug_qids:
        print("split KC debug qid mappings:")
        for qid in debug_qids:
            qid_s = str(qid).strip()
            print(f"  qid={qid_s}: {qid2new_concepts.get(qid_s)}")


# ============================================================
# Main preprocessing
# ============================================================

def read_data_from_csv(
    read_file,
    write_file,
    split_questions_file=None,
    kc_routes_map_file=None,
    require_split_kc=None,
    split_new_kc_min_id=None,
):
    """
    XES3G5M preprocessing.

    Input is usually:
        question_level/train_valid_sequences_quelevel.csv

    Main behavior:
    - remove padded positions by selectmasks == -1
    - keep only valid responses in {0, 1}
    - replace old leaf KC ids with newly split leaf KC ids by QUESTION ID
    - write pyKT standard 6-line block format to data.txt

    The replacement is:
        question_id -> decoded_questions_kc_routes_strict_split.json
                    -> leaf KC name(s)
                    -> kc_routes_map_strict_split.json
                    -> new KC id(s)

    Recommended environment variables on Windows PowerShell:
        $env:SPLIT_QUESTIONS_JSON="G:\\...\\decoded_questions_kc_routes_strict_split.json"
        $env:KC_ROUTES_MAP_JSON="G:\\...\\kc_routes_map_strict_split.json"

    Safety:
    - By default, missing split files raise an error, because silently using old
      concepts is dangerous for this experiment.
    - Set REQUIRE_SPLIT_KC=0 only if you intentionally want the old behavior.
    """
    if require_split_kc is None:
        require_split_kc = _parse_bool_env("REQUIRE_SPLIT_KC", default=True)
    if split_new_kc_min_id is None:
        # For your current strict map, newly added KC ids start around 1175.
        # Override with env SPLIT_NEW_KC_MIN_ID if needed.
        split_new_kc_min_id = _parse_int_env("SPLIT_NEW_KC_MIN_ID", default=1175)

    # Resolve helper files. Do NOT default to metadata/questions.json because that
    # is usually the old unsplit JSON and will keep concept_num unchanged.
    split_questions_file = (
        split_questions_file
        or os.environ.get("SPLIT_QUESTIONS_JSON")
        or _find_default_file(
            read_file,
            write_file,
            [
                r"G:\ANU_course\pykt-ukulele-May-WIN\pykt-ukelele\data\xes3g5m_tree\metadata\questions.json"
            ],
        )
    )
    kc_routes_map_file = (
        kc_routes_map_file
        or os.environ.get("KC_ROUTES_MAP_JSON")
        or _find_default_file(
            read_file,
            write_file,
            [
                r"G:\ANU_course\pykt-ukulele-May-WIN\pykt-ukelele\data\xes3g5m_tree\metadata\kc_routes_map.json"
            ],
        )
    )

    qid2new_concepts, split_info = _build_question_new_concepts_map(
        split_questions_file, kc_routes_map_file
    )

    debug_qids = os.environ.get("SPLIT_KC_DEBUG_QIDS", "8,9,15").split(",")
    _print_split_info(split_info, qid2new_concepts, debug_qids=debug_qids)

    if require_split_kc and not split_info.get("enabled"):
        raise FileNotFoundError(
            "Split-KC replacement is required but helper files were not found. "
            "Please set SPLIT_QUESTIONS_JSON and KC_ROUTES_MAP_JSON to the strict split files."
        )

    if split_info.get("enabled") and split_info.get("missing_leaf_count", 0) > 0:
        raise ValueError(
            "Some leaf KC names in split questions JSON are missing from kc_routes_map. "
            f"First missing names: {split_info['missing_leaf_names'][:20]}"
        )

    # Build a unified raw pool before re-splitting:
    # include both train_valid_sequences_quelevel.csv and test_quelevel.csv if available.
    read_files = [read_file]
    norm_path = str(read_file).replace("\\", "/")
    if norm_path.endswith("/question_level/train_valid_sequences_quelevel.csv"):
        qlevel_dir = norm_path.rsplit("/", 1)[0]
        test_path = f"{qlevel_dir}/test_quelevel.csv"
        if test_path != read_file and os.path.exists(test_path):
            read_files.append(test_path)

    dfs = [pd.read_csv(fp, encoding="utf-8", low_memory=False) for fp in read_files]
    df = pd.concat(dfs, ignore_index=True)

    required_cols = {"uid", "questions", "concepts", "responses"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in source files {read_files}: {sorted(missing)}")

    has_timestamps = "timestamps" in df.columns
    has_selectmasks = "selectmasks" in df.columns

    user_inter = []
    total_interactions = 0
    valid_rows = 0

    uniq_users = set()
    uniq_questions = set()
    uniq_concepts_raw = set()
    uniq_concepts_leaf = set()
    user_chunks = dict()

    replaced_positions = 0
    unchanged_positions = 0
    questions_not_in_split_map = set()
    changed_examples = []
    same_examples = []
    new_split_ids_seen = set()

    for ridx, row in df.iterrows():
        uid = str(row["uid"])
        questions = _split_seq(row["questions"])
        concepts = _split_seq(row["concepts"])
        responses = _split_seq(row["responses"])
        timestamps = _split_seq(row["timestamps"]) if has_timestamps else []
        selectmasks = _split_seq(row["selectmasks"]) if has_selectmasks else []

        n = min(len(questions), len(concepts), len(responses))
        if n == 0:
            continue

        q2, c2, r2, t2 = [], [], [], []
        for i in range(n):
            if has_selectmasks and i < len(selectmasks) and selectmasks[i] == "-1":
                continue
            if not _valid_response(responses[i]):
                continue

            qid = str(questions[i]).strip()
            old_concept = str(concepts[i]).strip()

            # Strict replacement: do not trust the old concepts column if qid exists.
            new_concept = qid2new_concepts.get(qid, old_concept)

            if qid2new_concepts:
                if qid in qid2new_concepts:
                    if new_concept != old_concept:
                        replaced_positions += 1
                        if len(changed_examples) < 20:
                            changed_examples.append((qid, old_concept, new_concept))
                    else:
                        unchanged_positions += 1
                        if len(same_examples) < 10:
                            same_examples.append((qid, old_concept, new_concept))
                else:
                    questions_not_in_split_map.add(qid)

            for cid in str(new_concept).split("_"):
                cid = cid.strip()
                try:
                    if split_new_kc_min_id is not None and int(cid) >= split_new_kc_min_id:
                        new_split_ids_seen.add(int(cid))
                except Exception:
                    pass

            q2.append(qid)
            c2.append(new_concept)
            r2.append(str(int(responses[i])))
            if has_timestamps:
                t2.append(timestamps[i] if i < len(timestamps) else "NA")

        if len(q2) == 0:
            continue

        valid_rows += 1

        uniq_users.add(uid)
        uniq_questions.update(q2)
        uniq_concepts_raw.update(c2)
        for c in c2:
            for leaf in str(c).split("_"):
                leaf = leaf.strip()
                if leaf and leaf not in {"-1", "NA"}:
                    uniq_concepts_leaf.add(leaf)

        row_ts_key = _ts_sort_value(t2[0]) if (has_timestamps and len(t2) > 0) else ridx
        user_chunks.setdefault(uid, []).append((row_ts_key, q2, c2, r2, t2))

    # Merge repeated uid rows by timestamp order.
    for uid, chunks in user_chunks.items():
        chunks = sorted(chunks, key=lambda x: x[0])
        merged_q, merged_c, merged_r, merged_t = [], [], [], []
        for _, q2, c2, r2, t2 in chunks:
            merged_q.extend(q2)
            merged_c.extend(c2)
            merged_r.extend(r2)
            if has_timestamps:
                merged_t.extend(t2)

        seq_len = len(merged_q)
        if seq_len == 0:
            continue

        total_interactions += seq_len
        user_inter.append(
            [
                [uid, str(seq_len)],
                [str(x) for x in merged_q],
                [str(x) for x in merged_c],
                [str(x) for x in merged_r],
                [str(x) for x in merged_t] if has_timestamps else ["NA"],
                ["NA"],
            ]
        )

    write_txt(write_file, user_inter)

    avg_ins = round(total_interactions / len(uniq_users), 4) if uniq_users else 0.0
    print(
        "after xes3g5m preprocess, "
        f"source files: {len(read_files)}, "
        f"interaction num: {total_interactions}, "
        f"user num: {len(uniq_users)}, "
        f"question num: {len(uniq_questions)}, "
        f"concept num (leaf): {len(uniq_concepts_leaf)}, "
        f"concept num (raw token): {len(uniq_concepts_raw)}, "
        f"avg(ins) per s: {avg_ins}, "
        f"seq rows kept: {valid_rows}, "
        f"uid merged sequences: {len(user_inter)}"
    )

    if qid2new_concepts:
        print(
            "split KC replacement summary, "
            f"replaced positions: {replaced_positions}, "
            f"same-as-before positions: {unchanged_positions}, "
            f"questions not in split map: {len(questions_not_in_split_map)}, "
            f"new split concept ids seen >= {split_new_kc_min_id}: {len(new_split_ids_seen)}"
        )

        if changed_examples:
            print("split KC replacement changed examples:")
            for qid, old_c, new_c in changed_examples:
                print(f"  qid={qid}: {old_c} -> {new_c}")

        if same_examples:
            print("split KC replacement same-as-before examples:")
            for qid, old_c, new_c in same_examples:
                print(f"  qid={qid}: {old_c} -> {new_c}")

        if questions_not_in_split_map:
            print("questions not in split map examples:", sorted(list(questions_not_in_split_map))[:20])

        if replaced_positions == 0:
            raise RuntimeError(
                "No concept positions were replaced. "
                "CSV question ids probably do not match decoded_questions JSON keys."
            )

        if split_new_kc_min_id is not None and len(new_split_ids_seen) == 0:
            raise RuntimeError(
                f"No newly added split KC ids >= {split_new_kc_min_id} appeared in output. "
                "You are probably reading the old unsplit questions JSON, not "
                "decoded_questions_kc_routes_strict_split.json."
            )

        print("first new split ids seen:", sorted(new_split_ids_seen)[:50])
