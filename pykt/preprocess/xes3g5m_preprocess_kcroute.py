import os
import json
import pandas as pd
from .utils import write_txt


KEYS = ["uid", "concepts", "questions"]
# 根据 questions.json 里的 kc_routes 重新生成 concepts

# -----------------------------
# Basic sequence utilities
# -----------------------------
def _split_seq(x):
    """Split a pyKT sequence cell like "1,2,3" into a clean list of tokens."""
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


def _normalize_qid(qid):
    """
    Normalize question id tokens so CSV values like 12, "12", or "12.0"
    can all match keys in questions.json such as "12".
    """
    s = str(qid).strip()
    if s == "":
        return s
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s


# -----------------------------
# JSON path discovery
# -----------------------------
def _find_json_file(read_file, filename, explicit_path=None):
    """
    Find sidecar JSON file automatically.

    Search order:
    1. explicit_path, if provided
    2. environment variable, e.g. QUESTIONS_JSON / KC_ROUTES_MAP_JSON
    3. read_file directory and its ancestors
    4. metadata/ subfolder under each ancestor
    5. current working directory
    """
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Explicit JSON path does not exist: {explicit_path}")

    env_name = filename.upper().replace(".", "_")
    env_path = os.environ.get(env_name)
    if env_path and os.path.exists(env_path):
        return env_path

    candidates = []
    read_dir = os.path.abspath(os.path.dirname(str(read_file)))

    cur = read_dir
    for _ in range(8):
        candidates.append(os.path.join(cur, filename))
        candidates.append(os.path.join(cur, "metadata", filename))
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    candidates.append(os.path.join(os.getcwd(), filename))
    candidates.append(os.path.join(os.getcwd(), "metadata", filename))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Cannot find {filename}. Put it near the CSV/dataset root/metadata folder, "
        f"or pass `{filename.replace('.json', '_path')}` explicitly. Tried examples: "
        f"{candidates[:6]} ..."
    )


# -----------------------------
# Concept mapping from questions.json + kc_routes_map.json
# -----------------------------
def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_kc_name_to_id(kc_routes_map):
    """
    kc_routes_map.json is assumed to be {kc_id: kc_name}.

    We build two indexes:
    - exact name -> id, preserving leading/trailing spaces when they exist
    - stripped name -> id, only when the stripped name is not ambiguous

    This handles data like:
    route leaf: " 几何思想平移"
    map value:  " 几何思想平移"
    while still avoiding silent wrong mapping when duplicate names exist.
    """
    exact = {}
    stripped_to_ids = {}

    for kc_id, kc_name in kc_routes_map.items():
        kc_id = str(kc_id).strip()
        kc_name_exact = str(kc_name)
        kc_name_strip = kc_name_exact.strip()

        if kc_name_exact in exact:
            raise ValueError(
                f"Duplicate exact KC name in kc_routes_map: {repr(kc_name_exact)} -> "
                f"{exact[kc_name_exact]} and {kc_id}"
            )
        exact[kc_name_exact] = kc_id
        stripped_to_ids.setdefault(kc_name_strip, []).append(kc_id)

    stripped = {}
    ambiguous = {}
    for name, ids in stripped_to_ids.items():
        uniq_ids = sorted(set(ids), key=lambda x: int(x) if str(x).isdigit() else str(x))
        if len(uniq_ids) == 1:
            stripped[name] = uniq_ids[0]
        else:
            ambiguous[name] = uniq_ids

    return exact, stripped, ambiguous


def _route_to_leaf_name(route):
    """
    Extract the final KC name from a route string.

    Example:
    "拓展思维----几何模块----直线型----几何思想与方法----几何方法整体减空白"
    -> "几何方法整体减空白"
    """
    if route is None:
        return ""
    parts = str(route).split("----")
    return parts[-1]


def _leaf_name_to_kc_id(leaf_name, exact_index, stripped_index, ambiguous_stripped):
    """
    Convert a leaf KC name into its numeric KC id.

    Matching rule:
    1. exact match first
    2. stripped match second, only if this stripped name is unambiguous
    3. otherwise raise an error immediately
    """
    leaf_exact = str(leaf_name)
    leaf_strip = leaf_exact.strip()

    if leaf_exact in exact_index:
        return exact_index[leaf_exact]

    if leaf_strip in stripped_index:
        return stripped_index[leaf_strip]

    if leaf_strip in ambiguous_stripped:
        raise ValueError(
            f"Ambiguous KC leaf name after strip: {repr(leaf_strip)} can map to ids "
            f"{ambiguous_stripped[leaf_strip]}. Please make the route leaf name exactly match "
            f"one value in kc_routes_map.json, including spaces if needed."
        )

    raise KeyError(
        f"KC leaf name not found in kc_routes_map.json: exact={repr(leaf_exact)}, "
        f"stripped={repr(leaf_strip)}"
    )


def build_question_to_concepts(questions_json_path, kc_routes_map_json_path):
    """
    Build mapping:
        question_id -> concept_id or concept_id_concept_id_...

    If a question has multiple kc_routes, it will map to multiple concept ids,
    joined by underscore, which is the common multi-concept format in pyKT.
    """
    questions = _load_json(questions_json_path)
    kc_routes_map = _load_json(kc_routes_map_json_path)

    if not isinstance(questions, dict):
        raise TypeError(f"questions.json must be a dict, got {type(questions)}")
    if not isinstance(kc_routes_map, dict):
        raise TypeError(f"kc_routes_map.json must be a dict, got {type(kc_routes_map)}")

    exact_index, stripped_index, ambiguous_stripped = _build_kc_name_to_id(kc_routes_map)

    qid_to_concepts = {}
    multi_route_questions = 0
    total_routes = 0

    for raw_qid, qobj in questions.items():
        qid = _normalize_qid(raw_qid)
        if not isinstance(qobj, dict):
            raise TypeError(f"questions.json[{raw_qid}] must be a dict, got {type(qobj)}")

        routes = qobj.get("kc_routes", [])
        if routes is None:
            routes = []
        elif isinstance(routes, str):
            routes = [routes]
        elif not isinstance(routes, list):
            raise TypeError(
                f"questions.json[{raw_qid}]['kc_routes'] must be list/string/None, "
                f"got {type(routes)}"
            )

        if len(routes) == 0:
            raise ValueError(f"Question {raw_qid} has no kc_routes in questions.json")
        if len(routes) > 1:
            multi_route_questions += 1

        concept_ids = []
        for route in routes:
            leaf_name = _route_to_leaf_name(route)
            kc_id = _leaf_name_to_kc_id(leaf_name, exact_index, stripped_index, ambiguous_stripped)
            concept_ids.append(str(kc_id))
            total_routes += 1

        # Remove duplicate ids inside the same question while preserving order.
        # Example: if two routes accidentally point to the same final KC, keep it once.
        deduped = list(dict.fromkeys(concept_ids))
        qid_to_concepts[qid] = "_".join(deduped)

    print(
        "question-to-concepts mapping loaded, "
        f"question num: {len(qid_to_concepts)}, "
        f"total kc_routes: {total_routes}, "
        f"multi-route questions: {multi_route_questions}, "
        f"ambiguous stripped KC names ignored unless exact match fails: {len(ambiguous_stripped)}"
    )

    return qid_to_concepts


# -----------------------------
# Main preprocessing function
# -----------------------------
def read_data_from_csv(
    read_file,
    write_file,
    questions_json_path=None,
    kc_routes_map_json_path=None,
):
    """
    XES3G5M preprocessing.

    Important change:
    - concepts are NOT taken from the original CSV anymore.
    - For each question id, this function looks up questions.json[question_id]['kc_routes'].
    - For every route, it takes the final route segment as the leaf KC name.
    - Then it maps that KC name to id through kc_routes_map.json.
    - If a question has multiple routes, concepts becomes "id1_id2_...".

    Input CSV still needs:
    - uid
    - questions
    - responses

    Optional columns:
    - timestamps
    - selectmasks
    """
    questions_json_path = _find_json_file(read_file, "questions.json", questions_json_path)
    kc_routes_map_json_path = _find_json_file(read_file, "kc_routes_map.json", kc_routes_map_json_path)

    qid_to_concepts = build_question_to_concepts(questions_json_path, kc_routes_map_json_path)

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

    # concepts is intentionally removed from required columns.
    # We rebuild concepts from questions.json + kc_routes_map.json.
    required_cols = {"uid", "questions", "responses"}
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

    missing_qids = set()

    for ridx, row in df.iterrows():
        uid = str(row["uid"])
        questions = _split_seq(row["questions"])
        responses = _split_seq(row["responses"])
        timestamps = _split_seq(row["timestamps"]) if has_timestamps else []
        selectmasks = _split_seq(row["selectmasks"]) if has_selectmasks else []

        n = min(len(questions), len(responses))
        if n == 0:
            continue

        q2, c2, r2, t2 = [], [], [], []
        for i in range(n):
            if has_selectmasks and i < len(selectmasks) and selectmasks[i] == "-1":
                continue
            if not _valid_response(responses[i]):
                continue

            qid = _normalize_qid(questions[i])
            if qid not in qid_to_concepts:
                missing_qids.add(qid)
                continue

            concept_str = qid_to_concepts[qid]

            q2.append(qid)
            c2.append(concept_str)
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

    if missing_qids:
        examples = sorted(missing_qids, key=lambda x: int(x) if str(x).isdigit() else str(x))[:20]
        raise KeyError(
            f"Found {len(missing_qids)} question ids in CSV but not in questions.json. "
            f"Examples: {examples}"
        )

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
        f"questions_json: {questions_json_path}, "
        f"kc_routes_map_json: {kc_routes_map_json_path}, "
        f"interaction num: {total_interactions}, "
        f"user num: {len(uniq_users)}, "
        f"question num: {len(uniq_questions)}, "
        f"concept num (leaf id): {len(uniq_concepts_leaf)}, "
        f"concept num (raw multi-concept token): {len(uniq_concepts_raw)}, "
        f"avg(ins) per s: {avg_ins}, "
        f"seq rows kept: {valid_rows}, "
        f"uid merged sequences: {len(user_inter)}"
    )
