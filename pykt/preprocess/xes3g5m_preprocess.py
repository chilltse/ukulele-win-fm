import os
import json
import shutil
from collections import defaultdict

import pandas as pd
from .utils import write_txt


KEYS = ["uid", "concepts", "questions"]
# 根据 questions.json 里的 kc_routes 重新生成 concepts
# Optional roll-up support:
#   leaf KC ids from questions.json/kc_routes_map.json
#       -> active KC ids after specified parent-node roll-up
#       -> write active concepts into sequence file
#       -> optionally write active keyid2idx.json + kc_rollup_meta.json


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


def _safe_int_sort_key(x):
    """Sort numeric ids numerically and non-numeric ids lexicographically."""
    s = str(x).strip()
    try:
        return (0, int(s))
    except Exception:
        return (1, s)


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for x in items:
        sx = str(x).strip()
        if sx == "":
            continue
        if sx not in seen:
            seen.add(sx)
            out.append(sx)
    return out


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


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Concept mapping from questions.json + kc_routes_map.json
# -----------------------------
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
        uniq_ids = sorted(set(ids), key=_safe_int_sort_key)
        if len(uniq_ids) == 1:
            stripped[name] = uniq_ids[0]
        else:
            ambiguous[name] = uniq_ids

    return exact, stripped, ambiguous


def _route_to_leaf_name(route):
    """
    Extract the final KC name from a route string.

    Important:
    build_kc_tree.py strips route segments by default, so preprocessing should
    use the same normalization policy. Otherwise the same route leaf may map to
    different kc_id values in the tree-building stage and preprocessing stage.
    """
    if route is None:
        return ""

    parts = str(route).split("----")
    leaf = parts[-1]

    return leaf.strip()

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
        qid_to_concepts[qid] = "_".join(_dedupe_keep_order(concept_ids))

    print(
        "question-to-concepts mapping loaded, "
        f"question num: {len(qid_to_concepts)}, "
        f"total kc_routes: {total_routes}, "
        f"multi-route questions: {multi_route_questions}, "
        f"ambiguous stripped KC names ignored unless exact match fails: {len(ambiguous_stripped)}"
    )

    return qid_to_concepts


# -----------------------------
# Roll-up utilities
# -----------------------------
def _parse_rollup_node_ids(rollup_node_ids):
    """
    Accepted formats:
    - None / "" / [] -> []
    - "100016,100280"
    - [100016, 100280]
    - 100016
    """
    if rollup_node_ids is None:
        return []
    if isinstance(rollup_node_ids, str):
        text = rollup_node_ids.strip()
        if not text:
            return []
        vals = [x.strip() for x in text.split(",") if x.strip()]
        return sorted(set(str(int(float(x))) if _looks_numeric(x) else str(x) for x in vals), key=_safe_int_sort_key)
    if isinstance(rollup_node_ids, (int, float)):
        if isinstance(rollup_node_ids, float) and not rollup_node_ids.is_integer():
            raise ValueError(f"rollup_node_ids contains non-integer float: {rollup_node_ids}")
        return [str(int(rollup_node_ids))]
    if isinstance(rollup_node_ids, (list, tuple, set)):
        vals = []
        for x in rollup_node_ids:
            sx = str(x).strip()
            if sx:
                vals.append(str(int(float(sx))) if _looks_numeric(sx) else sx)
        return sorted(set(vals), key=_safe_int_sort_key)
    raise ValueError(f"Unsupported rollup_node_ids type: {type(rollup_node_ids)}")


def _looks_numeric(x):
    try:
        float(str(x).strip())
        return True
    except Exception:
        return False


def _collect_tree_nodes(tree_data):
    nodes = []

    def collect(obj):
        if isinstance(obj, list):
            for item in obj:
                collect(item)
            return
        if not isinstance(obj, dict):
            return

        if "node_id" in obj:
            nodes.append(obj)

        for child in obj.get("children", []) or []:
            collect(child)

    collect(tree_data)
    return nodes


def _build_tree_maps(tree_data):
    """
    Build maps from nested kc_knowledge_tree.json.

    node ids are used for internal nodes.
    leaf kc ids are also indexed so route-derived leaf ids can be resolved safely.
    """
    nodes = _collect_tree_nodes(tree_data)

    node_map = {}
    parent_by_node = {}
    children_by_node = defaultdict(list)
    name_by_node = {}
    type_by_node = {}
    leaf_kcid_to_nodeid = {}

    for node in nodes:
        node_id = node.get("node_id", None)
        if node_id is None:
            continue
        node_id = str(node_id).strip()
        if node_id in node_map:
            raise ValueError(f"Duplicate node_id in kc_knowledge_tree.json: {node_id}")

        node_map[node_id] = node
        name_by_node[node_id] = str(node.get("name", ""))
        type_by_node[node_id] = str(node.get("type", ""))

        parent_id = node.get("parent_id", None)
        if parent_id is not None:
            parent_id = str(parent_id).strip()
            parent_by_node[node_id] = parent_id
            children_by_node[parent_id].append(node_id)

        if str(node.get("type", "")) == "kc_leaf":
            kc_id = node.get("kc_id", node.get("node_id", None))
            if kc_id is not None:
                kc_id = str(kc_id).strip()
                if kc_id in leaf_kcid_to_nodeid and leaf_kcid_to_nodeid[kc_id] != node_id:
                    raise ValueError(
                        f"Duplicate kc_id in kc_knowledge_tree.json: kc_id={kc_id}, "
                        f"node_ids={leaf_kcid_to_nodeid[kc_id]} and {node_id}"
                    )
                leaf_kcid_to_nodeid[kc_id] = node_id

    depth_by_node = {}

    def get_depth(node_id):
        node_id = str(node_id)
        if node_id in depth_by_node:
            return depth_by_node[node_id]
        parent = parent_by_node.get(node_id, None)
        if parent is None:
            depth_by_node[node_id] = 0
        else:
            depth_by_node[node_id] = get_depth(parent) + 1
        return depth_by_node[node_id]

    for node_id in node_map:
        get_depth(node_id)

    return {
        "nodes": nodes,
        "node_map": node_map,
        "parent_by_node": dict(parent_by_node),
        "children_by_node": {k: list(v) for k, v in children_by_node.items()},
        "name_by_node": name_by_node,
        "type_by_node": type_by_node,
        "leaf_kcid_to_nodeid": leaf_kcid_to_nodeid,
        "depth_by_node": depth_by_node,
    }


def _resolve_leaf_concept_to_tree_node(leaf_concept_id, tree_maps):
    """
    Route-derived concepts are leaf kc_id values.
    In your current tree, leaf node_id usually equals kc_id, but this function
    also supports the safer case where node_id and kc_id differ.
    """
    raw = str(leaf_concept_id).strip()
    if raw in tree_maps["leaf_kcid_to_nodeid"]:
        return tree_maps["leaf_kcid_to_nodeid"][raw]
    if raw in tree_maps["node_map"] and tree_maps["type_by_node"].get(raw) == "kc_leaf":
        return raw
    raise ValueError(
        f"Leaf concept id {raw} from questions/kc_routes_map cannot be found as a kc_leaf "
        "in kc_knowledge_tree.json. Check whether kc_routes_map ids match tree kc_id/node_id."
    )


def _is_descendant_or_self(node_id, ancestor_id, parent_by_node):
    node_id = str(node_id).strip()
    ancestor_id = str(ancestor_id).strip()
    cur = node_id
    seen = set()

    while cur is not None:
        if cur == ancestor_id:
            return True
        if cur in seen:
            raise ValueError(f"Cycle detected in kc_knowledge_tree.json around node {cur}")
        seen.add(cur)
        cur = parent_by_node.get(cur, None)

    return False


def _node_path(node_id, tree_maps):
    node_id = str(node_id).strip()
    parent_by_node = tree_maps["parent_by_node"]
    name_by_node = tree_maps["name_by_node"]

    parts = []
    cur = node_id
    seen = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        name = name_by_node.get(cur, str(cur))
        if name:
            parts.append(name)
        cur = parent_by_node.get(cur, None)

    return "----".join(reversed(parts))


def _active_raw_to_tree_node(active_raw_id, tree_maps):
    """
    Active raw ids are either:
    - internal node_id selected by roll-up
    - original leaf kc_id if the leaf was not rolled-up
    """
    raw = str(active_raw_id).strip()
    if raw in tree_maps["node_map"]:
        return raw
    if raw in tree_maps["leaf_kcid_to_nodeid"]:
        return tree_maps["leaf_kcid_to_nodeid"][raw]
    return None


def _resolve_rollup_active_raw_id(leaf_concept_id, rollup_node_ids, tree_maps, conflict_policy="deepest"):
    """
    Map one original leaf kc id to active raw id.

    If the leaf is under one or more selected roll-up nodes, return the selected
    roll-up node_id. If not, return the original leaf kc_id.
    """
    leaf_raw = str(leaf_concept_id).strip()
    leaf_node_id = _resolve_leaf_concept_to_tree_node(leaf_raw, tree_maps)

    matched = []
    for rid in rollup_node_ids:
        rid = str(rid).strip()
        if rid not in tree_maps["node_map"]:
            raise ValueError(f"rollup_node_id={rid} not found in kc_knowledge_tree.json")
        if _is_descendant_or_self(leaf_node_id, rid, tree_maps["parent_by_node"]):
            matched.append(rid)

    if not matched:
        return leaf_raw

    conflict_policy = str(conflict_policy).lower().strip()
    if conflict_policy != "deepest":
        raise ValueError(
            f"Unsupported conflict_policy={conflict_policy}. Current safe implementation supports only 'deepest'."
        )

    matched.sort(key=lambda x: tree_maps["depth_by_node"].get(x, 0), reverse=True)
    return matched[0]


def _apply_rollup_to_concept_string(concept_str, rollup_mapper):
    """
    Convert "leaf1_leaf2" -> "active1_active2", preserving order and removing duplicates.
    """
    active = []
    for raw_leaf in str(concept_str).split("_"):
        raw_leaf = raw_leaf.strip()
        if raw_leaf == "":
            continue
        active.append(rollup_mapper.get(raw_leaf, raw_leaf))
    return "_".join(_dedupe_keep_order(active))


def build_rollup_artifacts(
    qid_to_leaf_concepts,
    kc_tree_path,
    rollup_node_ids,
    conflict_policy="deepest",
):
    """
    Build all information needed for preprocessing-time KC roll-up.

    Returns a dict containing:
    - qid_to_active_concepts: question -> active concept string
    - leaf_raw_to_active_raw: leaf kc id -> active raw id
    - active_raw_ids: all active ids appearing in qid_to_active_concepts
    - tree_maps: parsed tree helper maps
    - meta_core: reproducibility metadata
    """
    rollup_node_ids = _parse_rollup_node_ids(rollup_node_ids)
    if not rollup_node_ids:
        return {
            "enabled": False,
            "qid_to_active_concepts": dict(qid_to_leaf_concepts),
            "leaf_raw_to_active_raw": {},
            "active_raw_ids": sorted(
                {leaf for c in qid_to_leaf_concepts.values() for leaf in str(c).split("_") if leaf.strip()},
                key=_safe_int_sort_key,
            ),
            "tree_maps": None,
            "meta_core": {},
        }

    if not kc_tree_path:
        raise ValueError("rollup_node_ids is set, so kc_tree_path / kc_knowledge_tree.json is required.")

    tree_data = _load_json(kc_tree_path)
    tree_maps = _build_tree_maps(tree_data)

    for rid in rollup_node_ids:
        if rid not in tree_maps["node_map"]:
            raise ValueError(f"rollup_node_id={rid} not found in kc_knowledge_tree.json")

    all_leaf_raw_ids = sorted(
        {leaf.strip() for c in qid_to_leaf_concepts.values() for leaf in str(c).split("_") if leaf.strip()},
        key=_safe_int_sort_key,
    )

    leaf_raw_to_active_raw = {}
    rollup_to_covered_leaves = {str(rid): [] for rid in rollup_node_ids}

    for leaf_raw in all_leaf_raw_ids:
        active_raw = _resolve_rollup_active_raw_id(
            leaf_concept_id=leaf_raw,
            rollup_node_ids=rollup_node_ids,
            tree_maps=tree_maps,
            conflict_policy=conflict_policy,
        )
        leaf_raw_to_active_raw[leaf_raw] = active_raw
        if active_raw in rollup_to_covered_leaves and active_raw != leaf_raw:
            rollup_to_covered_leaves[active_raw].append(leaf_raw)

    qid_to_active_concepts = {}
    for qid, concept_str in qid_to_leaf_concepts.items():
        qid_to_active_concepts[qid] = _apply_rollup_to_concept_string(concept_str, leaf_raw_to_active_raw)

    active_raw_ids = sorted(
        {leaf for c in qid_to_active_concepts.values() for leaf in str(c).split("_") if leaf.strip()},
        key=_safe_int_sort_key,
    )

    active_rollup_nodes_used = sorted(
        [rid for rid, covered in rollup_to_covered_leaves.items() if len(covered) > 0],
        key=_safe_int_sort_key,
    )

    meta_core = {
        "mode": "rollup",
        "rollup_node_ids": [int(x) if str(x).isdigit() else x for x in rollup_node_ids],
        "conflict_policy": conflict_policy,
        "num_c_original_leaf_from_questions": len(all_leaf_raw_ids),
        "num_c_active_from_questions": len(active_raw_ids),
        "num_requested_rollup_nodes": len(rollup_node_ids),
        "num_active_rollup_nodes_used": len(active_rollup_nodes_used),
        "num_covered_leaf_nodes_from_questions": sum(len(v) for v in rollup_to_covered_leaves.values()),
        "active_rollup_nodes_used": [int(x) if str(x).isdigit() else x for x in active_rollup_nodes_used],
        "rollup_to_covered_leaf_raw_ids": {
            rid: sorted(leaves, key=_safe_int_sort_key)
            for rid, leaves in rollup_to_covered_leaves.items()
            if len(leaves) > 0
        },
        "leaf_raw_to_active_raw": leaf_raw_to_active_raw,
    }

    return {
        "enabled": True,
        "qid_to_active_concepts": qid_to_active_concepts,
        "leaf_raw_to_active_raw": leaf_raw_to_active_raw,
        "active_raw_ids": active_raw_ids,
        "tree_maps": tree_maps,
        "meta_core": meta_core,
    }


def _build_active_keyid2idx(active_raw_ids, tree_maps=None):
    """
    Build a keyid2idx.json-like dict for active concepts only.

    This file is intentionally concept-focused. If your later global
    data_preprocess.py also creates question/user mappings, let that stage
    merge/overwrite as needed.
    """
    active_raw_ids = sorted({str(x).strip() for x in active_raw_ids if str(x).strip()}, key=_safe_int_sort_key)
    concepts = {raw_id: idx for idx, raw_id in enumerate(active_raw_ids)}

    out = {
        "concepts": concepts,
        "num_c": len(concepts),
    }

    if tree_maps is not None:
        concept_names = {}
        concept_types = {}
        concept_paths = {}
        for raw_id in active_raw_ids:
            node_id = _active_raw_to_tree_node(raw_id, tree_maps)
            if node_id is None:
                concept_names[raw_id] = raw_id
                concept_types[raw_id] = "unknown"
                concept_paths[raw_id] = raw_id
            else:
                concept_names[raw_id] = tree_maps["name_by_node"].get(node_id, raw_id)
                concept_types[raw_id] = tree_maps["type_by_node"].get(node_id, "unknown")
                concept_paths[raw_id] = _node_path(node_id, tree_maps)

        out["concept_names"] = concept_names
        out["concept_types"] = concept_types
        out["concept_paths"] = concept_paths

    return out


def _copy_tree_to_output(kc_tree_path, output_dir):
    if not kc_tree_path:
        return None
    os.makedirs(output_dir, exist_ok=True)
    dst = os.path.join(output_dir, "kc_knowledge_tree.json")
    if os.path.abspath(kc_tree_path) != os.path.abspath(dst):
        shutil.copy2(kc_tree_path, dst)
    return dst


# -----------------------------
# Main preprocessing function
# -----------------------------
def read_data_from_csv(
    read_file,
    write_file,
    questions_json_path=None,
    kc_routes_map_json_path=None,
    kc_tree_path=None,
    rollup_node_ids=None,
    rollup_conflict_policy="deepest",
    output_dataset_dir=None,
    write_keyid2idx=True,
    write_rollup_meta=True,
    copy_kc_tree_to_output=True,
):
    """
    XES3G5M preprocessing.

    Important behavior:
    - concepts are NOT taken from the original CSV anymore.
    - For each question id, this function looks up questions.json[question_id]['kc_routes'].
    - For every route, it takes the final route segment as the leaf KC name.
    - Then it maps that KC name to id through kc_routes_map.json.
    - If a question has multiple routes, concepts becomes "id1_id2_...".

    Roll-up behavior:
    - If rollup_node_ids is empty/None, output is leaf-only, same as before.
    - If rollup_node_ids is provided, each leaf concept id is replaced by the
      deepest selected ancestor that covers it.
    - The output sequence file directly contains active raw KC ids.
    - keyid2idx.json, if written, contains only active concepts.
    - kc_rollup_meta.json records the exact mapping for reproducibility.

    Recommended usage:
        read_data_from_csv(
            read_file=".../assist2017_tree/question_level/train_valid_sequences_quelevel.csv",
            write_file=".../assist2017_tree_rollup_100016_100280/train_valid_sequences.csv",
            kc_tree_path=".../assist2017_tree/kc_knowledge_tree.json",
            rollup_node_ids="100016,100280",
        )

    Input CSV still needs:
    - uid
    - questions
    - responses

    Optional columns:
    - timestamps
    - selectmasks
    """
    if output_dataset_dir:
        os.makedirs(output_dataset_dir, exist_ok=True)
        # If caller passed only a filename as write_file, place it under output_dataset_dir.
        if not os.path.dirname(str(write_file)):
            write_file = os.path.join(output_dataset_dir, str(write_file))

    write_dir = os.path.abspath(os.path.dirname(str(write_file)))
    if write_dir:
        os.makedirs(write_dir, exist_ok=True)

    questions_json_path = _find_json_file(read_file, "questions.json", questions_json_path)
    kc_routes_map_json_path = _find_json_file(read_file, "kc_routes_map.json", kc_routes_map_json_path)

    rollup_node_ids_parsed = _parse_rollup_node_ids(rollup_node_ids)
    rollup_enabled = len(rollup_node_ids_parsed) > 0

    if rollup_enabled:
        kc_tree_path = _find_json_file(read_file, "kc_knowledge_tree.json", kc_tree_path)
    elif kc_tree_path:
        kc_tree_path = _find_json_file(read_file, "kc_knowledge_tree.json", kc_tree_path)

    qid_to_leaf_concepts = build_question_to_concepts(questions_json_path, kc_routes_map_json_path)

    rollup_artifacts = build_rollup_artifacts(
        qid_to_leaf_concepts=qid_to_leaf_concepts,
        kc_tree_path=kc_tree_path,
        rollup_node_ids=rollup_node_ids_parsed,
        conflict_policy=rollup_conflict_policy,
    )
    qid_to_concepts = rollup_artifacts["qid_to_active_concepts"]

    # Build a unified raw pool before re-splitting:
    # include both train_valid_sequences_quelevel.csv and test_quelevel.csv if available.
    read_files = [read_file]
    norm_path = str(read_file).replace("\\", "/")
    if norm_path.endswith("/question_level/train_valid_sequences_quelevel.csv"):
        qlevel_dir = os.path.dirname(str(read_file))
        test_path = os.path.join(qlevel_dir, "test_quelevel.csv")
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                "Expected paired test file not found for xes3g5m preprocessing: "
                f"{test_path}"
            )
        if os.path.abspath(test_path) != os.path.abspath(read_file):
            read_files.append(test_path)

    dfs = [pd.read_csv(fp, encoding="utf-8", low_memory=False) for fp in read_files]
    df = pd.concat(dfs, ignore_index=True)

    # concepts is intentionally removed from required columns.
    # We rebuild concepts from questions.json + kc_routes_map.json, then optionally apply roll-up.
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
    uniq_concepts_raw_tokens = set()
    uniq_concepts_active = set()
    uniq_concepts_leaf_before_rollup = set()
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

            active_concept_str = qid_to_concepts[qid]
            leaf_concept_str = qid_to_leaf_concepts[qid]

            q2.append(qid)
            c2.append(active_concept_str)
            r2.append(str(int(responses[i])))
            if has_timestamps:
                t2.append(timestamps[i] if i < len(timestamps) else "NA")

            for leaf in str(leaf_concept_str).split("_"):
                leaf = leaf.strip()
                if leaf and leaf not in {"-1", "NA"}:
                    uniq_concepts_leaf_before_rollup.add(leaf)

        if len(q2) == 0:
            continue

        valid_rows += 1

        uniq_users.add(uid)
        uniq_questions.update(q2)
        uniq_concepts_raw_tokens.update(c2)
        for c in c2:
            for active in str(c).split("_"):
                active = active.strip()
                if active and active not in {"-1", "NA"}:
                    uniq_concepts_active.add(active)

        row_ts_key = _ts_sort_value(t2[0]) if (has_timestamps and len(t2) > 0) else ridx
        user_chunks.setdefault(uid, []).append((row_ts_key, q2, c2, r2, t2))

    if missing_qids:
        examples = sorted(missing_qids, key=_safe_int_sort_key)[:20]
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

    tree_maps = rollup_artifacts["tree_maps"]
    keyid2idx_path = None
    rollup_meta_path = None
    copied_tree_path = None

    if copy_kc_tree_to_output and kc_tree_path:
        copied_tree_path = _copy_tree_to_output(kc_tree_path, write_dir)

    active_ids_for_observed_sequences = sorted(uniq_concepts_active, key=_safe_int_sort_key)

    if write_keyid2idx:
        # In roll-up mode this is the active vocab. In leaf-only mode this is the observed leaf vocab.
        # Downstream pyKT can either use this directly or regenerate the same mapping from the output sequences.
        keyid2idx = _build_active_keyid2idx(active_ids_for_observed_sequences, tree_maps=tree_maps)
        keyid2idx_path = os.path.join(write_dir, "keyid2idx.json")
        _write_json(keyid2idx_path, keyid2idx)

    if write_rollup_meta and rollup_enabled:
        observed_leaf_to_active = {}
        for leaf in sorted(uniq_concepts_leaf_before_rollup, key=_safe_int_sort_key):
            active = rollup_artifacts["leaf_raw_to_active_raw"].get(leaf, leaf)
            observed_leaf_to_active[leaf] = active

        # This is useful if your old leaf-only keyid2idx has ids 0..num_leaf-1 where raw leaf id == dense idx.
        # Do NOT use it as dense mapping unless that assumption holds in your base dataset.
        active_keyid2idx = _build_active_keyid2idx(active_ids_for_observed_sequences, tree_maps=tree_maps)
        active_rawid2idx = active_keyid2idx["concepts"]
        old_leaf_raw_to_new_active_idx = {
            leaf: active_rawid2idx[active]
            for leaf, active in observed_leaf_to_active.items()
            if active in active_rawid2idx
        }

        meta = {
            **rollup_artifacts["meta_core"],
            "source_files": {
                "read_files": read_files,
                "questions_json": questions_json_path,
                "kc_routes_map_json": kc_routes_map_json_path,
                "kc_knowledge_tree": kc_tree_path,
            },
            "output_files": {
                "sequence_file": write_file,
                "keyid2idx": keyid2idx_path,
                "kc_knowledge_tree": copied_tree_path,
            },
            "num_c_original_leaf_observed_sequences": len(uniq_concepts_leaf_before_rollup),
            "num_c_active_observed_sequences": len(active_ids_for_observed_sequences),
            "leaf_raw_to_active_raw_observed_sequences": observed_leaf_to_active,
            "active_rawid2idx_observed_sequences": active_rawid2idx,
            "old_leaf_raw_to_new_active_idx_observed_sequences": old_leaf_raw_to_new_active_idx,
            "notes": [
                "Sequence concepts have already been remapped to active raw KC ids.",
                "keyid2idx.json contains only active concepts for this output dataset folder.",
                "data_config['num_c'] should equal len(keyid2idx['concepts']).",
                "If using cached .pkl files, delete them before training this variant.",
            ],
        }
        rollup_meta_path = os.path.join(write_dir, "kc_rollup_meta.json")
        _write_json(rollup_meta_path, meta)

    avg_ins = round(total_interactions / len(uniq_users), 4) if uniq_users else 0.0
    print(
        "after xes3g5m preprocess, "
        f"source files: {len(read_files)}, "
        f"questions_json: {questions_json_path}, "
        f"kc_routes_map_json: {kc_routes_map_json_path}, "
        f"kc_tree_path: {kc_tree_path}, "
        f"rollup_enabled: {rollup_enabled}, "
        f"rollup_node_ids: {rollup_node_ids_parsed}, "
        f"interaction num: {total_interactions}, "
        f"user num: {len(uniq_users)}, "
        f"question num: {len(uniq_questions)}, "
        f"concept num before rollup observed: {len(uniq_concepts_leaf_before_rollup)}, "
        f"concept num active observed: {len(active_ids_for_observed_sequences)}, "
        f"concept num raw multi-concept token active: {len(uniq_concepts_raw_tokens)}, "
        f"avg(ins) per s: {avg_ins}, "
        f"seq rows kept: {valid_rows}, "
        f"uid merged sequences: {len(user_inter)}, "
        f"write_file: {write_file}, "
        f"keyid2idx_path: {keyid2idx_path}, "
        f"rollup_meta_path: {rollup_meta_path}"
    )

    return {
        "write_file": write_file,
        "keyid2idx_path": keyid2idx_path,
        "rollup_meta_path": rollup_meta_path,
        "kc_tree_path": copied_tree_path or kc_tree_path,
        "num_c_before_rollup_observed": len(uniq_concepts_leaf_before_rollup),
        "num_c_active_observed": len(active_ids_for_observed_sequences),
        "rollup_enabled": rollup_enabled,
        "rollup_node_ids": rollup_node_ids_parsed,
    }
