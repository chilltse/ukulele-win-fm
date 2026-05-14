import os
import sys
import pandas as pd
import numpy as np
import json
import copy

ALL_KEYS = ["fold", "uid", "questions", "concepts", "concepts_dense", "responses", "timestamps",
            "usetimes", "selectmasks", "is_repeat", "qidxs", "rest", "orirow", "cidxs"]
ONE_KEYS = ["fold", "uid"]


def read_data(fname, min_seq_len=3, response_set=[0, 1]):
    effective_keys = set()
    dres = dict()
    delstu, delnum, badr = 0, 0, 0
    goodnum = 0
    with open(fname, "r", encoding="utf8") as fin:
        i = 0
        lines = fin.readlines()
        dcur = dict()
        while i < len(lines):
            line = lines[i].strip()
            if i % 6 == 0:  # stuid
                effective_keys.add("uid")
                tmps = line.split(",")
                if "(" in tmps[0]:
                    stuid, seq_len = tmps[0].replace('(', ''), int(tmps[2])
                else:
                    stuid, seq_len = tmps[0], int(tmps[1])
                if seq_len < min_seq_len:  # delete use seq len less than min_seq_len
                    i += 6
                    dcur = dict()
                    delstu += 1
                    delnum += seq_len
                    continue
                dcur["uid"] = stuid
                goodnum += seq_len
            elif i % 6 == 1:  # question ids / names
                qs = []
                if line.find("NA") == -1:
                    effective_keys.add("questions")
                    qs = line.split(",")
                dcur["questions"] = qs
            elif i % 6 == 2:  # concept ids / names
                cs = []
                if line.find("NA") == -1:
                    effective_keys.add("concepts")
                    cs = line.split(",")
                dcur["concepts"] = cs
            elif i % 6 == 3:  # responses
                effective_keys.add("responses")
                rs = []
                if line.find("NA") == -1:
                    flag = True
                    for r in line.split(","):
                        try:
                            r = int(r)
                            if r not in response_set:  # check if r in response set.
                                print(f"error response in line: {i}")
                                flag = False
                                break
                            rs.append(r)
                        except:
                            print(f"error response in line: {i}")
                            flag = False
                            break
                    if not flag:
                        i += 3
                        dcur = dict()
                        badr += 1
                        continue
                dcur["responses"] = rs
            elif i % 6 == 4:  # timestamps
                ts = []
                if line.find("NA") == -1:
                    effective_keys.add("timestamps")
                    ts = line.split(",")
                dcur["timestamps"] = ts
            elif i % 6 == 5:  # usets
                usets = []
                if line.find("NA") == -1:
                    effective_keys.add("usetimes")
                    usets = line.split(",")
                dcur["usetimes"] = usets

                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key != "uid":
                        dres[key].append(",".join([str(k) for k in dcur[key]]))
                    else:
                        dres[key].append(dcur[key])
                dcur = dict()
            i += 1
    df = pd.DataFrame(dres)
    print(
        f"delete bad stu num of len: {delstu}, delete interactions: {delnum}, of r: {badr}, good num: {goodnum}")
    return df, effective_keys


def extend_multi_concepts(df, effective_keys):
    if "questions" not in effective_keys or "concepts" not in effective_keys:
        print("has no questions or concepts! return original.")
        return df, effective_keys
    extend_keys = set(df.columns) - {"uid"}

    dres = {"uid": df["uid"]}
    for _, row in df.iterrows():
        dextend_infos = dict()
        for key in extend_keys:
            dextend_infos[key] = row[key].split(",")
        dextend_res = dict()
        for i in range(len(dextend_infos["questions"])):
            dextend_res.setdefault("is_repeat", [])
            if dextend_infos["concepts"][i].find("_") != -1:
                ids = dextend_infos["concepts"][i].split("_")
                dextend_res.setdefault("concepts", [])
                dextend_res["concepts"].extend(ids)
                for key in extend_keys:
                    if key != "concepts":
                        dextend_res.setdefault(key, [])
                        dextend_res[key].extend(
                            [dextend_infos[key][i]] * len(ids))
                dextend_res["is_repeat"].extend(
                    ["0"] + ["1"] * (len(ids) - 1))  # 1: repeat, 0: original
            else:
                for key in extend_keys:
                    dextend_res.setdefault(key, [])
                    dextend_res[key].append(dextend_infos[key][i])
                dextend_res["is_repeat"].append("0")
        for key in dextend_res:
            dres.setdefault(key, [])
            dres[key].append(",".join(dextend_res[key]))

    finaldf = pd.DataFrame(dres)
    effective_keys.add("is_repeat")
    return finaldf, effective_keys


def id_mapping(df):
    id_keys = ["questions", "concepts", "uid"]
    dres = dict()
    dkeyid2idx = dict()
    print(f"df.columns: {df.columns}")
    for key in df.columns:
        if key not in id_keys:
            dres[key] = df[key]
    for i, row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            dkeyid2idx.setdefault(key, dict())
            dres.setdefault(key, [])
            curids = []
            for id in row[key].split(","):
                if id not in dkeyid2idx[key]:
                    dkeyid2idx[key][id] = len(dkeyid2idx[key])
                curids.append(str(dkeyid2idx[key][id]))
            dres[key].append(",".join(curids))
    finaldf = pd.DataFrame(dres)
    return finaldf, dkeyid2idx


def id_mapping_fmkc(df):
    """Map multi-field concepts raw0|raw1|... -> ids per field; CSV cell i0^i1^..."""
    id_keys = ["questions", "concepts", "uid"]
    dres = dict()
    dkeyid2idx = {"concepts_dense": {}}
    num_fields = None
    print(f"df.columns (fmkc): {df.columns}")
    for key in df.columns:
        if key not in id_keys:
            dres[key] = df[key]
    for i, row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            if key == "concepts":
                dres.setdefault("concepts", [])
                dres.setdefault("concepts_dense", [])
                new_cs = []
                dense_cs = []
                for token in row["concepts"].split(","):
                    parts = token.split("|")
                    if num_fields is None:
                        num_fields = len(parts)
                        dkeyid2idx["concepts_fmkc"] = [{} for _ in range(num_fields)]
                    if len(parts) != num_fields:
                        raise ValueError(
                            f"kc_fmkc expects {num_fields} fields separated by | in each token, "
                            f"got {len(parts)} in {token!r}"
                        )
                    ids = []
                    for j in range(num_fields):
                        field_d = dkeyid2idx["concepts_fmkc"][j]
                        p = parts[j]
                        if p not in field_d:
                            field_d[p] = len(field_d)
                        ids.append(str(field_d[p]))
                    new_cs.append("^".join(ids))
                    if token not in dkeyid2idx["concepts_dense"]:
                        dkeyid2idx["concepts_dense"][token] = len(dkeyid2idx["concepts_dense"])
                    dense_cs.append(str(dkeyid2idx["concepts_dense"][token]))
                dres["concepts"].append(",".join(new_cs))
                dres["concepts_dense"].append(",".join(dense_cs))
                continue
            dkeyid2idx.setdefault(key, dict())
            dres.setdefault(key, [])
            curids = []
            for id in row[key].split(","):
                if id not in dkeyid2idx[key]:
                    dkeyid2idx[key][id] = len(dkeyid2idx[key])
                curids.append(str(dkeyid2idx[key][id]))
            dres[key].append(",".join(curids))
    finaldf = pd.DataFrame(dres)
    return finaldf, dkeyid2idx

def train_test_split(df, test_ratio=0.2):
    df = df.sample(frac=1.0, random_state=1024)
    datanum = df.shape[0]
    test_num = int(datanum * test_ratio)
    train_num = datanum - test_num
    train_df = df[0:train_num]
    test_df = df[train_num:]
    # report
    print(
        f"total num: {datanum}, train+valid num: {train_num}, test num: {test_num}")
    return train_df, test_df


def KFold_split(df, k=5):
    df = df.sample(frac=1.0, random_state=1024)
    datanum = df.shape[0]
    test_ratio = 1 / k
    test_num = int(datanum * test_ratio)
    rest = datanum % k

    start = 0
    folds = []
    for i in range(0, k):
        if rest > 0:
            end = start + test_num + 1
            rest -= 1
        else:
            end = start + test_num
        folds.extend([i] * (end - start))
        print(f"fold: {i+1}, start: {start}, end: {end}, total num: {datanum}")
        start = end
    # report
    finaldf = copy.deepcopy(df)
    finaldf["fold"] = folds
    return finaldf


def save_dcur(row, effective_keys):
    dcur = dict()
    for key in effective_keys:
        if key not in ONE_KEYS:
            # [int(i) for i in row[key].split(",")]
            dcur[key] = row[key].split(",")
        else:
            dcur[key] = row[key]
    return dcur


def generate_sequences(df, effective_keys, min_seq_len=3, maxlen=200, pad_val=-1):
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    dropnum = 0
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)

        rest, lenrs = len(dcur["responses"]), len(dcur["responses"])
        j = 0
        while lenrs >= j + maxlen:
            rest = rest - (maxlen)
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    # [str(k) for k in dcur[key][j: j + maxlen]]))
                    dres[key].append(",".join(dcur[key][j: j + maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))

            j += maxlen
        if rest < min_seq_len:  # delete sequence len less than min_seq_len
            dropnum += rest
            continue

        pad_dim = maxlen - rest
        for key in effective_keys:
            dres.setdefault(key, [])
            if key not in ONE_KEYS:
                paded_info = np.concatenate(
                    [dcur[key][j:], np.array([pad_val] * pad_dim)])
                dres[key].append(",".join([str(k) for k in paded_info]))
            else:
                dres[key].append(dcur[key])
        dres["selectmasks"].append(
            ",".join(["1"] * rest + [str(pad_val)] * pad_dim))

    # after preprocess data, report
    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    print(f"dropnum: {dropnum}")
    return finaldf


def generate_window_sequences(df, effective_keys, maxlen=200, pad_val=-1):
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)
        lenrs = len(dcur["responses"])
        if lenrs > maxlen:
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    # [str(k) for k in dcur[key][0: maxlen]]))
                    dres[key].append(",".join(dcur[key][0: maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))
            for j in range(maxlen+1, lenrs+1):
                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key not in ONE_KEYS:
                        dres[key].append(",".join([str(k)
                                         for k in dcur[key][j-maxlen: j]]))
                    else:
                        dres[key].append(dcur[key])
                dres["selectmasks"].append(
                    ",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
        else:
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    pad_dim = maxlen - lenrs
                    paded_info = np.concatenate(
                        [dcur[key][0:], np.array([pad_val] * pad_dim)])
                    dres[key].append(",".join([str(k) for k in paded_info]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(
                ",".join(["1"] * lenrs + [str(pad_val)] * pad_dim))

    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            # print(f"key: {key}, len: {len(dres[key])}")
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    return finaldf


def get_inter_qidx(df):
    """add global id for each interaction"""
    qidx_ids = []
    bias = 0
    inter_num = 0
    for _, row in df.iterrows():
        ids_list = [str(x+bias)
                    for x in range(len(row['responses'].split(',')))]
        inter_num += len(ids_list)
        ids = ",".join(ids_list)
        qidx_ids.append(ids)
        bias += len(ids_list)
    assert inter_num-1 == int(ids_list[-1])

    return qidx_ids


def add_qidx(dcur, global_qidx):
    idxs, rests = [], []
    # idx = -1
    for r in dcur["is_repeat"]:
        if str(r) == "0":
            global_qidx += 1
        idxs.append(global_qidx)
    # print(dcur["is_repeat"])
    # print(f"idxs: {idxs}")
    # print("="*20)
    for i in range(0, len(idxs)):
        rests.append(idxs[i+1:].count(idxs[i]))
    return idxs, rests, global_qidx


def expand_question(dcur, global_qidx, pad_val=-1):
    dextend, dlast = dict(), dict()
    repeats = dcur["is_repeat"]
    last = -1
    dcur["qidxs"], dcur["rest"], global_qidx = add_qidx(dcur, global_qidx)
    for i in range(len(repeats)):
        if str(repeats[i]) == "0":
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dlast[key] = dcur[key][0: i]
        if i == 0:
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dextend.setdefault(key, [])
                dextend[key].append([dcur[key][0]])
            dextend.setdefault("selectmasks", [])
            dextend["selectmasks"].append([pad_val])
        else:
            # print(f"i: {i}, dlast: {dlast.keys()}")
            for key in dcur.keys():
                if key in ONE_KEYS:
                    continue
                dextend.setdefault(key, [])
                if last == "0" and str(repeats[i]) == "0":
                    dextend[key][-1] += [dcur[key][i]]
                else:
                    dextend[key].append(dlast[key] + [dcur[key][i]])
            dextend.setdefault("selectmasks", [])
            if last == "0" and str(repeats[i]) == "0":
                dextend["selectmasks"][-1] += [1]
            elif len(dlast["responses"]) == 0:  # the first question
                dextend["selectmasks"].append([pad_val])
            else:
                dextend["selectmasks"].append(
                    len(dlast["responses"]) * [pad_val] + [1])

        last = str(repeats[i])

    return dextend, global_qidx


def generate_question_sequences(df, effective_keys, window=True, min_seq_len=3, maxlen=200, pad_val=-1):
    if "questions" not in effective_keys or "concepts" not in effective_keys:
        print(f"has no questions or concepts, has no question sequences!")
        return False, None
    save_keys = list(effective_keys) + \
        ["selectmasks", "qidxs", "rest", "orirow"]
    dres = {}  # "selectmasks": []}
    global_qidx = -1
    df["index"] = list(range(0, df.shape[0]))
    for i, row in df.iterrows():
        dcur = save_dcur(row, effective_keys)
        dcur["orirow"] = [row["index"]] * len(dcur["responses"])

        dexpand, global_qidx = expand_question(dcur, global_qidx)
        seq_num = len(dexpand["responses"])
        for j in range(seq_num):
            curlen = len(dexpand["responses"][j])
            if curlen < 2:  # 不预测第一个题
                continue
            if curlen < maxlen:
                for key in dexpand:
                    pad_dim = maxlen - curlen
#                     print(key, j, len(dexpand[key]))
                    paded_info = np.concatenate(
                        [dexpand[key][j][0:], np.array([pad_val] * pad_dim)])
                    dres.setdefault(key, [])
                    dres[key].append(",".join([str(k) for k in paded_info]))
                for key in ONE_KEYS:
                    dres.setdefault(key, [])
                    dres[key].append(dcur[key])
            else:
                # window
                if window:
                    if dexpand["selectmasks"][j][maxlen-1] == 1:
                        for key in dexpand:
                            dres.setdefault(key, [])
                            dres[key].append(
                                ",".join([str(k) for k in dexpand[key][j][0:maxlen]]))
                        for key in ONE_KEYS:
                            dres.setdefault(key, [])
                            dres[key].append(dcur[key])

                    for n in range(maxlen+1, curlen+1):
                        if dexpand["selectmasks"][j][n-1] == 1:
                            for key in dexpand:
                                dres.setdefault(key, [])
                                if key == "selectmasks":
                                    dres[key].append(
                                        ",".join([str(pad_val)] * (maxlen - 1) + ["1"]))
                                else:
                                    dres[key].append(
                                        ",".join([str(k) for k in dexpand[key][j][n-maxlen: n]]))
                            for key in ONE_KEYS:
                                dres.setdefault(key, [])
                                dres[key].append(dcur[key])
                else:
                    # not window
                    k = 0
                    rest = curlen
                    while curlen >= k + maxlen:
                        rest = rest - maxlen
                        if dexpand["selectmasks"][j][k + maxlen - 1] == 1:
                            for key in dexpand:
                                dres.setdefault(key, [])
                                dres[key].append(
                                    ",".join([str(s) for s in dexpand[key][j][k: k + maxlen]]))
                            for key in ONE_KEYS:
                                dres.setdefault(key, [])
                                dres[key].append(dcur[key])
                        k += maxlen
                    if rest < min_seq_len:  # 剩下长度<min_seq_len不预测
                        continue
                    pad_dim = maxlen - rest
                    for key in dexpand:
                        dres.setdefault(key, [])
                        paded_info = np.concatenate(
                            [dexpand[key][j][k:], np.array([pad_val] * pad_dim)])
                        dres[key].append(",".join([str(s)
                                         for s in paded_info]))
                    for key in ONE_KEYS:
                        dres.setdefault(key, [])
                        dres[key].append(dcur[key])
                #####

    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            # print(f"key: {key}, len: {len(dres[key])}")
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    return True, finaldf


def save_id2idx(dkeyid2idx, save_path):
    with open(save_path, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(dkeyid2idx, ensure_ascii=False, indent=4))


def _find_existing_path(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return ""


def resolve_kc_tree_path(dname, dataset_name, explicit_path=None):
    """Resolve the slim tree JSON path for *_tree datasets.

    The preferred file is kc_knowledge_tree_slim.json, but several fallback
    locations are supported to avoid breaking existing dataset layouts.
    """
    if explicit_path:
        return explicit_path if os.path.exists(explicit_path) else ""

    candidates = []
    if dataset_name in ["dbe_kt22_tree", "dbe_kt22"]:
        candidates.extend([
            os.path.join(dname, "2_DBE_KT22_datafiles_100102_csv", "kc_knowledge_tree_slim.json"),
            os.path.join(dname, "2_DBE_KT22_datafiles_100102_csv", "kc_knowledge_tree_original.json"),
        ])
    if dataset_name == "xes3g5m_tree":
        candidates.extend([
            os.path.join(dname, "metadata", "kc_knowledge_tree_slim.json"),
            os.path.join(dname, "metadata", "kc_knowledge_tree_original.json"),
        ])

    candidates.extend([
        os.path.join(dname, "kc_knowledge_tree_slim.json"),
        os.path.join(dname, "kc_knowledge_tree_original.json"),
    ])
    return _find_existing_path(candidates)


def load_tree_root(kc_tree_path):
    """Load either a pure-node slim JSON or a metadata-wrapped tree JSON."""
    with open(kc_tree_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "tree" in data and isinstance(data["tree"], dict):
        root = data["tree"]
        wrapped = data
    elif isinstance(data, dict) and "children" in data:
        root = data
        wrapped = None
    else:
        raise ValueError(
            f"Unsupported KC tree JSON format: {kc_tree_path}. "
            "Expected either a pure root node or a dict containing `tree`."
        )
    return root, wrapped


def iter_tree_nodes(root):
    stack = [(root, None)]
    while stack:
        node, parent = stack.pop()
        yield node, parent
        for child in reversed(node.get("children", []) or []):
            stack.append((child, node))


def build_tree_artifacts(kc_tree_path, pad_val=-1):
    """Build tree artifacts from kc_knowledge_tree_slim.json.

    Supported input JSON formats:
      1) pure node tree: {node_id, type, name, parent_id, children}
      2) pure node tree with explicit tree_idx fields
      3) wrapper: {tree, parent_index, node_index, keyid2idx_tree, ...}

    If tree_idx is missing, compact tree indices are generated automatically
    from the traversal order. Therefore the JSON can stay as a clean node-only
    tree file.

    Returns:
      keyid2idx_tree: full node_id -> tree_idx and leaf kc_id -> tree_idx
      parent_index: parent_index[child_tree_idx] = parent_tree_idx or -1
      node_index: node metadata indexed by tree_idx
      tree_report: validation and coverage information
    """
    root, wrapped = load_tree_root(kc_tree_path)

    nodes = []
    for node, parent in iter_tree_nodes(root):
        if "node_id" not in node:
            raise ValueError(f"Tree node missing node_id: {node}")
        nodes.append((node, parent))

    if not nodes:
        raise ValueError(f"Tree JSON contains no nodes: {kc_tree_path}")

    has_tree_idx = ["tree_idx" in node for node, _ in nodes]
    if any(has_tree_idx) and not all(has_tree_idx):
        bad = [node.get("name") for node, _ in nodes if "tree_idx" not in node][:20]
        raise ValueError(
            "Invalid tree JSON: either every node must have tree_idx, or no node should have tree_idx. "
            f"Missing tree_idx examples: {bad}"
        )

    # If the slim JSON only contains node information, generate compact tree_idx
    # deterministically according to the traversal order. This keeps the JSON clean
    # while still producing contiguous embedding indices for qid_tree.
    generated_tree_idx = not all(has_tree_idx)
    if generated_tree_idx:
        tree_idx_by_obj_id = {id(node): i for i, (node, _) in enumerate(nodes)}
        get_tree_idx = lambda node: tree_idx_by_obj_id[id(node)]
    else:
        get_tree_idx = lambda node: int(node["tree_idx"])

    node_id_to_tree_idx = {}
    tree_idx_to_node = {}
    duplicate_node_ids = []
    duplicate_tree_indices = []

    for node, _ in nodes:
        node_id = int(node["node_id"])
        tree_idx = int(get_tree_idx(node))
        if str(node_id) in node_id_to_tree_idx:
            duplicate_node_ids.append(node_id)
        if tree_idx in tree_idx_to_node:
            duplicate_tree_indices.append(tree_idx)
        node_id_to_tree_idx[str(node_id)] = tree_idx
        tree_idx_to_node[tree_idx] = node

    if duplicate_node_ids or duplicate_tree_indices:
        raise ValueError(
            f"Invalid tree JSON: duplicate_node_ids={duplicate_node_ids}, "
            f"duplicate_tree_indices={duplicate_tree_indices}"
        )

    num_c_tree = max(tree_idx_to_node.keys()) + 1 if tree_idx_to_node else 0
    missing_tree_indices = [i for i in range(num_c_tree) if i not in tree_idx_to_node]
    if missing_tree_indices:
        raise ValueError(
            "tree_idx must be contiguous from 0 to num_c-1. "
            f"Missing: {missing_tree_indices[:20]}"
        )

    parent_index = [-1] * num_c_tree
    node_index = [None] * num_c_tree
    original_kc_id_to_tree_idx = {}
    internal_node_id_to_tree_idx = {}
    duplicate_kc_ids = []
    missing_parent_ids = []

    for node, _ in nodes:
        node_id = int(node["node_id"])
        tree_idx = int(get_tree_idx(node))
        parent_id = node.get("parent_id", None)
        node_type = node.get("type", "")
        name = node.get("name", "")

        if parent_id is not None:
            parent_raw = str(int(parent_id))
            if parent_raw not in node_id_to_tree_idx:
                missing_parent_ids.append({"node_id": node_id, "parent_id": int(parent_id), "name": name})
            else:
                parent_index[tree_idx] = int(node_id_to_tree_idx[parent_raw])

        meta = {
            "tree_idx": tree_idx,
            "node_id": node_id,
            "type": node_type,
            "name": name,
            "parent_id": None if parent_id is None else int(parent_id),
            "kc_id": None,
            "is_observed_kc": node_type == "kc_leaf" or node.get("kc_id") is not None,
        }

        if node.get("kc_id", None) is not None:
            kc_id = int(node["kc_id"])
            meta["kc_id"] = kc_id
            if str(kc_id) in original_kc_id_to_tree_idx:
                duplicate_kc_ids.append(kc_id)
            original_kc_id_to_tree_idx[str(kc_id)] = tree_idx
        else:
            internal_node_id_to_tree_idx[str(node_id)] = tree_idx

        node_index[tree_idx] = meta

    if duplicate_kc_ids:
        raise ValueError(f"Invalid tree JSON: duplicate kc_id values: {duplicate_kc_ids[:20]}")
    if missing_parent_ids:
        raise ValueError(f"Invalid tree JSON: missing parent IDs: {missing_parent_ids[:20]}")

    # Cycle check over compact tree indices.
    state = [0] * num_c_tree
    cycle_nodes = []

    def dfs(u):
        if state[u] == 2:
            return
        if state[u] == 1:
            cycle_nodes.append(u)
            return
        state[u] = 1
        p = parent_index[u]
        if 0 <= p < num_c_tree:
            dfs(p)
        state[u] = 2

    for i in range(num_c_tree):
        dfs(i)
    if cycle_nodes:
        raise ValueError(f"Cycle detected in tree parent_index: {cycle_nodes[:20]}")

    keyid2idx_tree = {
        "num_c": num_c_tree,
        "concepts": node_id_to_tree_idx,
        "original_kc_id_to_tree_idx": original_kc_id_to_tree_idx,
        "internal_node_id_to_tree_idx": internal_node_id_to_tree_idx,
    }

    tree_report = {
        "kc_tree_path": kc_tree_path,
        "tree_idx_source": "generated_preorder" if generated_tree_idx else "from_json",
        "num_c_tree": num_c_tree,
        "num_nodes": len(nodes),
        "num_observed_leaf_kcs": len(original_kc_id_to_tree_idx),
        "num_internal_nodes": len(internal_node_id_to_tree_idx),
        "num_edges": sum(1 for p in parent_index if p >= 0),
        "root_indices": [i for i, p in enumerate(parent_index) if p < 0],
        "passed": True,
    }

    return keyid2idx_tree, parent_index, node_index, tree_report


def remap_concepts_to_tree_indices(df, keyid2idx_tree, concepts_col="concepts", pad_val=-1):
    """Remap raw leaf KC ids in df[concepts] to tree_idx values.

    This must run before sequence generation for qid_tree datasets, because the
    Dataset loader reads CSV concepts directly as integer embedding indices.
    """
    if concepts_col not in df.columns:
        return df

    leaf_map = keyid2idx_tree.get("original_kc_id_to_tree_idx", {})
    if not leaf_map:
        raise ValueError("keyid2idx_tree has no original_kc_id_to_tree_idx mapping.")

    df = copy.deepcopy(df)
    missing = set()
    remapped_rows = []

    for _, row in df.iterrows():
        new_tokens = []
        for tok in str(row[concepts_col]).split(","):
            tok = tok.strip()
            if tok == str(pad_val):
                new_tokens.append(str(pad_val))
                continue
            if tok not in leaf_map:
                missing.add(tok)
                new_tokens.append(tok)
            else:
                new_tokens.append(str(leaf_map[tok]))
        remapped_rows.append(",".join(new_tokens))

    if missing:
        raise ValueError(
            "Some observed concept ids are not found as kc_leaf nodes in the tree JSON. "
            f"Missing examples: {sorted(missing)[:20]}"
        )

    df[concepts_col] = remapped_rows
    return df


def id_mapping_tree(df, keyid2idx_tree):
    """Map questions/uids normally, but map concepts with the external tree_idx.

    dkeyid2idx["concepts"] intentionally contains *all* tree nodes, not only
    observed leaves. Therefore num_c for qid_tree becomes the full tree size.
    """
    id_keys = ["questions", "concepts", "uid"]
    dres = dict()
    dkeyid2idx = {
        "concepts": dict(keyid2idx_tree["concepts"]),
        "original_kc_id_to_tree_idx": dict(keyid2idx_tree.get("original_kc_id_to_tree_idx", {})),
        "internal_node_id_to_tree_idx": dict(keyid2idx_tree.get("internal_node_id_to_tree_idx", {})),
    }
    leaf_map = keyid2idx_tree.get("original_kc_id_to_tree_idx", {})

    print(f"df.columns (tree): {df.columns}")
    for key in df.columns:
        if key not in id_keys:
            dres[key] = df[key]

    for _, row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            dres.setdefault(key, [])

            if key == "concepts":
                curids = []
                for raw_id in row[key].split(","):
                    raw_id = raw_id.strip()
                    if raw_id not in leaf_map:
                        raise ValueError(f"Observed KC id {raw_id!r} is not a kc_leaf in the tree JSON.")
                    curids.append(str(leaf_map[raw_id]))
                dres[key].append(",".join(curids))
                continue

            dkeyid2idx.setdefault(key, dict())
            curids = []
            for raw_id in row[key].split(","):
                if raw_id not in dkeyid2idx[key]:
                    dkeyid2idx[key][raw_id] = len(dkeyid2idx[key])
                curids.append(str(dkeyid2idx[key][raw_id]))
            dres[key].append(",".join(curids))

    finaldf = pd.DataFrame(dres)
    return finaldf, dkeyid2idx


def write_tree_artifacts(dname, keyid2idx_tree, parent_index, node_index, tree_report):
    save_id2idx(keyid2idx_tree, os.path.join(dname, "keyid2idx_tree.json"))
    save_id2idx(parent_index, os.path.join(dname, "tree_parent_index.json"))
    save_id2idx(node_index, os.path.join(dname, "tree_node_index.json"))
    save_id2idx(tree_report, os.path.join(dname, "tree_mapping_report.json"))


def write_config(
    dataset_name,
    dkeyid2idx,
    effective_keys,
    configf,
    dpath,
    k=5,
    min_seq_len=3,
    maxlen=200,
    flag=False,
    other_config=None,
    keyid2idx_tree=None,
):
    if other_config is None:
        other_config = {}

    input_type, num_q, num_c = [], 0, 0
    if "questions" in effective_keys:
        input_type.append("questions")
        num_q = len(dkeyid2idx["questions"])
    if "concepts" in effective_keys:
        input_type.append("concepts")
        if "concepts_fmkc" in dkeyid2idx:
            num_c = len(dkeyid2idx.get("concepts_dense", {}))
            if num_c <= 0:
                raise ValueError("kc_fmkc requires non-empty concepts_dense mapping.")
        else:
            num_c = len(dkeyid2idx["concepts"])

    folds = list(range(0, k))
    dconfig = {
        "dpath": dpath,
        "num_q": num_q,
        "num_c": num_c,
        "input_type": input_type,
        "max_concepts": dkeyid2idx["max_concepts"],
        "min_seq_len": min_seq_len,
        "maxlen": maxlen,
        "emb_path": "",
        "train_valid_original_file": "train_valid.csv",
        "train_valid_file": "train_valid_sequences.csv",
        "folds": folds,
        "test_original_file": "test.csv",
        "test_file": "test_sequences.csv",
        "test_window_file": "test_window_sequences.csv",
    }

    if "concepts_fmkc" in dkeyid2idx:
        c_fields = dkeyid2idx["concepts_fmkc"]
        dconfig["kc_fmkc"] = True
        dconfig["num_c_fmkc"] = [len(field_d) for field_d in c_fields]

    if keyid2idx_tree is not None:
        dconfig["kc_tree"] = True
        dconfig["num_c_tree"] = int(keyid2idx_tree["num_c"])
        dconfig["num_c"] = int(keyid2idx_tree["num_c"])
        dconfig["keyid2idx_tree_file"] = "keyid2idx_tree.json"
        dconfig["tree_parent_index_file"] = "tree_parent_index.json"
        dconfig["tree_node_index_file"] = "tree_node_index.json"
        dconfig["tree_mapping_report_file"] = "tree_mapping_report.json"

    dconfig.update(other_config)

    if flag:
        dconfig["test_question_file"] = "test_question_sequences.csv"
        dconfig["test_question_window_file"] = "test_question_window_sequences.csv"

    # load old config
    if not os.path.exists(configf):
        data_config = {dataset_name: dconfig}
    else:
        with open(configf, encoding="utf-8") as fin:
            read_text = fin.read()
            if read_text.strip() == "":
                data_config = {dataset_name: dconfig}
            else:
                data_config = json.loads(read_text)
                if dataset_name in data_config:
                    data_config[dataset_name].update(dconfig)
                else:
                    data_config[dataset_name] = dconfig

    with open(configf, "w", encoding="utf-8") as fout:
        data = json.dumps(data_config, ensure_ascii=False, indent=4)
        fout.write(data)


def calStatistics(df, stares, key):
    allin, allselect = 0, 0
    allqs, allcs = set(), set()
    for i, row in df.iterrows():
        rs = row["responses"].split(",")
        curlen = len(rs) - rs.count("-1")
        allin += curlen
        if "selectmasks" in row:
            ss = row["selectmasks"].split(",")
            slen = ss.count("1")
            allselect += slen
        if "concepts" in row:
            cs = row["concepts"].split(",")
            fc = list()
            for c in cs:
                if "^" in c:
                    fc.extend(c.split("^"))
                else:
                    cc = c.split("_")
                    fc.extend(cc)
            curcs = set(fc) - {"-1"}
            allcs |= curcs
        if "questions" in row:
            qs = row["questions"].split(",")
            curqs = set(qs) - {"-1"}
            allqs |= curqs
    stares.append(",".join([str(s)
                  for s in [key, allin, df.shape[0], allselect]]))
    return allin, allselect, len(allqs), len(allcs), df.shape[0]


def get_max_concepts(df):
    max_concepts = 1
    for i, row in df.iterrows():
        cs = row["concepts"].split(",")
        for c in cs:
            if "^" in c:
                n = len(c.split("^"))
            elif "|" in c:
                n = len(c.split("|"))
            else:
                n = len(c.split("_"))
            max_concepts = max(max_concepts, n)
    return max_concepts


def main(dname, fname, dataset_name, configf, min_seq_len=3, maxlen=200, kfold=5):
    """Split and preprocess a KT dataset.

    Tree mode is enabled automatically for dataset names ending with `_tree`.
    In tree mode, this script expects kc_knowledge_tree_slim.json and writes:
      - keyid2idx.json: ordinary pyKT mapping, but concepts covers all tree nodes
      - keyid2idx_tree.json: tree-specific mapping
      - tree_parent_index.json: compact parent index used by qid_tree
      - tree_node_index.json: metadata for interpretability
      - tree_mapping_report.json: validation report

    Important for qid_tree:
      CSV concepts are remapped from original leaf kc_id to tree_idx.
      Internal parent nodes are included in num_c but do not appear in sequences.
    """
    stares = []

    is_tree_dataset = dataset_name.endswith("_tree")
    other_config = {}
    keyid2idx_tree = None
    parent_index = None
    node_index = None
    tree_report = None

    if is_tree_dataset:
        kc_tree_path = resolve_kc_tree_path(dname, dataset_name)
        if not kc_tree_path:
            raise FileNotFoundError(
                f"Dataset {dataset_name!r} is a tree dataset, but no tree JSON was found under {dname}. "
                "Expected kc_knowledge_tree_slim.json under the dataset folder, metadata folder, "
                "or 2_DBE_KT22_datafiles_100102_csv folder."
            )
        keyid2idx_tree, parent_index, node_index, tree_report = build_tree_artifacts(kc_tree_path)
        other_config["kc_tree_path"] = kc_tree_path
        print("=" * 20)
        print(f"Tree mode enabled for {dataset_name}")
        print(f"kc_tree_path: {kc_tree_path}")
        print(
            f"tree nodes: {tree_report['num_c_tree']}, "
            f"leaf KCs: {tree_report['num_observed_leaf_kcs']}, "
            f"internal nodes: {tree_report['num_internal_nodes']}, "
            f"edges: {tree_report['num_edges']}"
        )

    total_df, effective_keys = read_data(fname)

    # cal max_concepts on the original raw concept string before id mapping
    if "concepts" in effective_keys:
        max_concepts = get_max_concepts(total_df)
    else:
        max_concepts = -1

    oris, _, qs, cs, seqnum = calStatistics(total_df, stares, "original")
    print("=" * 20)
    print(f"original total interactions: {oris}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    total_df, effective_keys = extend_multi_concepts(total_df, effective_keys)

    if dataset_name == "yousician_fmkc":
        total_df, dkeyid2idx = id_mapping_fmkc(total_df)
        effective_keys.add("concepts_dense")
    elif is_tree_dataset:
        if keyid2idx_tree is None:
            raise ValueError("Internal error: tree dataset has no keyid2idx_tree.")
        total_df, dkeyid2idx = id_mapping_tree(total_df, keyid2idx_tree)
    else:
        total_df, dkeyid2idx = id_mapping(total_df)

    dkeyid2idx["max_concepts"] = max_concepts

    extends, _, qs, cs, seqnum = calStatistics(total_df, stares, "extend multi")
    print("=" * 20)
    print(f"after extend multi, total interactions: {extends}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    # In tree mode, keyid2idx.json now contains the full tree-node concept mapping.
    save_id2idx(dkeyid2idx, os.path.join(dname, "keyid2idx.json"))
    if is_tree_dataset:
        write_tree_artifacts(dname, keyid2idx_tree, parent_index, node_index, tree_report)

    effective_keys.add("fold")
    config = []
    for key in ALL_KEYS:
        if key in effective_keys:
            config.append(key)

    # train test split & generate sequences
    train_df, test_df = train_test_split(total_df, 0.2)
    splitdf = KFold_split(train_df, kfold)

    splitdf[config].to_csv(os.path.join(dname, "train_valid.csv"), index=None)
    ins, ss, qs, cs, seqnum = calStatistics(splitdf, stares, "original train+valid")
    print(f"train+valid original interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    split_seqs = generate_sequences(splitdf, effective_keys, min_seq_len, maxlen)
    ins, ss, qs, cs, seqnum = calStatistics(split_seqs, stares, "train+valid sequences")
    print(f"train+valid sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    split_seqs.to_csv(os.path.join(dname, "train_valid_sequences.csv"), index=None)

    # add default fold -1 to test
    test_df = copy.deepcopy(test_df)
    test_df["fold"] = [-1] * test_df.shape[0]
    test_df["cidxs"] = get_inter_qidx(test_df)

    test_seqs = generate_sequences(test_df, list(effective_keys) + ["cidxs"], min_seq_len, maxlen)
    ins, ss, qs, cs, seqnum = calStatistics(test_df, stares, "test original")
    print(f"original test interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    ins, ss, qs, cs, seqnum = calStatistics(test_seqs, stares, "test sequences")
    print(f"test sequences interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
    print("=" * 20)

    test_window_seqs = generate_window_sequences(test_df, list(effective_keys) + ["cidxs"], maxlen)
    flag, test_question_seqs = generate_question_sequences(test_df, effective_keys, False, min_seq_len, maxlen)
    flag, test_question_window_seqs = generate_question_sequences(test_df, effective_keys, True, min_seq_len, maxlen)

    test_df = test_df[config + ["cidxs"]]

    test_df.to_csv(os.path.join(dname, "test.csv"), index=None)
    test_seqs.to_csv(os.path.join(dname, "test_sequences.csv"), index=None)
    test_window_seqs.to_csv(os.path.join(dname, "test_window_sequences.csv"), index=None)

    ins, ss, qs, cs, seqnum = calStatistics(test_window_seqs, stares, "test window")
    print(f"test window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    if flag:
        test_question_seqs.to_csv(os.path.join(dname, "test_question_sequences.csv"), index=None)
        test_question_window_seqs.to_csv(os.path.join(dname, "test_question_window_sequences.csv"), index=None)

        ins, ss, qs, cs, seqnum = calStatistics(test_question_seqs, stares, "test question")
        print(f"test question interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")
        ins, ss, qs, cs, seqnum = calStatistics(test_question_window_seqs, stares, "test question window")
        print(f"test question window interactions num: {ins}, select num: {ss}, qs: {qs}, cs: {cs}, seqnum: {seqnum}")

    write_config(
        dataset_name=dataset_name,
        dkeyid2idx=dkeyid2idx,
        effective_keys=effective_keys,
        configf=configf,
        dpath=dname,
        k=kfold,
        min_seq_len=min_seq_len,
        maxlen=maxlen,
        flag=flag,
        other_config=other_config,
        keyid2idx_tree=keyid2idx_tree,
    )

    print("=" * 20)
    if is_tree_dataset:
        print("Tree artifacts written:")
        print(f"  {os.path.join(dname, 'keyid2idx_tree.json')}")
        print(f"  {os.path.join(dname, 'tree_parent_index.json')}")
        print(f"  {os.path.join(dname, 'tree_node_index.json')}")
        print(f"  {os.path.join(dname, 'tree_mapping_report.json')}")
    print("\n".join(stares))


def process_raw_data(dataset_name, dname2paths):
    """Convert raw source file to pyKT 6-line `data.txt`, return (dname, write_file)."""
    is_tree_dataset = dataset_name.endswith("_tree")
    base_dataset = dataset_name[:-5] if is_tree_dataset else dataset_name

    # Do not silently fallback tree dataset to non-tree source path.
    if is_tree_dataset and dataset_name not in dname2paths:
        raise KeyError(
            f"Missing raw path mapping for tree dataset {dataset_name!r}. "
            "Please add it to dname2paths explicitly."
        )

    src_key = dataset_name if dataset_name in dname2paths else base_dataset
    if src_key not in dname2paths:
        raise KeyError(f"Unknown dataset_name: {dataset_name}")

    read_file = dname2paths[src_key]
    dname = os.path.dirname(dname2paths.get(dataset_name, read_file))
    write_file = os.path.join(dname, "data.txt")

    if base_dataset == "assist2017":
        from .assist2017_preprocess import read_data_from_csv
        read_data_from_csv(read_file, write_file)
    elif base_dataset == "assist2009":
        from .assist2009_preprocess import read_data_from_csv
        read_data_from_csv(read_file, write_file)
    elif base_dataset == "assist2012":
        from .assist2012_preprocess import read_data_from_csv
        read_data_from_csv(read_file, write_file)
    elif base_dataset == "assist2015":
        from .assist2015_preprocess import read_data_from_csv
        read_data_from_csv(read_file, write_file)
    elif base_dataset == "algebra2005":
        from .algebra2005_preprocess import read_data_from_csv
        read_data_from_csv(read_file, write_file)
    elif base_dataset == "bridge2algebra2006":
        from .bridge2algebra2006_preprocess import read_data_from_csv
        read_data_from_csv(read_file, write_file)
    elif base_dataset == "statics2011":
        from .statics2011_preprocess import read_data_from_csv
        read_data_from_csv(read_file, write_file)
    elif base_dataset == "nips_task34":
        from .nips_task34_preprocess import read_data_from_csv
        read_data_from_csv(read_file, os.path.join(dname, "metadata"), "task_3_4", write_file)
    elif base_dataset == "poj":
        from .poj_preprocess import read_data_from_csv
        read_data_from_csv(read_file, write_file)
    elif base_dataset == "slepemapy":
        from .slepemapy_preprocess import read_data_from_csv
        read_data_from_csv(read_file, write_file)
    elif base_dataset == "xes3g5m":
        from .xes3g5m_preprocess import read_data_from_csv
        read_data_from_csv(read_file, write_file)
    elif base_dataset == "junyi2015":
        from .junyi2015_preprocess import read_data_from_csv, load_q2c
        dq2c = load_q2c(os.path.join(dname, "junyi_Exercise_table.csv"))
        read_data_from_csv(read_file, write_file, dq2c)
    elif base_dataset == "peiyou":
        from .aaai2022_competition import read_data_from_csv, load_q2c
        dq2c = load_q2c(os.path.join(dname, "map_questions.json"))
        read_data_from_csv(read_file, write_file, dq2c)
    elif base_dataset in {"ednet", "ednet5w"}:
        from .ednet_preprocess import read_data_from_csv
        read_data_from_csv(read_file, write_file, dataset_name=base_dataset)
    elif base_dataset == "dbe_kt22":
        from .dbe_kt22_preprocess import read_data_from_json
        read_data_from_json(read_file, write_file)
    elif base_dataset == "yousician":
        from .yousician_preprocess import read_data_from_json
        read_data_from_json(read_file, write_file)
    elif base_dataset == "yousician_fmkc":
        from .yousician_preprocess_kc_fmkc import read_data_from_json
        read_data_from_json(read_file, write_file)
    else:
        raise NotImplementedError(f"process_raw_data not implemented for dataset: {dataset_name}")

    return dname, write_file
