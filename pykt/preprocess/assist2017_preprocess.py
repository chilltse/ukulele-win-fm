import json
import os
import pandas as pd
from .utils import sta_infos, write_txt, format_list2str

keys = ["studentId", "skill", "problemId"]


def _iter_tree_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for child in reversed(node.get("children", []) or []):
            stack.append(child)


def _load_kc_name_to_id(kc_tree_path):
    with open(kc_tree_path, "r", encoding="utf-8") as f:
        tree = json.load(f)

    if isinstance(tree, dict) and "tree" in tree and isinstance(tree["tree"], dict):
        root = tree["tree"]
    elif isinstance(tree, dict) and "children" in tree:
        root = tree
    else:
        raise ValueError(
            f"Unsupported KC tree format: {kc_tree_path}. "
            "Expected a root node dict or a wrapper containing `tree`."
        )

    name_to_kcid = {}
    duplicate_name_conflicts = []
    for node in _iter_tree_nodes(root):
        if node.get("kc_id", None) is None:
            continue
        name = str(node.get("name", "")).strip()
        if not name:
            continue
        kcid = int(node["kc_id"])
        if name in name_to_kcid and name_to_kcid[name] != kcid:
            duplicate_name_conflicts.append((name, name_to_kcid[name], kcid))
        name_to_kcid[name] = kcid

    if duplicate_name_conflicts:
        raise ValueError(
            "KC tree contains duplicated leaf names with different kc_id values. "
            f"Examples: {duplicate_name_conflicts[:10]}"
        )
    return name_to_kcid


def _map_skill_to_kcid_token(skill_token, name_to_kcid):
    # ASSIST2017 can contain multiple concepts joined by "_".
    parts = [p.strip() for p in str(skill_token).split("_")]
    mapped = []
    missing = []
    for p in parts:
        if p == "":
            continue
        if p not in name_to_kcid:
            missing.append(p)
        else:
            mapped.append(str(name_to_kcid[p]))
    return "_".join(mapped), missing


def read_data_from_csv(read_file, write_file, kc_tree_path=None):
    df = pd.read_csv(read_file, encoding='utf-8', low_memory=False)

    stares = []
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, keys, stares)
    print(
        f"original interaction num: {df.shape[0]}, user num: {df['studentId'].nunique()}, question num: {df['problemId'].nunique()}, "
        f"concept num: {df['skill'].nunique()}, avg(ins) per s:{avgins}, avg(c) per q:{avgcq}, na:{na}")

    df["index"] = range(len(df))

    df = df.dropna(subset=["studentId", "problemId", "correct", "skill", "startTime"])
    df = df[df['correct'].isin([0, 1])]  
    df.loc[:, 'timeTaken'] = df['timeTaken'].apply(lambda x: round(x * 1000))

    if kc_tree_path:
        if not os.path.exists(kc_tree_path):
            raise FileNotFoundError(f"assist2017_tree requires kc_tree_path, missing: {kc_tree_path}")
        name_to_kcid = _load_kc_name_to_id(kc_tree_path)
        mapped_skills = []
        missing_skills = set()
        for raw_skill in df["skill"].tolist():
            mapped, missing = _map_skill_to_kcid_token(raw_skill, name_to_kcid)
            if missing:
                missing_skills.update(missing)
            mapped_skills.append(mapped)
        if missing_skills:
            raise ValueError(
                "assist2017_tree skill->kc_id mapping failed. "
                f"Missing skill names in tree JSON (examples): {sorted(missing_skills)[:20]}"
            )
        df.loc[:, "skill"] = mapped_skills

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, keys, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df2 = df[["index", "studentId", "problemId", "skill", "correct", "timeTaken", "startTime"]]
    ui_df = df2.groupby('studentId', sort=False)

    user_inter = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]  
        tmp_inter.loc[:, 'startTime'] = tmp_inter.loc[:, 'startTime'].apply(lambda t: int(t) * 1000)
        tmp_inter = tmp_inter.sort_values(by=['startTime', 'index'])

        tmp_inter['startTime'] = tmp_inter['startTime']

        seq_len = len(tmp_inter)
        seq_problems = tmp_inter['problemId'].tolist()
        seq_skills = tmp_inter['skill'].tolist()
        seq_ans = tmp_inter['correct'].tolist()
        seq_submit_time = tmp_inter['startTime'].tolist()
        seq_response_cost = tmp_inter['timeTaken'].tolist()

        assert seq_len == len(seq_problems) == len(seq_skills) == len(seq_ans) == len(seq_submit_time) == len(seq_response_cost)

        user_inter.append(
            [[str(user), str(seq_len)], format_list2str(seq_problems), seq_skills, format_list2str(seq_ans), format_list2str(seq_submit_time), format_list2str(seq_response_cost)])

    write_txt(write_file, user_inter)


