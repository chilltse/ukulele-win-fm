import json
import os
import pandas as pd

from .utils import sta_infos, write_txt

KEYS = ["user_id", "sequence_id", "question_id"]


def _build_question_to_kc_map(base_dir: str):
    rel_path = os.path.join(base_dir, "2_DBE_KT22_datafiles_100102_csv", "Question_KC_Relationships.csv")
    rel_df = pd.read_csv(rel_path)
    q2c = {}
    for qid, grp in rel_df.groupby("question_id"):
        kc_ids = sorted({str(int(k)) for k in grp["knowledgecomponent_id"].dropna().tolist()})
        if kc_ids:
            q2c[str(int(qid))] = "_".join(kc_ids)
    return q2c


def _truncate_tokens(token_str, seq_len):
    toks = [t.strip() for t in str(token_str).split(",")]
    return toks[:seq_len]


def read_data_from_json(read_file, write_file):
    stares = []
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(read_file)))
    q2c = _build_question_to_kc_map(base_dir)

    with open(read_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []
    user2seq = {}
    for rec in records:
        user_id = rec.get("student_id")
        seq_len = int(rec.get("seq_len", 0))
        if user_id is None or seq_len <= 0:
            continue

        qids = _truncate_tokens(rec.get("question_ids", ""), seq_len)
        answers = _truncate_tokens(rec.get("answers", ""), seq_len)
        if len(qids) != len(answers):
            continue

        user_id = str(user_id)
        if user_id not in user2seq:
            user2seq[user_id] = {"questions": [], "concepts": [], "answers": []}

        for qid, ans in zip(qids, answers):
            if qid in {"$", "-1", ""} or ans in {"$", "-1", ""}:
                continue
            qid = str(int(float(qid)))
            ans = str(int(float(ans)))
            cid = q2c.get(qid, qid)
            rows.append(
                {
                    "user_id": user_id,
                    "question_id": qid,
                    "sequence_id": cid,
                    "correct": int(ans),
                }
            )

            user2seq[user_id]["questions"].append(qid)
            user2seq[user_id]["concepts"].append(cid)
            user2seq[user_id]["answers"].append(ans)

    df = pd.DataFrame(rows)
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, "
        f"avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}"
    )

    user_inters = []
    for user_id, seqs in user2seq.items():
        if not seqs["answers"]:
            continue
        user_inters.append(
            [
                [user_id, str(len(seqs["answers"]))],
                seqs["questions"],
                seqs["concepts"],
                seqs["answers"],
                ["NA"],
                ["NA"],
            ]
        )

    write_txt(write_file, user_inters)
    print("\n".join(stares))
    return
