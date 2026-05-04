import json
import pandas as pd
from .utils import sta_infos, write_txt

# KC 多域（与 yousician_preprocess 一致）：prev_pitches | prev_strings | pitches | strings
# 本文件将多域原始字符串用 "|" 连接写入 skills 行，供 split 阶段按域独立建词典；
# 序列 CSV 中每步概念为 i0^i1^...（split_datasets.id_mapping_fmkc）。

KEYS = ["user_id", "sequence_id"]


def sanitize_field(x, sep="^"):
    s = str(x)
    s = s.replace(",", sep)
    s = s.replace("\n", "").replace("\r", "").replace("\t", "").replace(" ", "")
    return s


def _events_to_event_rows(events_data_str):
    """每事件输出 question_id（多域拼接）与 KC 多域字符串 raw0|raw1|...。"""
    data = json.loads(events_data_str)
    inner = data.get("data", data)

    duration = inner["duration"]
    pitches = inner["pitches"]
    strings = inner["strings"]
    reject_reason = inner["reject_reason"]

    n = len(pitches)
    assert n == len(duration) == len(strings) == len(reject_reason)

    out = []
    for i in range(n):
        pitch_s = sanitize_field(str(pitches[i]), sep="^")
        string_s = sanitize_field(str(strings[i]), sep="^")
        prev_ps = "inf" if i == 0 else sanitize_field(str(pitches[i - 1]), sep="^")
        prev_ss = "inf" if i == 0 else sanitize_field(str(strings[i - 1]), sep="^")

        qid = f"{prev_ps}|{prev_ss}|{pitch_s}|{string_s}"
        kc_multi = f"{prev_ps}|{prev_ss}|{pitch_s}|{string_s}"
        resp = 1 if reject_reason[i] == 0 else 0
        out.append((qid, kc_multi, resp))

    return out


def read_data_from_json(read_file, write_file):
    stares = []

    with open(read_file, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    records = [r for r in raw_list if r.get("play_mode") == "play"]
    if not records:
        raise ValueError("No records with play_mode=='play'")

    rows = []
    for r in records:
        uid = r["user_id"]
        days = r["days_since_signup"]
        part = r.get("exercise_part_index", 0)
        sess = r.get("session_index", 0)
        try:
            event_list = _events_to_event_rows(r["events_data"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        for qid, kc_multi, resp in event_list:
            rows.append(
                {
                    "user_id": uid,
                    "days_since_signup": days,
                    "exercise_part_index": part,
                    "session_index": sess,
                    "question_id": qid,
                    "sequence_id": kc_multi,
                    "correct": resp,
                }
            )

    df = pd.DataFrame(rows)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"after filter play_mode=play, interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}"
    )

    user_inters = []
    for user, grp in df.groupby("user_id", sort=False):
        tmp = grp.sort_values(by=["days_since_signup", "session_index", "exercise_part_index"])
        seq_problems = tmp["question_id"].astype(str).tolist()
        seq_skills = tmp["sequence_id"].astype(str).tolist()
        seq_ans = tmp["correct"].astype(str).tolist()
        seq_len = len(seq_problems)
        seq_start_time = ["NA"]
        seq_response_cost = ["NA"]
        user_inters.append(
            [[str(user), str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_response_cost]
        )

    write_txt(write_file, user_inters)
    print("\n".join(stares))
    return
