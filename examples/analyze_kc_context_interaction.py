import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Basic parsing utilities
# -----------------------------

def normalize_token(token: object) -> str:
    return str(token).strip()


def numeric_sort_key(x: object):
    s = str(x)
    return (0, int(s)) if s.isdigit() else (1, s)


def split_seq(raw: object) -> List[str]:
    return [x.strip() for x in str(raw).split(',')]


def parse_folds(folds_text: str) -> Set[int]:
    parts = [p.strip() for p in folds_text.split(',') if p.strip()]
    if not parts:
        raise ValueError("`--train-folds` is empty. Example: 1,2,3,4")
    return {int(p) for p in parts}


def is_pad_token(token: object, pad_val: int = -1) -> bool:
    token = normalize_token(token)
    if token == '':
        return True

    # Some pyKT sequences may use ^ for repeated concepts in a token.
    parts = [p.strip() for p in token.split('^')]
    if not parts:
        return True

    try:
        values = [int(p) for p in parts]
    except ValueError:
        return False

    return all(v == pad_val for v in values)


def split_concept_token_to_discrete_kcs(token: str, pad_val: int = -1) -> List[str]:
    """
    Split packed KC tokens into discrete original KC ids.

    Examples:
        "304_484_399" -> ["304", "484", "399"]
        "304" -> ["304"]
        "-1" -> []
    """
    token = normalize_token(token)
    if token == '' or is_pad_token(token, pad_val=pad_val):
        return []

    parts = [p.strip() for p in token.split('_') if p.strip()]
    out, seen = [], set()

    for p in parts:
        if p == '' or p == str(pad_val):
            continue
        if p not in seen:
            out.append(p)
            seen.add(p)

    return out


def load_json(path: Path) -> Dict[str, object]:
    with path.open('r', encoding='utf-8') as fin:
        return json.load(fin)


def build_idx2key_maps(keyid2idx_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if keyid2idx_path is None:
        return {}
    if not keyid2idx_path.exists():
        raise FileNotFoundError(f'keyid2idx file not found: {keyid2idx_path}')

    keyid2idx = load_json(keyid2idx_path)
    idx2key_maps: Dict[str, Dict[str, str]] = {}

    for field_name, mapping in keyid2idx.items():
        if not isinstance(mapping, dict):
            continue
        idx2key_maps[field_name] = {}
        for original_id, internal_id in mapping.items():
            internal_id_str = str(internal_id)
            original_id_str = str(original_id)
            if internal_id_str in idx2key_maps[field_name]:
                raise ValueError(
                    f'Duplicate internal id in keyid2idx[{field_name!r}]: {internal_id_str}. '
                    f'This makes reverse mapping ambiguous.'
                )
            idx2key_maps[field_name][internal_id_str] = original_id_str

    return idx2key_maps


def restore_id_token(token: object, idx2key_map: Optional[Dict[str, str]], pad_val: int = -1) -> str:
    token = normalize_token(token)
    if token == '' or token == str(pad_val):
        return token
    if not idx2key_map:
        return token
    if token in idx2key_map:
        return idx2key_map[token]

    # Rare fallback: token is like "12_13", where each part is an internal id.
    if '_' in token:
        parts = [p.strip() for p in token.split('_') if p.strip()]
        return '_'.join(idx2key_map.get(p, p) for p in parts)

    return token


def choose_train_folds(df: pd.DataFrame, valid_fold: int, train_folds_text: str) -> Set[int]:
    all_folds = {int(x) for x in sorted(df['fold'].dropna().unique())}

    if train_folds_text.strip():
        train_folds = parse_folds(train_folds_text)
        unknown = sorted(train_folds - all_folds)
        if unknown:
            raise ValueError(
                f'Unknown folds in --train-folds: {unknown}. Available folds: {sorted(all_folds)}'
            )
        return train_folds

    if valid_fold not in all_folds:
        raise ValueError(f'--valid-fold={valid_fold} not in data folds {sorted(all_folds)}')

    train_folds = all_folds - {valid_fold}
    if not train_folds:
        raise ValueError('No train folds left after excluding valid fold.')
    return train_folds


def load_data_config(data_config_path: Path) -> Dict[str, object]:
    if not data_config_path.exists():
        raise FileNotFoundError(f'data_config.json not found: {data_config_path}')
    with data_config_path.open('r', encoding='utf-8') as fin:
        return json.load(fin)


def resolve_input_path(dataset_name: str, data_config_path: Path, input_csv_arg: str) -> Path:
    if input_csv_arg.strip():
        return Path(input_csv_arg)

    config = load_data_config(data_config_path)
    if dataset_name not in config:
        raise ValueError(f'dataset_name={dataset_name!r} not found in {data_config_path}')

    dataset_cfg = config[dataset_name]
    dpath = (data_config_path.parent / str(dataset_cfg['dpath'])).resolve()
    filename = dataset_cfg.get('train_valid_original_file_quelevel')
    if not filename:
        raise ValueError(f'Missing train_valid_original_file_quelevel in data_config for {dataset_name}')
    return (dpath / str(filename)).resolve()


def context_for_target(kc_set: Set[str], target: str, max_partner_names: int = 3) -> Optional[str]:
    if target not in kc_set:
        return None
    others = sorted(kc_set - {target}, key=numeric_sort_key)
    if not others:
        return f'{target}_core'
    if len(others) == 1:
        return f'{target}_with_{others[0]}_subskill'
    shown = others[:max_partner_names]
    return f'{target}_with_multi_' + '_'.join(shown)


def make_pair(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((str(a), str(b)), key=numeric_sort_key))  # type: ignore


@dataclass
class Stat:
    n: int = 0
    correct: int = 0
    qids: Set[str] = field(default_factory=set)
    users: Set[str] = field(default_factory=set)

    def add(self, response: int, qid: Optional[str] = None, uid: Optional[str] = None) -> None:
        self.n += 1
        self.correct += int(response)
        if qid is not None:
            self.qids.add(str(qid))
        if uid is not None and str(uid) != '':
            self.users.add(str(uid))

    @property
    def acc(self) -> float:
        return self.correct / self.n if self.n else float('nan')

    @property
    def question_count(self) -> int:
        return len(self.qids)

    @property
    def student_count(self) -> int:
        return len(self.users)


# -----------------------------
# One-pass aggregation
# -----------------------------

def aggregate_fast(
    df_train: pd.DataFrame,
    idx2key_maps: Dict[str, Dict[str, str]],
    target_kcs: Optional[Set[str]],
    pad_val: int = -1,
    max_context_partner_names: int = 3,
    progress_every: int = 5000,
) -> Dict[str, object]:
    """
    This function avoids the expensive previous design:
      - no expanded full events DataFrame by default
      - no per-pair full rescan
      - all counts are collected in one pass through train sequences
    """
    q_idx2key = idx2key_maps.get('questions')
    c_idx2key = idx2key_maps.get('concepts')
    u_idx2key = idx2key_maps.get('uid')

    kc_total: Dict[str, Stat] = defaultdict(Stat)
    kc_only: Dict[str, Stat] = defaultdict(Stat)
    pair_ab: Dict[Tuple[str, str], Stat] = defaultdict(Stat)
    context_stats: Dict[Tuple[str, str], Stat] = defaultdict(Stat)
    question_context_stats: Dict[Tuple[str, str, str], Stat] = defaultdict(Stat)

    # qid -> union of all discrete KCs ever attached to the question.
    q_to_kcs: Dict[str, Set[str]] = defaultdict(set)

    total_events = 0
    invalid_responses = 0
    misaligned_sequences = 0
    max_kcs_in_event = 0
    total_kc_occurrences = 0

    target_all = target_kcs is None or len(target_kcs) == 0

    for ridx, row in df_train.iterrows():
        if progress_every > 0 and ridx > 0 and ridx % progress_every == 0:
            print(f'[progress] processed train rows: {ridx}/{len(df_train)}')

        uid = restore_id_token(row.get('uid', ''), idx2key_map=u_idx2key, pad_val=pad_val)
        q_seq = split_seq(row['questions'])
        c_seq = split_seq(row['concepts'])
        r_seq = split_seq(row['responses'])

        seq_len = min(len(q_seq), len(c_seq), len(r_seq))
        if len(q_seq) != len(c_seq) or len(c_seq) != len(r_seq):
            misaligned_sequences += 1

        for pos in range(seq_len):
            q_raw = normalize_token(q_seq[pos])
            c_raw = normalize_token(c_seq[pos])
            r_raw = normalize_token(r_seq[pos])

            if q_raw == '' or q_raw == str(pad_val):
                continue
            if c_raw == '' or is_pad_token(c_raw, pad_val=pad_val):
                continue
            if r_raw == '' or r_raw == str(pad_val):
                continue

            try:
                response = int(r_raw)
            except ValueError:
                invalid_responses += 1
                continue
            if response not in (0, 1):
                invalid_responses += 1
                continue

            qid = restore_id_token(q_raw, idx2key_map=q_idx2key, pad_val=pad_val)
            concept_token = restore_id_token(c_raw, idx2key_map=c_idx2key, pad_val=pad_val)
            kcs = split_concept_token_to_discrete_kcs(concept_token, pad_val=pad_val)
            if not kcs:
                continue

            # Deduplicate and sort for stable pair/context definitions.
            kc_set = set(kcs)
            kcs_sorted = sorted(kc_set, key=numeric_sort_key)

            total_events += 1
            total_kc_occurrences += len(kcs_sorted)
            max_kcs_in_event = max(max_kcs_in_event, len(kcs_sorted))
            q_to_kcs[str(qid)].update(kcs_sorted)

            # KC-level and context-level aggregation.
            for kc in kcs_sorted:
                kc_total[kc].add(response, qid=qid, uid=uid)
                if len(kcs_sorted) == 1:
                    kc_only[kc].add(response, qid=qid, uid=uid)

                if target_all or kc in target_kcs:
                    ctx = context_for_target(kc_set, kc, max_partner_names=max_context_partner_names)
                    if ctx is not None:
                        context_stats[(kc, ctx)].add(response, qid=qid, uid=uid)
                        question_context_stats[(kc, str(qid), ctx)].add(response, qid=qid, uid=uid)

            # Pair co-performance aggregation. This counts practices where pair co-occurs.
            if len(kcs_sorted) >= 2:
                for a, b in combinations(kcs_sorted, 2):
                    pair_ab[(a, b)].add(response, qid=qid, uid=uid)

    return {
        'kc_total': kc_total,
        'kc_only': kc_only,
        'pair_ab': pair_ab,
        'context_stats': context_stats,
        'question_context_stats': question_context_stats,
        'q_to_kcs': q_to_kcs,
        'meta': {
            'total_events': total_events,
            'invalid_responses': invalid_responses,
            'misaligned_sequences': misaligned_sequences,
            'max_kcs_in_event': max_kcs_in_event,
            'total_kc_occurrences': total_kc_occurrences,
        }
    }


# -----------------------------
# Reports
# -----------------------------

def build_pair_association_from_questions(
    q_to_kcs: Dict[str, Set[str]],
    min_pair_questions: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, Set[str]], Dict[Tuple[str, str], Set[str]]]:
    kc_to_q: Dict[str, Set[str]] = defaultdict(set)
    pair_to_q: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for qid, kc_set in q_to_kcs.items():
        kcs_sorted = sorted(kc_set, key=numeric_sort_key)
        for kc in kcs_sorted:
            kc_to_q[kc].add(qid)
        if len(kcs_sorted) >= 2:
            for a, b in combinations(kcs_sorted, 2):
                pair_to_q[(a, b)].add(qid)

    total_q = len(q_to_kcs)
    rows = []

    for (a, b), shared_qids in pair_to_q.items():
        co_q = len(shared_qids)
        if co_q < min_pair_questions:
            continue

        qa = kc_to_q[a]
        qb = kc_to_q[b]
        union = qa | qb
        p_a = len(qa) / total_q if total_q else 0.0
        p_b = len(qb) / total_q if total_q else 0.0
        p_ab = co_q / total_q if total_q else 0.0
        lift = p_ab / (p_a * p_b) if p_a > 0 and p_b > 0 else 0.0

        rows.append({
            'kc_a': a,
            'kc_b': b,
            'co_question_count': co_q,
            'a_question_count': len(qa),
            'b_question_count': len(qb),
            'jaccard': co_q / len(union) if union else 0.0,
            'P_b_given_a': co_q / len(qa) if qa else 0.0,
            'P_a_given_b': co_q / len(qb) if qb else 0.0,
            'lift': lift,
            'pmi': math.log2(lift) if lift > 0 else 0.0,
            'shared_question_ids': json.dumps(sorted(shared_qids, key=numeric_sort_key), ensure_ascii=False),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ['co_question_count', 'lift', 'jaccard'],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    return df, kc_to_q, pair_to_q


def build_pair_interaction_performance(
    assoc_df: pd.DataFrame,
    kc_total: Dict[str, Stat],
    pair_ab: Dict[Tuple[str, str], Stat],
    total_events: int,
    min_pair_practices: int = 50,
    min_pair_share: float = 0.001,
    gap_threshold: float = 0.05,
) -> pd.DataFrame:
    rows = []

    if assoc_df.empty:
        return pd.DataFrame()

    for _, r in assoc_df.iterrows():
        a, b = str(r['kc_a']), str(r['kc_b'])
        pair = make_pair(a, b)

        ab = pair_ab.get(pair, Stat())
        ta = kc_total.get(a, Stat())
        tb = kc_total.get(b, Stat())

        # a_only with respect to b = all practices containing a minus practices containing both a and b.
        a_only_n = ta.n - ab.n
        b_only_n = tb.n - ab.n
        a_only_correct = ta.correct - ab.correct
        b_only_correct = tb.correct - ab.correct

        acc_a_only = a_only_correct / a_only_n if a_only_n > 0 else float('nan')
        acc_b_only = b_only_correct / b_only_n if b_only_n > 0 else float('nan')
        acc_ab = ab.acc

        if math.isnan(acc_a_only) or math.isnan(acc_b_only) or math.isnan(acc_ab):
            expected_simple = float('nan')
            expected_weighted = float('nan')
            gap_simple = float('nan')
            gap_weighted = float('nan')
            gap_vs_easier = float('nan')
            gap_vs_harder = float('nan')
        else:
            expected_simple = (acc_a_only + acc_b_only) / 2
            denom = a_only_n + b_only_n
            expected_weighted = (
                (acc_a_only * a_only_n + acc_b_only * b_only_n) / denom
                if denom > 0 else expected_simple
            )
            gap_simple = acc_ab - expected_simple
            gap_weighted = acc_ab - expected_weighted
            gap_vs_easier = acc_ab - max(acc_a_only, acc_b_only)
            gap_vs_harder = acc_ab - min(acc_a_only, acc_b_only)

        pair_share = ab.n / total_events if total_events else 0.0

        if float(r['P_b_given_a']) >= 0.7 and float(r['P_a_given_b']) >= 0.7:
            structural_note = 'mutually_bound'
        elif float(r['P_b_given_a']) >= 0.7:
            structural_note = f'{a}_mostly_with_{b}'
        elif float(r['P_a_given_b']) >= 0.7:
            structural_note = f'{b}_mostly_with_{a}'
        elif float(r['lift']) >= 2:
            structural_note = 'high_lift_association'
        else:
            structural_note = 'weak_or_moderate_association'

        if (
            ab.n >= min_pair_practices
            and pair_share >= min_pair_share
            and not math.isnan(gap_weighted)
            and abs(gap_weighted) >= gap_threshold
            and int(r['co_question_count']) >= 2
        ):
            decision = 'composite_or_interaction_candidate'
        elif ab.n >= min_pair_practices and pair_share >= min_pair_share:
            decision = 'structural_association_but_weak_performance_gap'
        else:
            decision = 'insufficient_pair_samples'

        rows.append({
            'kc_a': a,
            'kc_b': b,
            'co_question_count': int(r['co_question_count']),
            'lift': float(r['lift']),
            'jaccard': float(r['jaccard']),
            'P_b_given_a': float(r['P_b_given_a']),
            'P_a_given_b': float(r['P_a_given_b']),
            'a_total_practice_count': ta.n,
            'b_total_practice_count': tb.n,
            'ab_practice_count': ab.n,
            'a_only_practice_count': a_only_n,
            'b_only_practice_count': b_only_n,
            'a_total_question_count': ta.question_count,
            'b_total_question_count': tb.question_count,
            'ab_question_count': ab.question_count,
            'acc_a_only': acc_a_only,
            'acc_b_only': acc_b_only,
            'acc_ab': acc_ab,
            'expected_ab_acc_simple_avg': expected_simple,
            'expected_ab_acc_weighted_avg': expected_weighted,
            'interaction_gap_simple': gap_simple,
            'interaction_gap_weighted': gap_weighted,
            'gap_vs_easier_single': gap_vs_easier,
            'gap_vs_harder_single': gap_vs_harder,
            'ab_practice_share': pair_share,
            'structural_note': structural_note,
            'decision': decision,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Put likely useful candidates first.
        decision_rank = {
            'composite_or_interaction_candidate': 0,
            'structural_association_but_weak_performance_gap': 1,
            'insufficient_pair_samples': 2,
        }
        df['_decision_rank'] = df['decision'].map(decision_rank).fillna(9)
        df['_abs_gap'] = df['interaction_gap_weighted'].abs()
        df = df.sort_values(
            ['_decision_rank', 'ab_practice_count', '_abs_gap', 'lift'],
            ascending=[True, False, False, False]
        ).drop(columns=['_decision_rank', '_abs_gap']).reset_index(drop=True)

    return df


def build_context_performance(context_stats: Dict[Tuple[str, str], Stat], kc_total: Dict[str, Stat], min_context_practices: int = 20) -> pd.DataFrame:
    rows = []
    for (kc, ctx), st in context_stats.items():
        if st.n < min_context_practices:
            continue
        overall_acc = kc_total[kc].acc if kc in kc_total else float('nan')
        rows.append({
            'target_kc': kc,
            'context': ctx,
            'practice_count': st.n,
            'question_count': st.question_count,
            'student_count': st.student_count,
            'mean_acc': st.acc,
            'target_overall_acc': overall_acc,
            'context_acc_gap_from_target_overall': st.acc - overall_acc if not math.isnan(overall_acc) else float('nan'),
            'question_ids': json.dumps(sorted(st.qids, key=numeric_sort_key), ensure_ascii=False),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['target_kc', 'practice_count'], ascending=[True, False]).reset_index(drop=True)
    return df


def build_question_and_internal_difficulty_reports(
    question_context_stats: Dict[Tuple[str, str, str], Stat],
    min_question_practices: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    q_rows = []
    for (kc, qid, ctx), st in question_context_stats.items():
        if st.n < min_question_practices:
            continue
        q_rows.append({
            'kc': kc,
            'question_id': qid,
            'context': ctx,
            'practice_count': st.n,
            'student_count': st.student_count,
            'mean_acc': st.acc,
        })

    qdf = pd.DataFrame(q_rows)
    if qdf.empty:
        return pd.DataFrame(), pd.DataFrame()

    qdf = qdf.sort_values(['kc', 'context', 'mean_acc'], ascending=[True, True, True]).reset_index(drop=True)

    summary_rows = []
    for kc, g in qdf.groupby('kc'):
        if len(g) < 2:
            continue
        weights = g['practice_count'].astype(float)
        wsum = weights.sum()
        mean = float((g['mean_acc'] * weights).sum() / wsum)
        total_var = float(((g['mean_acc'] - mean) ** 2 * weights).sum() / wsum)

        context_info = []
        between_var = 0.0
        for ctx, cg in g.groupby('context'):
            cw = float(cg['practice_count'].sum())
            cm = float((cg['mean_acc'] * cg['practice_count']).sum() / cg['practice_count'].sum())
            between_var += cw * (cm - mean) ** 2
            context_info.append({
                'context': ctx,
                'context_practice_count': int(cw),
                'context_question_count': int(len(cg)),
                'context_mean_acc': cm,
            })
        between_var = float(between_var / wsum) if wsum else 0.0
        explained_ratio = between_var / total_var if total_var > 1e-12 else 0.0

        hard = g[g['mean_acc'] <= g['mean_acc'].quantile(0.25)]
        easy = g[g['mean_acc'] >= g['mean_acc'].quantile(0.75)]
        hard_contexts = Counter(hard['context']).most_common(5)
        easy_contexts = Counter(easy['context']).most_common(5)

        if explained_ratio >= 0.4 and len(context_info) >= 2:
            interpretation = 'context_explains_large_share_of_question_difficulty_subskill_candidate'
        elif total_var >= 0.02:
            interpretation = 'large_question_difficulty_variation_not_fully_context_explained'
        else:
            interpretation = 'low_internal_question_variation'

        summary_rows.append({
            'kc': kc,
            'question_count_used': int(len(g)),
            'practice_count_used': int(wsum),
            'weighted_question_mean_acc': mean,
            'question_mean_acc_std_weighted': math.sqrt(total_var),
            'question_mean_acc_min': float(g['mean_acc'].min()),
            'question_mean_acc_max': float(g['mean_acc'].max()),
            'question_mean_acc_range': float(g['mean_acc'].max() - g['mean_acc'].min()),
            'context_count': int(len(context_info)),
            'between_context_variance': between_var,
            'total_question_variance': total_var,
            'context_explained_difficulty_ratio': explained_ratio,
            'hard_question_top_contexts': json.dumps(hard_contexts, ensure_ascii=False),
            'easy_question_top_contexts': json.dumps(easy_contexts, ensure_ascii=False),
            'context_mean_details': json.dumps(context_info, ensure_ascii=False),
            'difficulty_interpretation': interpretation,
        })

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ['context_explained_difficulty_ratio', 'question_mean_acc_std_weighted', 'practice_count_used'],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    return qdf, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description='FAST KC context interaction analysis without per-pair rescanning.'
    )
    parser.add_argument('--dataset_name', type=str, default='yousician_fmkc')
    parser.add_argument('--input_csv', type=str, default='')
    parser.add_argument('--keyid2idx', type=str, default='')
    parser.add_argument('--data_config', type=str, default=str(PROJECT_ROOT / 'configs' / 'data_config.json'))
    parser.add_argument('--valid-fold', type=int, default=0)
    parser.add_argument('--train-folds', type=str, default='')
    parser.add_argument('--pad-val', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--target-kcs', type=str, default='', help='Comma-separated target KCs. Empty = all KCs.')
    parser.add_argument('--min-pair-questions', type=int, default=2)
    parser.add_argument('--min-pair-practices', type=int, default=50)
    parser.add_argument('--min-pair-share', type=float, default=0.001)
    parser.add_argument('--interaction-gap-threshold', type=float, default=0.05)
    parser.add_argument('--min-context-practices', type=int, default=20)
    parser.add_argument('--min-question-practices', type=int, default=5)
    parser.add_argument('--max-context-partner-names', type=int, default=3)
    parser.add_argument('--progress-every', type=int, default=5000)
    args = parser.parse_args()

    input_csv = resolve_input_path(args.dataset_name, Path(args.data_config), args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f'Input file not found: {input_csv}')

    if args.keyid2idx.strip():
        keyid2idx_path = Path(args.keyid2idx)
    else:
        candidates = [input_csv.parent / 'keyid2idx.json', input_csv.parent.parent / 'keyid2idx.json']
        keyid2idx_path = next((p for p in candidates if p.exists()), None)

    idx2key_maps = build_idx2key_maps(keyid2idx_path)

    # Keep dtype=str to avoid accidental float conversion of ids.
    df = pd.read_csv(input_csv, dtype=str)
    required = {'fold', 'questions', 'concepts', 'responses'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns in input file: {sorted(missing)}')

    # fold needs numeric comparison.
    df['fold'] = df['fold'].astype(int)
    train_folds = choose_train_folds(df, args.valid_fold, args.train_folds)
    df_train = df[df['fold'].isin(train_folds)].copy()

    if args.output_dir.strip():
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_csv.parent / 'kc_context_diagnostics_fast'
    output_dir.mkdir(parents=True, exist_ok=True)

    target_kcs = None
    if args.target_kcs.strip():
        target_kcs = {x.strip() for x in args.target_kcs.split(',') if x.strip()}

    print(f'input_csv: {input_csv.resolve()}')
    print(f'keyid2idx: {keyid2idx_path.resolve() if keyid2idx_path else None}')
    print(f'train_folds_used: {sorted(train_folds)}')
    print(f'train_rows_used: {len(df_train)}')
    print(f'target_kcs: {sorted(target_kcs, key=numeric_sort_key) if target_kcs else "ALL"}')
    print(f'output_dir: {output_dir.resolve()}')

    agg = aggregate_fast(
        df_train=df_train,
        idx2key_maps=idx2key_maps,
        target_kcs=target_kcs,
        pad_val=args.pad_val,
        max_context_partner_names=args.max_context_partner_names,
        progress_every=args.progress_every,
    )

    kc_total: Dict[str, Stat] = agg['kc_total']  # type: ignore
    kc_only: Dict[str, Stat] = agg['kc_only']  # type: ignore
    pair_ab: Dict[Tuple[str, str], Stat] = agg['pair_ab']  # type: ignore
    context_stats: Dict[Tuple[str, str], Stat] = agg['context_stats']  # type: ignore
    question_context_stats: Dict[Tuple[str, str, str], Stat] = agg['question_context_stats']  # type: ignore
    q_to_kcs: Dict[str, Set[str]] = agg['q_to_kcs']  # type: ignore
    meta: Dict[str, int] = agg['meta']  # type: ignore

    assoc_df, _, _ = build_pair_association_from_questions(q_to_kcs, min_pair_questions=args.min_pair_questions)
    assoc_df.to_csv(output_dir / 'kc_pair_association.csv', index=False, encoding='utf-8-sig')

    pair_perf_df = build_pair_interaction_performance(
        assoc_df=assoc_df,
        kc_total=kc_total,
        pair_ab=pair_ab,
        total_events=int(meta['total_events']),
        min_pair_practices=args.min_pair_practices,
        min_pair_share=args.min_pair_share,
        gap_threshold=args.interaction_gap_threshold,
    )
    pair_perf_df.to_csv(output_dir / 'kc_pair_interaction_performance.csv', index=False, encoding='utf-8-sig')

    ctx_df = build_context_performance(
        context_stats=context_stats,
        kc_total=kc_total,
        min_context_practices=args.min_context_practices,
    )
    ctx_df.to_csv(output_dir / 'kc_target_context_performance.csv', index=False, encoding='utf-8-sig')

    qdf, qsum = build_question_and_internal_difficulty_reports(
        question_context_stats=question_context_stats,
        min_question_practices=args.min_question_practices,
    )
    qdf.to_csv(output_dir / 'kc_question_mean_acc_by_context.csv', index=False, encoding='utf-8-sig')
    qsum.to_csv(output_dir / 'kc_internal_question_difficulty_summary.csv', index=False, encoding='utf-8-sig')

    # KC total summary is useful for quick inspection.
    kc_rows = []
    for kc, st in kc_total.items():
        only = kc_only.get(kc, Stat())
        kc_rows.append({
            'kc': kc,
            'practice_count': st.n,
            'mean_acc': st.acc,
            'question_count': st.question_count,
            'student_count': st.student_count,
            'only_practice_count': only.n,
            'only_mean_acc': only.acc,
            'multi_practice_count': st.n - only.n,
            'multi_practice_ratio': (st.n - only.n) / st.n if st.n else 0.0,
        })
    kc_summary = pd.DataFrame(kc_rows)
    if not kc_summary.empty:
        kc_summary = kc_summary.sort_values(['practice_count'], ascending=False).reset_index(drop=True)
    kc_summary.to_csv(output_dir / 'kc_total_summary.csv', index=False, encoding='utf-8-sig')

    print('\n=== Finished ===')
    for k, v in meta.items():
        print(f'{k}: {v}')
    print(f'unique_questions: {len(q_to_kcs)}')
    print(f'unique_kcs: {len(kc_total)}')
    print(f'pair_association_rows: {len(assoc_df)}')
    print(f'pair_interaction_rows: {len(pair_perf_df)}')
    print(f'context_rows: {len(ctx_df)}')
    print(f'question_context_rows: {len(qdf)}')
    print(f'output_dir: {output_dir.resolve()}')

    if not pair_perf_df.empty:
        cols = [
            'kc_a', 'kc_b', 'co_question_count', 'lift', 'P_b_given_a', 'P_a_given_b',
            'a_only_practice_count', 'b_only_practice_count', 'ab_practice_count',
            'acc_a_only', 'acc_b_only', 'acc_ab', 'expected_ab_acc_weighted_avg',
            'interaction_gap_weighted', 'ab_practice_share', 'decision'
        ]
        print('\nTop pair interaction rows:')
        print(pair_perf_df[cols].head(20).to_string(index=False))

    if not qsum.empty:
        cols = [
            'kc', 'question_count_used', 'practice_count_used',
            'question_mean_acc_std_weighted', 'context_count',
            'context_explained_difficulty_ratio', 'difficulty_interpretation'
        ]
        print('\nTop internal question difficulty rows:')
        print(qsum[cols].head(20).to_string(index=False))

if __name__ == '__main__':
# python analyze_kc_context_interaction.py --dataset_name xes3g5m_tree_manual_split --valid-fold 0 --target-kcs 18,166,177,73,9,64 --min-pair-questions 2 --min-pair-practices 50 --min-context-practices 20 --min-question-practices 5 --progress-every 2000
    main()
