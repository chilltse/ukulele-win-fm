import argparse
import json
import os
import ast

import numpy as np
import pandas as pd
from gensim.models import FastText


def parse_seq_cell(x):
    """
    Parse sequence cell from pyKT-style csv.

    Supports:
        "1,2,3"
        "[1, 2, 3]"
        "1 2 3"
    """
    if pd.isna(x):
        return []

    x = str(x).strip()

    if x == "":
        return []

    if x.startswith("[") and x.endswith("]"):
        try:
            arr = ast.literal_eval(x)
            return [int(v) for v in arr]
        except Exception:
            pass

    if "," in x:
        return [int(v) for v in x.split(",") if str(v).strip() != ""]

    return [int(v) for v in x.split() if str(v).strip() != ""]


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Cannot find any column from {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def q_char(qid):
    """
    Map question id to one Unicode character.

    The original paper describes a one-to-one mapping from question id to character.
    We use Unicode code points starting from 0x10000.

    This can cover far more than normal private-use BMP range.
    """
    code = 0x10000 + int(qid)

    if code > 0x10FFFD:
        raise ValueError(
            f"Question id too large for Unicode character mapping: {qid}"
        )

    return chr(code)


def r_char(r):
    """
    Map response to one Unicode character.
    """
    r = int(r)

    if r == 0:
        return chr(0xE000)
    elif r == 1:
        return chr(0xE001)
    else:
        raise ValueError(f"Response must be 0 or 1, got {r}")


def interaction_token(qid, r):
    """
    A two-character token:
        question-character + response-character

    This matches the paper's spirit:
        one interaction = question id + graded response.
    """
    return q_char(qid) + r_char(r)


def build_sentences_from_csv(csv_path, q_col=None, r_col=None):
    df = pd.read_csv(csv_path)

    if q_col is None:
        q_col = find_column(
            df,
            [
                "questions",
                "qseqs",
                "qseq",
                "q_seq",
                "cq",
                "q",
            ],
        )

    if r_col is None:
        r_col = find_column(
            df,
            [
                "responses",
                "rseqs",
                "rseq",
                "r_seq",
                "cr",
                "r",
            ],
        )

    sentences = []
    max_q = -1

    for _, row in df.iterrows():
        qs = parse_seq_cell(row[q_col])
        rs = parse_seq_cell(row[r_col])

        if len(qs) != len(rs):
            raise ValueError(
                f"Sequence length mismatch: len(qs)={len(qs)}, len(rs)={len(rs)}"
            )

        sent = []

        for q, r in zip(qs, rs):
            q = int(q)
            r = int(r)

            if q < 0:
                continue

            if r not in [0, 1]:
                continue

            sent.append(interaction_token(q, r))
            max_q = max(max_q, q)

        if len(sent) > 0:
            sentences.append(sent)

    return sentences, max_q


def train_fasttext(sentences, emb_size, window, min_count, epochs, workers, seed):
    """
    Train fastText-style embeddings.

    min_n=1 and max_n=2 are important because our token has two characters:
        [question_char, response_char]

    This allows the model to use:
        question component
        response component
        question-response component
    """
    model = FastText(
        vector_size=emb_size,
        window=window,
        min_count=min_count,
        sg=1,
        min_n=1,
        max_n=2,
        workers=workers,
        seed=seed,
    )

    model.build_vocab(corpus_iterable=sentences)
    model.train(
        corpus_iterable=sentences,
        total_examples=len(sentences),
        epochs=epochs,
    )

    return model


def export_interaction_embedding(model, num_q, emb_size, output_path):
    """
    Export matrix with shape:
        [2 * num_q, emb_size]

    Row convention:
        row = q + num_q * r
    """
    emb = np.zeros((num_q * 2, emb_size), dtype=np.float32)

    for q in range(num_q):
        for r in [0, 1]:
            token = interaction_token(q, r)
            row = q + num_q * r
            emb[row] = model.wv[token]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, emb)

    print(f"Saved embedding to: {output_path}")
    print(f"Embedding shape: {emb.shape}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--num_q", type=int, default=None)
    parser.add_argument("--emb_size", type=int, default=100)

    parser.add_argument("--q_col", type=str, default=None)
    parser.add_argument("--r_col", type=str, default=None)

    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)

    args = parser.parse_args()

    sentences, max_q = build_sentences_from_csv(
        args.csv_path,
        q_col=args.q_col,
        r_col=args.r_col,
    )

    if args.num_q is None:
        num_q = max_q + 1
    else:
        num_q = args.num_q

    print(f"Number of sentences: {len(sentences)}")
    print(f"Max qid in csv: {max_q}")
    print(f"num_q used for export: {num_q}")

    model = train_fasttext(
        sentences=sentences,
        emb_size=args.emb_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
    )

    export_interaction_embedding(
        model=model,
        num_q=num_q,
        emb_size=args.emb_size,
        output_path=args.output_path,
    )


if __name__ == "__main__":

    # python examples/prepare_qdkt_fasttext_init.py --csv_path ../data/assist2017/train_valid_sequences.csv --output_path ../data/assist2017/qdkt_fasttext_emb_100.npy --num_q 1183 --emb_size 200 --epochs 20 --window 5
    main()