import argparse
from pathlib import Path

import numpy as np


def read_vocab_size(prefix: Path) -> int:
    with prefix.with_suffix(".tsv").open("r", encoding="utf-8") as file:
        first = file.readline().rstrip("\n").split("\t")
    if len(first) != 2 or first[0] != "# n_vocab":
        raise ValueError(f"invalid metadata in {prefix.with_suffix('.tsv')}")
    return int(first[1])


def read_logits(prefix: Path, n_vocab: int) -> np.ndarray:
    values = np.fromfile(prefix.with_suffix(".f32"), dtype="<f4")
    if values.size % n_vocab != 0:
        raise ValueError(f"invalid row size in {prefix.with_suffix('.f32')}")
    return values.reshape(-1, n_vocab)


def top_two(row: np.ndarray) -> tuple[int, int]:
    ids = np.argpartition(row, -2)[-2:]
    ids = ids[np.argsort(row[ids])[::-1]]
    return int(ids[0]), int(ids[1])


def rank_of(row: np.ndarray, token: int) -> int:
    return int(np.count_nonzero(row > row[token]) + 1)


def print_candidate(label: str, row: np.ndarray, top1: int, top2: int, other_top1: int) -> None:
    print(f"{label}.top1_id={top1}")
    print(f"{label}.top1_logit={float(row[top1]):.9g}")
    print(f"{label}.top2_id={top2}")
    print(f"{label}.top2_logit={float(row[top2]):.9g}")
    print(f"{label}.margin={float(row[top1] - row[top2]):.9g}")
    print(f"{label}.other_top1_logit={float(row[other_top1]):.9g}")
    print(f"{label}.other_top1_rank={rank_of(row, other_top1)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--a-name", default="a")
    parser.add_argument("--b-name", default="b")
    args = parser.parse_args()

    n_vocab_a = read_vocab_size(args.a)
    n_vocab_b = read_vocab_size(args.b)
    if n_vocab_a != n_vocab_b:
        raise ValueError(f"vocab mismatch: {n_vocab_a} != {n_vocab_b}")

    a = read_logits(args.a, n_vocab_a)
    b = read_logits(args.b, n_vocab_a)
    rows = min(a.shape[0], b.shape[0])
    if rows == 0:
        raise ValueError("no common rows")

    a_top1 = np.argmax(a[:rows], axis=1)
    b_top1 = np.argmax(b[:rows], axis=1)
    mismatch = np.flatnonzero(a_top1 != b_top1)
    step = int(mismatch[0]) if mismatch.size else rows - 1
    comparable_rows = step + 1
    delta = a[:comparable_rows] - b[:comparable_rows]
    row_delta = delta[step]
    a1, a2 = top_two(a[step])
    b1, b2 = top_two(b[step])

    print(f"n_vocab={n_vocab_a}")
    print(f"{args.a_name}.rows={a.shape[0]}")
    print(f"{args.b_name}.rows={b.shape[0]}")
    print(f"argmax_prefix_equal={step if mismatch.size else rows}")
    print(f"first_argmax_mismatch={step if mismatch.size else 'none'}")
    print_candidate(args.a_name, a[step], a1, a2, b1)
    print_candidate(args.b_name, b[step], b1, b2, a1)
    print(f"row.max_abs={float(np.max(np.abs(row_delta))):.9g}")
    print(f"row.mean_abs={float(np.mean(np.abs(row_delta))):.9g}")
    print(f"row.rmse={float(np.sqrt(np.mean(row_delta * row_delta))):.9g}")
    print(f"row.p99_abs={float(np.quantile(np.abs(row_delta), 0.99)):.9g}")
    print(f"prefix.max_abs={float(np.max(np.abs(delta))):.9g}")
    print(f"prefix.mean_abs={float(np.mean(np.abs(delta))):.9g}")
    print(f"prefix.rmse={float(np.sqrt(np.mean(delta * delta))):.9g}")
    print(f"finite={bool(np.isfinite(a[:comparable_rows]).all() and np.isfinite(b[:comparable_rows]).all())}")


if __name__ == "__main__":
    main()
