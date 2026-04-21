"""
Track Z – Evaluation & Ablations
Compute perplexity on the validation set for any LanguageModel configuration.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
VAL_CORPUS = BASE_DIR / "data" / "clean" / "val.txt"
MODEL_PATH = BASE_DIR / "model" / "lm.pkl"
TOKENIZER_DIR = BASE_DIR / "tokenizer"


def compute_perplexity(
    lm,
    merges: list[tuple[str, str]],
    val_path: Path,
    max_order: int = 3,
    max_tokens: int | None = None,
) -> float:
    """
    Compute perplexity of lm on val_path.

    Args:
        lm:         LanguageModel instance.
        merges:     BPE merges (used for encoding val text).
        val_path:   Path to validation text (one paragraph per line).
        max_order:  Context window size for probability queries.
        max_tokens: Optional cap on total tokens evaluated (for speed).

    Returns:
        Perplexity (float). Lower is better.
    """
    from tokenizer.bpe import encode  # noqa: PLC0415

    log_prob_sum = 0.0
    n_tokens = 0
    history: list[int] = []

    with open(val_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue

            token_ids = encode(line, merges)
            for token in token_ids:
                context = tuple(history[-max_order:])
                p = lm.prob(token, context)
                log_prob_sum += math.log(max(p, 1e-300))
                n_tokens += 1

                history.append(token)
                if len(history) > max_order:
                    history.pop(0)

                if max_tokens and n_tokens >= max_tokens:
                    break
            else:
                continue
            break

    if n_tokens == 0:
        return float("inf")

    avg_log = log_prob_sum / n_tokens
    ppl = math.exp(-avg_log)
    return ppl


def main(
    model_path: Path = MODEL_PATH,
    val_path: Path = VAL_CORPUS,
    max_tokens: int | None = None,
) -> None:
    from model.language_model import LanguageModel  # noqa: PLC0415
    from tokenizer.bpe import load_tokenizer  # noqa: PLC0415

    lm = LanguageModel.load(model_path)
    merges, _ = load_tokenizer(TOKENIZER_DIR / "merges.json", TOKENIZER_DIR / "vocab.json")

    print(f"Evaluating perplexity on {val_path}…")
    ppl = compute_perplexity(lm, merges, val_path, max_tokens=max_tokens)
    print(f"Perplexity: {ppl:.2f}")
    return ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate language model perplexity")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--val", type=Path, default=VAL_CORPUS)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Cap evaluation at N tokens for speed",
    )
    args = parser.parse_args()
    main(args.model, args.val, args.max_tokens)
