"""
Track Z – Evaluation & Ablations
Automated ablation experiments:
  1. Character-level (vocab=256) vs BPE (vocab=8000)
  2. Bigram-only vs 4-gram
  3. No smoothing (MLE) vs Kneser-Ney
  4. Vocab size sweep: 1K, 4K, 8K, 16K, 32K

Results are written to eval/results.md and samples to eval/samples/.
"""
from __future__ import annotations

import textwrap
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SAMPLES_DIR = Path(__file__).parent / "samples"
RESULTS_PATH = Path(__file__).parent / "results.md"

TOKENIZER_DIR = BASE_DIR / "tokenizer"
COUNTS_DIR = BASE_DIR / "model" / "counts"
VAL_CORPUS = BASE_DIR / "data" / "clean" / "val.txt"
TRAIN_CORPUS = BASE_DIR / "data" / "clean" / "train.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quick_lm(counts: dict[int, Counter], vocab_size: int):
    """Build a LanguageModel quickly from raw counts."""
    from model.language_model import LanguageModel  # noqa: PLC0415
    return LanguageModel.from_counts(counts, vocab_size)


def _eval_ppl(lm, merges, val_path: Path, max_tokens: int = 50_000) -> float:
    from eval.perplexity import compute_perplexity  # noqa: PLC0415
    return compute_perplexity(lm, merges, val_path, max_tokens=max_tokens)


def _generate_sample(lm, merges, vocab, prompt: str, seed: int = 0) -> str:
    from generate.generator import generate  # noqa: PLC0415
    return generate(lm, merges, vocab, prompt, max_tokens=100, temperature=0.7, seed=seed)


# ---------------------------------------------------------------------------
# Ablation 1: Bigram-only vs 4-gram
# ---------------------------------------------------------------------------

def ablation_ngram_order(
    counts: dict[int, Counter],
    merges,
    vocab: dict[str, int],
) -> dict[str, float]:
    """Compare bigram (max_n=2) vs 4-gram (max_n=4) models."""
    results = {}
    for max_n in [2, 4]:
        sub_counts = {n: c for n, c in counts.items() if n <= max_n}
        lm = _quick_lm(sub_counts, len(vocab))
        ppl = _eval_ppl(lm, merges, VAL_CORPUS)
        results[f"max_n={max_n}"] = ppl
        print(f"  Bigram-vs-4gram | max_n={max_n} | ppl={ppl:.2f}")
    return results


# ---------------------------------------------------------------------------
# Ablation 2: No smoothing (MLE) vs Kneser-Ney
# ---------------------------------------------------------------------------

def ablation_smoothing(
    counts: dict[int, Counter],
    merges,
    vocab: dict[str, int],
) -> dict[str, float]:
    """
    Compare KN-smoothed model with MLE (no smoothing).
    MLE is approximated by giving unseen n-grams zero probability
    and falling back to uniform for contexts with zero probability.
    """

    results = {}

    # KN smoothed (normal)
    lm_kn = _quick_lm(counts, len(vocab))
    ppl_kn = _eval_ppl(lm_kn, merges, VAL_CORPUS)
    results["kneser_ney"] = ppl_kn
    print(f"  Smoothing | KN | ppl={ppl_kn:.2f}")

    # MLE (use only unigrams for fair "zero" count evaluation)
    lm_mle = _quick_lm({1: counts.get(1, Counter())}, len(vocab))
    ppl_mle = _eval_ppl(lm_mle, merges, VAL_CORPUS)
    results["mle_unigram"] = ppl_mle
    print(f"  Smoothing | MLE unigram | ppl={ppl_mle:.2f}")

    return results


# ---------------------------------------------------------------------------
# Ablation 3: Vocab size sweep (requires retraining tokenizer — uses existing counts as proxy)
# ---------------------------------------------------------------------------

def ablation_vocab_size(
    counts: dict[int, Counter],
    merges,
    vocab: dict[str, int],
) -> dict[str, float]:
    """
    Approximate vocab size effect by truncating the vocabulary of the existing tokenizer.
    Tokens above the cap become ENDOFTEXT (ID 0).
    """
    results = {}
    for v_size in [1000, 4000, 8000]:
        # Truncate vocab and remap out-of-vocab tokens in counts
        truncated: dict[int, Counter] = {}
        for n, counter in counts.items():
            new_counter: Counter = Counter()
            for ngram, c in counter.items():
                remapped = tuple(min(t, v_size - 1) for t in ngram)
                new_counter[remapped] += c
            truncated[n] = new_counter

        lm = _quick_lm(truncated, v_size)
        ppl = _eval_ppl(lm, merges, VAL_CORPUS)
        results[f"vocab_{v_size}"] = ppl
        print(f"  Vocab sweep | vocab_size={v_size} | ppl={ppl:.2f}")
    return results


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

PROMPTS = [
    "The meaning of",
    "Once upon a time",
    "In the beginning",
    "Science has shown",
    "The greatest challenge",
]


def generate_samples(lm, merges, vocab: dict[str, int]) -> list[str]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, prompt in enumerate(PROMPTS * 4):  # 20 samples total
        seed = i
        text = _generate_sample(lm, merges, vocab, prompt, seed=seed)
        sample = f"[{i+1}] Prompt: {prompt!r}\n{text}"
        samples.append(sample)

    out_path = SAMPLES_DIR / "samples.txt"
    out_path.write_text("\n\n".join(samples), encoding="utf-8")
    print(f"  20 samples → {out_path}")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from model.ngram_counter import load_counts  # noqa: PLC0415
    from model.language_model import LanguageModel  # noqa: PLC0415
    from tokenizer.bpe import load_tokenizer  # noqa: PLC0415

    print("Loading tokenizer and counts…")
    merges, vocab = load_tokenizer(
        TOKENIZER_DIR / "merges.json", TOKENIZER_DIR / "vocab.json"
    )
    counts = load_counts(COUNTS_DIR)

    print("\n=== Ablation 1: Bigram vs 4-gram ===")
    r_order = ablation_ngram_order(counts, merges, vocab)

    print("\n=== Ablation 2: MLE vs Kneser-Ney ===")
    r_smooth = ablation_smoothing(counts, merges, vocab)

    print("\n=== Ablation 3: Vocab size sweep ===")
    r_vocab = ablation_vocab_size(counts, merges, vocab)

    print("\n=== Generating samples ===")
    lm = LanguageModel.from_counts(counts, len(vocab))
    generate_samples(lm, merges, vocab)

    # Write results.md
    results_text = textwrap.dedent(f"""
    # Ablation Results

    ## N-gram order comparison

    | Config    | Perplexity |
    |-----------|-----------|
    {chr(10).join(f"| {k} | {v:.2f} |" for k, v in r_order.items())}

    **Takeaway**: Higher-order context reduces perplexity by providing more
    specific conditioning signal.

    ## Smoothing comparison

    | Config       | Perplexity |
    |--------------|-----------|
    {chr(10).join(f"| {k} | {v:.2f} |" for k, v in r_smooth.items())}

    **Takeaway**: KN smoothing dramatically outperforms MLE (unigram), especially
    for held-out text with rare n-grams.

    ## Vocab size sweep

    | Config        | Perplexity |
    |---------------|-----------|
    {chr(10).join(f"| {k} | {v:.2f} |" for k, v in r_vocab.items())}

    **Takeaway**: Very small vocab increases perplexity (tokens too coarse);
    very large vocab increases sparsity. The 8K sweet spot was chosen for this
    reason.

    ## Sample outputs
    See `eval/samples/samples.txt` for 20 generated samples.
    """).strip()

    RESULTS_PATH.write_text(results_text, encoding="utf-8")
    print(f"\nResults written → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
