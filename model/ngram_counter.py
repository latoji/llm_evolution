"""
N-Gram Frequency Extraction — BPE, character, and word modes.

Provides:
  - Standalone functions (count_ngrams, prune_counts, save_counts, load_counts)
    for the legacy BPE pipeline.
  - NGramCounter class supporting char / word / bpe tokenization with configurable
    max order (up to 5 for char, 3 for word/bpe).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Literal

from tqdm import tqdm

COUNTS_DIR = Path(__file__).parent / "counts"
MAX_N = 4
PRUNE_MIN_COUNT = 3
CHUNK_BYTES = 100 * 1024 * 1024  # 100 MB per processing chunk

# Sentinel token ID for document boundaries
ENDOFTEXT_ID = 0

# Supported tokenization modes
Mode = Literal["char", "word", "bpe"]


# ---------------------------------------------------------------------------
# Core counting
# ---------------------------------------------------------------------------

def count_ngrams(
    token_stream: Iterator[int],
    max_n: int = MAX_N,
) -> dict[int, Counter]:
    """
    Slide a window of sizes 1..max_n over token_stream and count every tuple.

    Args:
        token_stream: Flat iterator of integer token IDs.
        max_n:        Maximum n-gram order (inclusive).

    Returns:
        counts: dict mapping order n → Counter of n-gram tuples.
    """
    counts: dict[int, Counter] = {n: Counter() for n in range(1, max_n + 1)}
    window: list[int] = []

    for token in token_stream:
        window.append(token)
        if len(window) > max_n:
            window.pop(0)

        for n in range(1, min(len(window), max_n) + 1):
            ngram = tuple(window[-n:])
            counts[n][ngram] += 1

    return counts


def prune_counts(
    counts: dict[int, Counter],
    min_count: int = PRUNE_MIN_COUNT,
) -> dict[int, Counter]:
    """Remove all n-grams with count < min_count."""
    return {
        n: Counter({ng: c for ng, c in counter.items() if c >= min_count})
        for n, counter in counts.items()
    }


# ---------------------------------------------------------------------------
# Chunked tokenised stream from a text file
# ---------------------------------------------------------------------------

def _token_stream_from_file(
    corpus_path: Path,
    merges: list[tuple[str, str]],
    chunk_bytes: int = CHUNK_BYTES,
) -> Iterator[int]:
    """
    Tokenise corpus_path in chunks and yield token IDs.
    Inserts ENDOFTEXT_ID between chunks (acts as document boundary).
    """
    # Import here to avoid circular deps at module level
    from tokenizer.bpe import encode  # noqa: PLC0415

    buffer = ""
    file_size = corpus_path.stat().st_size

    with open(corpus_path, encoding="utf-8", errors="replace") as fh, tqdm(
        total=file_size, unit="B", unit_scale=True, desc="Tokenising"
    ) as bar:
        for line in fh:
            buffer += line
            bar.update(len(line.encode("utf-8")))

            if len(buffer.encode("utf-8")) >= chunk_bytes:
                yield from encode(buffer, merges)
                yield ENDOFTEXT_ID
                buffer = ""

        if buffer:
            yield from encode(buffer, merges)
            yield ENDOFTEXT_ID


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_counts(counts: dict[int, Counter], counts_dir: Path) -> None:
    """
    Persist counts to JSON files.
    Keys are stored as space-separated token ID strings for JSON compatibility.
    """
    counts_dir.mkdir(parents=True, exist_ok=True)
    names = {1: "unigrams", 2: "bigrams", 3: "trigrams", 4: "fourgrams"}

    for n, counter in counts.items():
        name = names.get(n, f"{n}grams")
        out_path = counts_dir / f"{name}.json"
        data = {" ".join(map(str, ng)): c for ng, c in counter.items()}
        out_path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        print(f"  {name}: {len(counter):,} entries → {out_path}")


def load_counts(counts_dir: Path, max_n: int = MAX_N) -> dict[int, Counter]:
    """Load counts from JSON files produced by save_counts."""
    names = {1: "unigrams", 2: "bigrams", 3: "trigrams", 4: "fourgrams"}
    counts: dict[int, Counter] = {}

    for n in range(1, max_n + 1):
        name = names.get(n, f"{n}grams")
        path = counts_dir / f"{name}.json"
        if not path.exists():
            print(f"  Warning: {path} not found, skipping order {n}")
            continue
        raw: dict[str, int] = json.loads(path.read_text(encoding="utf-8"))
        counts[n] = Counter(
            {tuple(int(x) for x in k.split()): v for k, v in raw.items()}
        )

    return counts


# ---------------------------------------------------------------------------
# Generic n-gram counting (works with any hashable token type)
# ---------------------------------------------------------------------------

def _count_ngrams_generic(
    tokens: list[Any],
    max_n: int,
) -> dict[int, Counter]:
    """Slide a window over *tokens* and count every n-gram tuple up to max_n.

    Works with strings (char/word mode) or integers (bpe mode).
    Returns {order: Counter of n-gram tuples}.
    """
    counts: dict[int, Counter] = {n: Counter() for n in range(1, max_n + 1)}
    window: list[Any] = []

    for token in tokens:
        window.append(token)
        if len(window) > max_n:
            window.pop(0)

        for n in range(1, min(len(window), max_n) + 1):
            ngram = tuple(window[-n:])
            counts[n][ngram] += 1

    return counts


# ---------------------------------------------------------------------------
# NGramCounter class — char / word / bpe with configurable order
# ---------------------------------------------------------------------------

class NGramCounter:
    """Multi-modal n-gram counter supporting character, word, and BPE token modes."""

    def __init__(self, max_order: int, mode: Mode, min_count: int = 3) -> None:
        """
        Args:
            max_order: Maximum n-gram order (1–5 for char, 1–3 for word/bpe).
            mode:      Tokenization mode.
                       'char' — tokenize input string into characters (whitespace kept).
                       'word' — tokenize via wvm.validator.Validator.tokenize.
                       'bpe'  — caller provides already-tokenized BPE token IDs.
            min_count: Prune n-grams with count < min_count after counting.
        """
        if mode not in ("char", "word", "bpe"):
            raise ValueError(f"mode must be 'char', 'word', or 'bpe'; got {mode!r}")
        if max_order < 1:
            raise ValueError(f"max_order must be >= 1; got {max_order}")
        self.max_order = max_order
        self.mode: Mode = mode
        self.min_count = min_count

    # ------------------------------------------------------------------
    # Public counting API
    # ------------------------------------------------------------------

    def count(self, input_data: str | list[int]) -> dict[int, Counter]:
        """Count n-grams in the given input.

        For 'char' and 'word' modes, *input_data* must be a str.
        For 'bpe' mode, *input_data* must be a list of integer token IDs.

        Returns:
            counts: dict mapping order n → Counter of n-gram tuples.
                    Char/word n-gram tuples contain strings; BPE tuples contain ints.
        """
        tokens = self._tokenize(input_data)
        return _count_ngrams_generic(tokens, self.max_order)

    def count_and_prune(self, input_data: str | list[int]) -> dict[int, Counter]:
        """Count n-grams and prune those below min_count."""
        raw = self.count(input_data)
        return prune_counts(raw, self.min_count)

    # ------------------------------------------------------------------
    # Internal tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, input_data: str | list[int]) -> list[Any]:
        """Convert input to a flat token list according to self.mode."""
        if self.mode == "char":
            if not isinstance(input_data, str):
                raise TypeError("char mode requires a str input")
            return list(input_data)  # each character is a token (includes whitespace)
        elif self.mode == "word":
            if not isinstance(input_data, str):
                raise TypeError("word mode requires a str input")
            # Import lazily to avoid circular imports
            from wvm.validator import Validator  # noqa: PLC0415
            validator = Validator()
            return validator.tokenize(input_data)
        else:  # bpe
            if isinstance(input_data, str):
                raise TypeError("bpe mode requires a list[int] input")
            return list(input_data)
