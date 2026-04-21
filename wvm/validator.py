"""Word Verification Module — classify tokens against the SCOWL size-70 English word list."""

import difflib
from pathlib import Path
from typing import NamedTuple

PUNCTUATION_BOUNDARY: set[str] = set(".,!?;:\"'()[]{}\u2014\u2026`")

_DEFAULT_WORDLIST = Path(__file__).parent / "scowl_70.txt"


class WordResult(NamedTuple):
    """Result of validating a single token."""

    raw: str        # original token as it appeared in the input
    word: str       # stripped + lowercased version used for lookup
    is_real: bool   # True if word is in the SCOWL set


class Validator:
    """SCOWL-backed English word validator.

    Loads the word list once at construction time; all subsequent calls are O(1) per lookup.
    """

    def __init__(self, wordlist_path: Path = _DEFAULT_WORDLIST) -> None:
        """Load SCOWL wordlist into memory as a frozenset of lowercase words."""
        if not wordlist_path.exists():
            raise FileNotFoundError(
                f"SCOWL wordlist not found at '{wordlist_path}'. "
                "Download SCOWL size-70 from http://wordlist.aspell.net/ and place it at "
                f"'{wordlist_path}'."
            )
        words: set[str] = set()
        with wordlist_path.open(encoding="utf-8") as fh:
            for line in fh:
                word = line.strip()
                # Skip blank lines, comment/header lines
                if not word or word.startswith("#") or word == "---":
                    continue
                words.add(word.lower())
        self._words: frozenset[str] = frozenset(words)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Split on whitespace; strip boundary punctuation; preserve internal apostrophes.

        Tokenization rules (applied in order):
        1. Split on any whitespace.
        2. Strip leading chars in PUNCTUATION_BOUNDARY until a non-punctuation char or end.
        3. Strip trailing chars in PUNCTUATION_BOUNDARY until a non-punctuation char or end.
        4. Lowercase.
        5. Skip empty tokens.
        Internal apostrophes survive because they are mid-token (e.g. ``it's``, ``don't``).
        """
        tokens: list[str] = []
        for raw_token in text.split():
            stripped = _strip_boundary(raw_token)
            lowered = stripped.lower()
            if lowered:
                tokens.append(lowered)
        return tokens

    def validate(self, text: str) -> tuple[list[WordResult], float]:
        """Return per-word results and the overall real-word percentage (0.0–1.0).

        Empty input returns ([], 0.0).
        """
        if not text or not text.strip():
            return [], 0.0

        results: list[WordResult] = []
        for raw_token in text.split():
            stripped = _strip_boundary(raw_token)
            lowered = stripped.lower()
            if not lowered:
                continue
            is_real = lowered in self._words
            results.append(WordResult(raw=raw_token, word=lowered, is_real=is_real))

        if not results:
            return [], 0.0

        real_count = sum(1 for r in results if r.is_real)
        pct = real_count / len(results)
        return results, pct

    def suggest(self, word: str, n: int = 1) -> list[str]:
        """Return up to *n* nearest valid words via difflib.get_close_matches.

        Used by the Generation page auto-correct feature.
        """
        return difflib.get_close_matches(word.lower(), self._words, n=n, cutoff=0.6)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _strip_boundary(token: str) -> str:
    """Strip leading and trailing PUNCTUATION_BOUNDARY characters from *token*."""
    left = 0
    while left < len(token) and token[left] in PUNCTUATION_BOUNDARY:
        left += 1
    right = len(token)
    while right > left and token[right - 1] in PUNCTUATION_BOUNDARY:
        right -= 1
    return token[left:right]
