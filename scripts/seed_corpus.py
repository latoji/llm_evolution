"""Create a deterministic seed corpus for integration and E2E smoke tests."""

from pathlib import Path

_SEED_PARAGRAPH = (
    "It was a bright cold day in April, and the clocks were striking thirteen. "
    "The hallway smelled of boiled cabbage and old paper, yet the open window "
    "still carried a clean breath of spring across the room. "
    "A careful reader could stand at the table, turn a page, and follow one "
    "clear sentence after another without stumbling over noise, code, or broken words. "
    "That is exactly what this seed file is meant to provide: steady English prose, "
    "plain vocabulary, and punctuation gentle enough for a word validator to accept. "
    "The passage is not here for literary importance. "
    "It exists so every test run can begin with the same dependable text, "
    "split into sensible chunks, pass the screening stage, and drive the full pipeline "
    "from upload to evaluation to generation."
)


def build_seed_text(repetitions: int = 8) -> str:
    """Return ~5 KB of deterministic English prose."""
    blocks = [_SEED_PARAGRAPH for _ in range(repetitions)]
    return "\n\n".join(blocks) + "\n"


def write_seed_file(path: Path = Path("data/seed.txt")) -> Path:
    """Write the deterministic seed corpus to *path* and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_seed_text(), encoding="utf-8")
    return path


if __name__ == "__main__":
    written = write_seed_file()
    print(f"Wrote deterministic seed corpus to {written}")
