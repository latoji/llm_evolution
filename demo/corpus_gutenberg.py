"""
Gutenberg corpus downloader for the demo setup.

Provides the canonical list of ~150 public-domain books and a downloader
that uses only stdlib (urllib) — no extra pip dependencies.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

MAX_CHARS_PER_BOOK = 120_000  # chars per book (matches original pipeline cap)
DELAY = 0.5  # seconds between requests

# ---------------------------------------------------------------------------
# Book catalogue
# ---------------------------------------------------------------------------

GUTENBERG_BOOKS: list[tuple[int, str]] = [
    # ── Novels & Fiction ──
    (1342, "Pride and Prejudice"),
    (11, "Alice in Wonderland"),
    (1661, "Adventures of Sherlock Holmes"),
    (84, "Frankenstein"),
    (98, "A Tale of Two Cities"),
    (174, "The Picture of Dorian Gray"),
    (1400, "Great Expectations"),
    (161, "Sense and Sensibility"),
    (158, "Emma"),
    (1260, "Jane Eyre"),
    (768, "Wuthering Heights"),
    (345, "Dracula"),
    (2701, "Moby Dick"),
    (76, "Adventures of Huckleberry Finn"),
    (74, "Adventures of Tom Sawyer"),
    (215, "The Call of the Wild"),
    (120, "Treasure Island"),
    (1399, "Anna Karenina"),
    (2600, "War and Peace"),
    (2554, "Crime and Punishment"),
    (28054, "The Brothers Karamazov"),
    (5200, "Metamorphosis"),
    (219, "Heart of Darkness"),
    (829, "Gullivers Travels"),
    (45, "Anne of Green Gables"),
    (55, "The Wonderful Wizard of Oz"),
    (16, "Peter Pan"),
    (46, "A Christmas Carol"),
    (43, "The Strange Case of Dr Jekyll and Mr Hyde"),
    (244, "A Study in Scarlet"),
    (2852, "The Hound of the Baskervilles"),
    (2097, "The Sign of the Four"),
    (1952, "The Yellow Wallpaper"),
    (25344, "The Scarlet Letter"),
    (4300, "Ulysses"),
    (2641, "A Room with a View"),
    (910, "White Fang"),
    (35, "The Time Machine"),
    (36, "The War of the Worlds"),
    (5230, "The Invisible Man"),
    (159, "The Island of Doctor Moreau"),
    (3600, "The Jungle Book"),
    (2591, "Grimms Fairy Tales"),
    (1184, "The Count of Monte Cristo"),
    (1080, "A Modest Proposal"),
    (394, "Cranford"),
    (145, "Middlemarch"),
    (730, "Oliver Twist"),
    (766, "David Copperfield"),
    (580, "The Pickwick Papers"),
    (1023, "Bleak House"),
    (564, "The Scarlet Pimpernel"),
    (6130, "The Iliad"),
    (1727, "The Odyssey"),
    (600, "Notes from Underground"),
    (996, "Don Quixote"),
    (1232, "The Prince"),
    (514, "Little Women"),
    (3207, "Leviathan"),
    (1497, "The Republic"),
    (4363, "Beyond Good and Evil"),
    (1998, "Thus Spake Zarathustra"),
    (2680, "Meditations"),
    (8800, "The Divine Comedy"),
    (8492, "The King in Yellow"),
    (2148, "The Phantom of the Opera"),
    (236, "The Jungle"),
    (113, "The Secret Garden"),
    (160, "The Awakening"),
    (209, "The Turn of the Screw"),
    (1250, "Tess of the dUrbervilles"),
    (110, "Taming of the Shrew"),
    (2500, "Siddhartha"),
    (4517, "Dubliners"),
    (521, "The Life and Adventures of Robinson Crusoe"),
    (3825, "Pygmalion"),
    (1322, "Leaves of Grass"),
    # ── Drama & Poetry ──
    (100, "The Complete Works of Shakespeare"),
    (1513, "Romeo and Juliet"),
    (1524, "Hamlet"),
    (1533, "Macbeth"),
    (1531, "A Midsummer Nights Dream"),
    (1532, "King Lear"),
    (1529, "The Merchant of Venice"),
    (1511, "Othello"),
    (1519, "The Tempest"),
    (1514, "Julius Caesar"),
    (1521, "Twelfth Night"),
    (844, "The Importance of Being Earnest"),
    (2542, "A Dolls House"),
    (1259, "Twenty Thousand Leagues Under the Sea"),
    (103, "Around the World in Eighty Days"),
    (164, "Journey to the Centre of the Earth"),
    # ── Non-Fiction, Philosophy, Science ──
    (205, "Walden"),
    (408, "The Souls of Black Folk"),
    (23, "Narrative of the Life of Frederick Douglass"),
    (1228, "On the Origin of Species"),
    (7370, "Second Treatise of Government"),
    (3296, "Confessions of St Augustine"),
    (16328, "Beowulf"),
    (22788, "The Federalist Papers"),
    (5827, "The Problems of Philosophy"),
    (815, "Democracy in America Vol 1"),
    (816, "Democracy in America Vol 2"),
    (1404, "An Enquiry Concerning Human Understanding"),
    (4280, "The Art of War"),
    (10762, "An Essay Concerning Human Understanding"),
    (4705, "A Treatise of Human Nature"),
    (3076, "Critique of Pure Reason"),
    (7142, "Discourse on the Method"),
    (2130, "Utopia"),
    (3300, "An Inquiry Into the Nature and Causes of the Wealth of Nations"),
    (4657, "A Vindication of the Rights of Woman"),
    (852, "The Communist Manifesto"),
    (30254, "The Elements of Style"),
    (34901, "Methods of Ethics"),
    (26184, "Simple Sabotage Field Manual"),
    (33283, "Calculus Made Easy"),
    (28520, "The Essays of Montaigne"),
    # ── Adventure & Misc ──
    (2814, "An Occurrence at Owl Creek Bridge"),
    (1257, "The Three Musketeers"),
    (27827, "The Kama Sutra"),
    (41, "The Legend of Sleepy Hollow"),
    (932, "The Fall of the House of Usher"),
    (2147, "Les Miserables"),
    (17135, "Anthem"),
    (5740, "Tractatus Logico-Philosophicus"),
    (10, "The Bible King James Version"),
    (3160, "The Decameron"),
    (7849, "Candide"),
    (30601, "The Raven"),
    (4217, "A Portrait of the Artist as a Young Man"),
    (1695, "The Moonstone"),
    (27780, "The Canterbury Tales"),
    (47629, "Winnie the Pooh"),
    (67979, "The Blue Castle"),
]


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def _download_book(book_id: int, title: str, idx: int = 0, total: int = 0) -> str:
    """Download a single Gutenberg book; return stripped text (or empty str)."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    tag = f"[{idx}/{total}]" if total else ""
    print(f"  {tag} {title} (ID {book_id})…", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Python-demo/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        # Strip Gutenberg header/footer boilerplate
        start_markers = [
            "*** START OF THE PROJECT GUTENBERG",
            "*** START OF THIS PROJECT GUTENBERG",
        ]
        end_markers = [
            "*** END OF THE PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
        ]
        text = raw
        for marker in start_markers:
            pos = text.find(marker)
            if pos != -1:
                newline = text.find("\n", pos)
                text = text[newline + 1:] if newline != -1 else text[pos + len(marker):]
                break
        for marker in end_markers:
            pos = text.find(marker)
            if pos != -1:
                text = text[:pos]
                break

        text = text[:MAX_CHARS_PER_BOOK]
        print(f"{len(text):,} chars")
        return text
    except Exception as exc:
        print(f"FAILED ({exc})")
        return ""


def download_all(delay: float = DELAY) -> str:
    """Download all configured Gutenberg books; return combined text."""
    import time

    # De-duplicate by book ID
    seen: set[int] = set()
    unique: list[tuple[int, str]] = []
    for book_id, title in GUTENBERG_BOOKS:
        if book_id not in seen:
            seen.add(book_id)
            unique.append((book_id, title))

    total = len(unique)
    parts: list[str] = []
    for i, (book_id, title) in enumerate(unique, 1):
        text = _download_book(book_id, title, idx=i, total=total)
        if text:
            parts.append(text)
        time.sleep(delay)

    print(f"\n  ✓ Downloaded {len(parts)}/{total} Gutenberg books")
    return "\n\n".join(parts)
