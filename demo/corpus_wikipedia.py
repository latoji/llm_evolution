"""
Wikipedia corpus downloader for the demo setup.

Downloads ~200 English Wikipedia articles spanning science, history,
geography, arts, philosophy, technology, and nature.  Uses the MediaWiki
REST API (plain-text extract endpoint) with stdlib urllib only.

Target volume: roughly equal to the Gutenberg corpus (~14 MB).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

MAX_CHARS_PER_ARTICLE = 80_000  # chars; comparable to Gutenberg's 120 K cap
DELAY = 0.4  # seconds between requests (be polite to the API)
API_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=query&titles={title}"
    "&prop=extracts&explaintext=1&format=json&redirects=1"
)

# ---------------------------------------------------------------------------
# Article catalogue (~200 titles across diverse categories)
# ---------------------------------------------------------------------------

WIKIPEDIA_ARTICLES: list[tuple[str, str]] = [
    # ── Science: Physics ──
    ("Physics", "Physics"),
    ("Quantum_mechanics", "Quantum mechanics"),
    ("General_relativity", "General relativity"),
    ("Special_relativity", "Special relativity"),
    ("Thermodynamics", "Thermodynamics"),
    ("Electromagnetism", "Electromagnetism"),
    ("Optics", "Optics"),
    ("Nuclear_physics", "Nuclear physics"),
    ("Particle_physics", "Particle physics"),
    ("Standard_Model", "Standard Model"),
    ("Classical_mechanics", "Classical mechanics"),
    ("Wave", "Wave"),
    # ── Science: Chemistry ──
    ("Chemistry", "Chemistry"),
    ("Chemical_element", "Chemical element"),
    ("Periodic_table", "Periodic table"),
    ("Organic_chemistry", "Organic chemistry"),
    ("Biochemistry", "Biochemistry"),
    ("Atom", "Atom"),
    # ── Science: Biology ──
    ("Biology", "Biology"),
    ("Cell_(biology)", "Cell (biology)"),
    ("Genetics", "Genetics"),
    ("Evolution", "Evolution"),
    ("Natural_selection", "Natural selection"),
    ("DNA", "DNA"),
    ("Protein", "Protein"),
    ("Ecology", "Ecology"),
    ("Photosynthesis", "Photosynthesis"),
    ("Neuroscience", "Neuroscience"),
    ("Human_brain", "Human brain"),
    ("Consciousness", "Consciousness"),
    # ── Science: Astronomy ──
    ("Astronomy", "Astronomy"),
    ("Solar_System", "Solar System"),
    ("Galaxy", "Galaxy"),
    ("Black_hole", "Black hole"),
    ("Big_Bang", "Big Bang"),
    ("Universe", "Universe"),
    ("Cosmology", "Cosmology"),
    ("Star", "Star"),
    ("Planet", "Planet"),
    # ── Science: Mathematics ──
    ("Mathematics", "Mathematics"),
    ("Calculus", "Calculus"),
    ("Algebra", "Algebra"),
    ("Geometry", "Geometry"),
    ("Statistics", "Statistics"),
    ("Probability_theory", "Probability theory"),
    ("Number_theory", "Number theory"),
    ("Graph_theory", "Graph theory"),
    # ── Technology & Computing ──
    ("Computer_science", "Computer science"),
    ("Algorithm", "Algorithm"),
    ("Machine_learning", "Machine learning"),
    ("Artificial_intelligence", "Artificial intelligence"),
    ("Deep_learning", "Deep learning"),
    ("Internet", "Internet"),
    ("World_Wide_Web", "World Wide Web"),
    ("Programming_language", "Programming language"),
    ("Operating_system", "Operating system"),
    ("Transistor", "Transistor"),
    ("Semiconductor", "Semiconductor"),
    ("Robotics", "Robotics"),
    ("Information_theory", "Information theory"),
    ("Cryptography", "Cryptography"),
    # ── History: Ancient ──
    ("Ancient_history", "Ancient history"),
    ("Ancient_Egypt", "Ancient Egypt"),
    ("Ancient_Greece", "Ancient Greece"),
    ("Roman_Empire", "Roman Empire"),
    ("Alexander_the_Great", "Alexander the Great"),
    ("Byzantine_Empire", "Byzantine Empire"),
    ("Mesopotamia", "Mesopotamia"),
    ("Silk_Road", "Silk Road"),
    ("Mongol_Empire", "Mongol Empire"),
    ("Chinese_civilization", "History of China"),
    # ── History: Medieval & Early Modern ──
    ("Middle_Ages", "Middle Ages"),
    ("Crusades", "Crusades"),
    ("Renaissance", "Renaissance"),
    ("Age_of_Discovery", "Age of Discovery"),
    ("Scientific_Revolution", "Scientific Revolution"),
    ("Ottoman_Empire", "Ottoman Empire"),
    ("Feudalism", "Feudalism"),
    ("Black_Death", "Black Death"),
    # ── History: Modern ──
    ("French_Revolution", "French Revolution"),
    ("Napoleon", "Napoleon"),
    ("Industrial_Revolution", "Industrial Revolution"),
    ("American_Revolution", "American Revolution"),
    ("American_Civil_War", "American Civil War"),
    ("Abolition_of_slavery", "Abolitionism"),
    ("World_War_I", "World War I"),
    ("World_War_II", "World War II"),
    ("Cold_War", "Cold War"),
    ("Space_Race", "Space Race"),
    ("Decolonization", "Decolonization"),
    ("Great_Depression", "Great Depression"),
    ("Holocaust", "Holocaust"),
    # ── Geography ──
    ("Earth", "Earth"),
    ("Europe", "Europe"),
    ("Asia", "Asia"),
    ("Africa", "Africa"),
    ("North_America", "North America"),
    ("South_America", "South America"),
    ("Australia_(continent)", "Australia (continent)"),
    ("Antarctica", "Antarctica"),
    ("Atlantic_Ocean", "Atlantic Ocean"),
    ("Pacific_Ocean", "Pacific Ocean"),
    ("Mediterranean_Sea", "Mediterranean Sea"),
    ("Amazon_River", "Amazon River"),
    ("Nile", "Nile"),
    ("Himalayas", "Himalayas"),
    ("Amazon_rainforest", "Amazon rainforest"),
    ("Sahara", "Sahara"),
    ("Arctic", "Arctic"),
    ("Climate", "Climate"),
    ("Volcano", "Volcano"),
    ("Earthquake", "Earthquake"),
    # ── Notable People: Scientists & Philosophers ──
    ("Isaac_Newton", "Isaac Newton"),
    ("Albert_Einstein", "Albert Einstein"),
    ("Charles_Darwin", "Charles Darwin"),
    ("Galileo_Galilei", "Galileo Galilei"),
    ("Marie_Curie", "Marie Curie"),
    ("Leonardo_da_Vinci", "Leonardo da Vinci"),
    ("Aristotle", "Aristotle"),
    ("Plato", "Plato"),
    ("Socrates", "Socrates"),
    ("Immanuel_Kant", "Immanuel Kant"),
    ("Friedrich_Nietzsche", "Friedrich Nietzsche"),
    ("Sigmund_Freud", "Sigmund Freud"),
    ("Karl_Marx", "Karl Marx"),
    ("Nikola_Tesla", "Nikola Tesla"),
    ("Ada_Lovelace", "Ada Lovelace"),
    ("Alan_Turing", "Alan Turing"),
    ("Stephen_Hawking", "Stephen Hawking"),
    # ── Notable People: Writers ──
    ("William_Shakespeare", "William Shakespeare"),
    ("Homer", "Homer"),
    ("Dante_Alighieri", "Dante Alighieri"),
    ("Virgil", "Virgil"),
    ("Leo_Tolstoy", "Leo Tolstoy"),
    ("Fyodor_Dostoevsky", "Fyodor Dostoevsky"),
    ("Jane_Austen", "Jane Austen"),
    ("Charles_Dickens", "Charles Dickens"),
    ("Mark_Twain", "Mark Twain"),
    ("Ernest_Hemingway", "Ernest Hemingway"),
    ("Virginia_Woolf", "Virginia Woolf"),
    ("Franz_Kafka", "Franz Kafka"),
    # ── Arts & Culture ──
    ("Literature", "Literature"),
    ("Novel", "Novel"),
    ("Poetry", "Poetry"),
    ("Theatre", "Theatre"),
    ("Music", "Music"),
    ("Classical_music", "Classical music"),
    ("Jazz", "Jazz"),
    ("Rock_music", "Rock music"),
    ("Film", "Film"),
    ("Painting", "Painting"),
    ("Sculpture", "Sculpture"),
    ("Architecture", "Architecture"),
    ("Renaissance_art", "Renaissance art"),
    ("Impressionism", "Impressionism"),
    ("Modernism", "Modernism"),
    ("Opera", "Opera"),
    ("Ballet", "Ballet"),
    # ── Philosophy & Society ──
    ("Philosophy", "Philosophy"),
    ("Ethics", "Ethics"),
    ("Epistemology", "Epistemology"),
    ("Metaphysics", "Metaphysics"),
    ("Logic", "Logic"),
    ("Aesthetics", "Aesthetics"),
    ("Political_philosophy", "Political philosophy"),
    ("Democracy", "Democracy"),
    ("Human_rights", "Human rights"),
    ("Capitalism", "Capitalism"),
    ("Socialism", "Socialism"),
    ("Psychology", "Psychology"),
    ("Sociology", "Sociology"),
    ("Anthropology", "Anthropology"),
    ("Economics", "Economics"),
    ("Linguistics", "Linguistics"),
    ("Language", "Language"),
    # ── Religion & Mythology ──
    ("Religion", "Religion"),
    ("Christianity", "Christianity"),
    ("Islam", "Islam"),
    ("Buddhism", "Buddhism"),
    ("Hinduism", "Hinduism"),
    ("Judaism", "Judaism"),
    ("Mythology", "Mythology"),
    ("Greek_mythology", "Greek mythology"),
    ("Roman_mythology", "Roman mythology"),
    # ── Nature & Environment ──
    ("Animal", "Animal"),
    ("Mammal", "Mammal"),
    ("Bird", "Bird"),
    ("Fish", "Fish"),
    ("Insect", "Insect"),
    ("Plant", "Plant"),
    ("Biodiversity", "Biodiversity"),
    ("Climate_change", "Climate change"),
    ("Atmosphere_of_Earth", "Atmosphere of Earth"),
    ("Biosphere", "Biosphere"),
    ("Extinction", "Extinction"),
    ("Food_chain", "Food chain"),
    ("Ocean", "Ocean"),
    ("Forest", "Forest"),
    ("Desert", "Desert"),
]


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def _download_article(slug: str, display_name: str, idx: int = 0, total: int = 0) -> str:
    """
    Download one Wikipedia article using the MediaWiki API extracts endpoint.
    Returns plain text or empty string on failure.
    """
    tag = f"[{idx}/{total}]" if total else ""
    print(f"  {tag} {display_name}…", end=" ", flush=True)

    encoded = urllib.parse.quote(slug, safe="()_")
    url = API_URL.format(title=encoded)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Python-demo/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            print("EMPTY")
            return ""

        page = next(iter(pages.values()))
        text = page.get("extract", "") or ""

        # Trim to cap
        text = text[:MAX_CHARS_PER_ARTICLE]
        if len(text) < 200:
            print(f"too short ({len(text)} chars)")
            return ""

        print(f"{len(text):,} chars")
        return text
    except Exception as exc:
        print(f"FAILED ({exc})")
        return ""


def download_all(delay: float = DELAY) -> str:
    """Download all configured Wikipedia articles; return combined text."""
    total = len(WIKIPEDIA_ARTICLES)
    parts: list[str] = []

    for i, (slug, display_name) in enumerate(WIKIPEDIA_ARTICLES, 1):
        text = _download_article(slug, display_name, idx=i, total=total)
        if text:
            parts.append(text)
        time.sleep(delay)

    total_chars = sum(len(p) for p in parts)
    print(f"\n  ✓ Downloaded {len(parts)}/{total} Wikipedia articles  "
          f"({total_chars / 1_000_000:.1f} MB)")
    return "\n\n".join(parts)
