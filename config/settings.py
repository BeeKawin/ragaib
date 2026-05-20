"""
config/settings.py
──────────────────
Central configuration: paths, model settings, and the full
Thailand High-School (ม.4 ม.6) curriculum map to OpenStax URLs.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv()

# ── Project Paths ─────────────────────────────────────────────────────────────
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

for _d in [RAW_DIR, PROCESSED_DIR, VECTOR_DB_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

# ── LLM / Embedding ───────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")   # "gemini" | "ollama"
LLM_MODEL    = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))

# Multilingual model — handles Thai + English mixed queries
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_DIMENSION = 384

# ── Vector Store ──────────────────────────────────────────────────────────────
VECTOR_STORE_TYPE  = "chroma"
CHROMA_COLLECTION  = "openstax_thailand_hs"
TOP_K_RETRIEVAL    = 6
SIMILARITY_THRESHOLD = 0.35

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800    # characters
CHUNK_OVERLAP = 150

# ── Scraper ───────────────────────────────────────────────────────────────────
SCRAPE_DELAY_SECONDS = 2.0
SCRAPE_MAX_RETRIES   = 3
SCRAPE_TIMEOUT       = 30
SCRAPE_USER_AGENT    = (
    "Mozilla/5.0 (compatible; EduBot-TH/1.0; +https://example.com/bot)"
)

# ─────────────────────────────────────────────────────────────────────────────
# CK-12 Curriculum Map  →  Thailand High-School ม.4–ม.6
# ─────────────────────────────────────────────────────────────────────────────
CK12_BASE = "https://www.ck12.org"

CURRICULUM: List[Dict] = [

    # ══════════════════════════════════════════════════════════════════════════
    # MATHEMATICS  (คณิตศาสตร์)
    # ══════════════════════════════════════════════════════════════════════════

    # ม.4
    {"subject": "math", "grade": "M4", "topic": "Sets",
     "ck12_url": f"{CK12_BASE}/arithmetic/sets/"},
    {"subject": "math", "grade": "M4", "topic": "Logic and Reasoning",
     "ck12_url": f"{CK12_BASE}/algebra/logic/"},
    {"subject": "math", "grade": "M4", "topic": "Real Number System",
     "ck12_url": f"{CK12_BASE}/arithmetic/real-numbers/"},
    {"subject": "math", "grade": "M4", "topic": "Relations and Functions",
     "ck12_url": f"{CK12_BASE}/algebra/functions/"},
    {"subject": "math", "grade": "M4", "topic": "Linear Equations and Inequalities",
     "ck12_url": f"{CK12_BASE}/algebra/linear-equations/"},
    {"subject": "math", "grade": "M4", "topic": "Quadratic Equations",
     "ck12_url": f"{CK12_BASE}/algebra/quadratic-equations/"},
    {"subject": "math", "grade": "M4", "topic": "Exponential Functions",
     "ck12_url": f"{CK12_BASE}/algebra/exponential-functions/"},
    {"subject": "math", "grade": "M4", "topic": "Logarithms",
     "ck12_url": f"{CK12_BASE}/algebra/logarithms/"},

    # ม.5
    {"subject": "math", "grade": "M5", "topic": "Trigonometric Functions",
     "ck12_url": f"{CK12_BASE}/trigonometry/trigonometric-functions/"},
    {"subject": "math", "grade": "M5", "topic": "Trigonometric Identities",
     "ck12_url": f"{CK12_BASE}/trigonometry/trigonometric-identities/"},
    {"subject": "math", "grade": "M5", "topic": "Arithmetic Sequences",
     "ck12_url": f"{CK12_BASE}/algebra/arithmetic-sequences/"},
    {"subject": "math", "grade": "M5", "topic": "Geometric Sequences",
     "ck12_url": f"{CK12_BASE}/algebra/geometric-sequences/"},
    {"subject": "math", "grade": "M5", "topic": "Statistics and Probability",
     "ck12_url": f"{CK12_BASE}/statistics/introduction-to-statistics/"},
    {"subject": "math", "grade": "M5", "topic": "Permutations and Combinations",
     "ck12_url": f"{CK12_BASE}/algebra/permutations-and-combinations/"},
    {"subject": "math", "grade": "M5", "topic": "Conic Sections",
     "ck12_url": f"{CK12_BASE}/algebra/conic-sections/"},

    # ม.6
    {"subject": "math", "grade": "M6", "topic": "Matrices",
     "ck12_url": f"{CK12_BASE}/algebra/matrices/"},
    {"subject": "math", "grade": "M6", "topic": "Vectors",
     "ck12_url": f"{CK12_BASE}/algebra/vectors/"},
    {"subject": "math", "grade": "M6", "topic": "Limits",
     "ck12_url": f"{CK12_BASE}/calculus/limits/"},
    {"subject": "math", "grade": "M6", "topic": "Derivatives",
     "ck12_url": f"{CK12_BASE}/calculus/derivatives/"},
    {"subject": "math", "grade": "M6", "topic": "Integrals",
     "ck12_url": f"{CK12_BASE}/calculus/integrals/"},

    # ══════════════════════════════════════════════════════════════════════════
    # CHEMISTRY  (เคมี)
    # ══════════════════════════════════════════════════════════════════════════

    # ม.4
    {"subject": "chemistry", "grade": "M4", "topic": "Atomic Structure",
     "ck12_url": f"{CK12_BASE}/chemistry/atomic-structure/"},
    {"subject": "chemistry", "grade": "M4", "topic": "Electron Configuration",
     "ck12_url": f"{CK12_BASE}/chemistry/electron-configuration/"},
    {"subject": "chemistry", "grade": "M4", "topic": "Periodic Table and Trends",
     "ck12_url": f"{CK12_BASE}/chemistry/periodic-table/"},
    {"subject": "chemistry", "grade": "M4", "topic": "Chemical Bonding",
     "ck12_url": f"{CK12_BASE}/chemistry/chemical-bonding/"},
    {"subject": "chemistry", "grade": "M4", "topic": "Ionic and Covalent Bonds",
     "ck12_url": f"{CK12_BASE}/chemistry/ionic-and-covalent-bonds/"},
    {"subject": "chemistry", "grade": "M4", "topic": "Intermolecular Forces",
     "ck12_url": f"{CK12_BASE}/chemistry/intermolecular-forces/"},
    {"subject": "chemistry", "grade": "M4", "topic": "Stoichiometry",
     "ck12_url": f"{CK12_BASE}/chemistry/stoichiometry/"},

    # ม.5
    {"subject": "chemistry", "grade": "M5", "topic": "Gas Laws",
     "ck12_url": f"{CK12_BASE}/chemistry/gas-laws/"},
    {"subject": "chemistry", "grade": "M5", "topic": "Solutions and Concentration",
     "ck12_url": f"{CK12_BASE}/chemistry/solutions/"},
    {"subject": "chemistry", "grade": "M5", "topic": "Acids and Bases",
     "ck12_url": f"{CK12_BASE}/chemistry/acids-and-bases/"},
    {"subject": "chemistry", "grade": "M5", "topic": "pH and Buffers",
     "ck12_url": f"{CK12_BASE}/chemistry/ph-and-buffers/"},
    {"subject": "chemistry", "grade": "M5", "topic": "Electrochemistry",
     "ck12_url": f"{CK12_BASE}/chemistry/electrochemistry/"},
    {"subject": "chemistry", "grade": "M5", "topic": "Oxidation and Reduction",
     "ck12_url": f"{CK12_BASE}/chemistry/oxidation-reduction-reactions/"},

    # ม.6
    {"subject": "chemistry", "grade": "M6", "topic": "Chemical Kinetics",
     "ck12_url": f"{CK12_BASE}/chemistry/reaction-kinetics/"},
    {"subject": "chemistry", "grade": "M6", "topic": "Chemical Equilibrium",
     "ck12_url": f"{CK12_BASE}/chemistry/chemical-equilibrium/"},
    {"subject": "chemistry", "grade": "M6", "topic": "Thermochemistry",
     "ck12_url": f"{CK12_BASE}/chemistry/thermochemistry/"},
    {"subject": "chemistry", "grade": "M6", "topic": "Organic Chemistry",
     "ck12_url": f"{CK12_BASE}/chemistry/organic-chemistry/"},
    {"subject": "chemistry", "grade": "M6", "topic": "Hydrocarbons",
     "ck12_url": f"{CK12_BASE}/chemistry/hydrocarbons/"},
    {"subject": "chemistry", "grade": "M6", "topic": "Functional Groups",
     "ck12_url": f"{CK12_BASE}/chemistry/functional-groups/"},
    {"subject": "chemistry", "grade": "M6", "topic": "Polymers",
     "ck12_url": f"{CK12_BASE}/chemistry/polymers/"},

    # ══════════════════════════════════════════════════════════════════════════
    # PHYSICS  (ฟิสิกส์)
    # ══════════════════════════════════════════════════════════════════════════

    # ม.4
    {"subject": "physics", "grade": "M4", "topic": "Measurement and Units",
     "ck12_url": f"{CK12_BASE}/physics/measurement/"},
    {"subject": "physics", "grade": "M4", "topic": "Kinematics 1D",
     "ck12_url": f"{CK12_BASE}/physics/kinematics/"},
    {"subject": "physics", "grade": "M4", "topic": "Kinematics 2D",
     "ck12_url": f"{CK12_BASE}/physics/two-dimensional-motion/"},
    {"subject": "physics", "grade": "M4", "topic": "Newton's Laws of Motion",
     "ck12_url": f"{CK12_BASE}/physics/newtons-laws-of-motion/"},
    {"subject": "physics", "grade": "M4", "topic": "Circular Motion",
     "ck12_url": f"{CK12_BASE}/physics/circular-motion/"},
    {"subject": "physics", "grade": "M4", "topic": "Work Energy and Power",
     "ck12_url": f"{CK12_BASE}/physics/work-energy-and-power/"},
    {"subject": "physics", "grade": "M4", "topic": "Momentum and Collisions",
     "ck12_url": f"{CK12_BASE}/physics/momentum/"},

    # ม.5
    {"subject": "physics", "grade": "M5", "topic": "Rotational Motion",
     "ck12_url": f"{CK12_BASE}/physics/rotational-motion/"},
    {"subject": "physics", "grade": "M5", "topic": "Gravitation",
     "ck12_url": f"{CK12_BASE}/physics/gravity/"},
    {"subject": "physics", "grade": "M5", "topic": "Simple Harmonic Motion",
     "ck12_url": f"{CK12_BASE}/physics/simple-harmonic-motion/"},
    {"subject": "physics", "grade": "M5", "topic": "Waves and Sound",
     "ck12_url": f"{CK12_BASE}/physics/waves-sound-and-light/"},
    {"subject": "physics", "grade": "M5", "topic": "Thermodynamics",
     "ck12_url": f"{CK12_BASE}/physics/thermodynamics/"},
    {"subject": "physics", "grade": "M5", "topic": "Heat Transfer",
     "ck12_url": f"{CK12_BASE}/physics/heat/"},

    # ม.6
    {"subject": "physics", "grade": "M6", "topic": "Electrostatics",
     "ck12_url": f"{CK12_BASE}/physics/electrostatics/"},
    {"subject": "physics", "grade": "M6", "topic": "Electric Current and Circuits",
     "ck12_url": f"{CK12_BASE}/physics/electric-current/"},
    {"subject": "physics", "grade": "M6", "topic": "Magnetism",
     "ck12_url": f"{CK12_BASE}/physics/magnetism/"},
    {"subject": "physics", "grade": "M6", "topic": "Electromagnetic Induction",
     "ck12_url": f"{CK12_BASE}/physics/electromagnetic-induction/"},
    {"subject": "physics", "grade": "M6", "topic": "Light and Optics",
     "ck12_url": f"{CK12_BASE}/physics/light/"},
    {"subject": "physics", "grade": "M6", "topic": "Quantum Physics",
     "ck12_url": f"{CK12_BASE}/physics/quantum-physics/"},
    {"subject": "physics", "grade": "M6", "topic": "Nuclear Physics",
     "ck12_url": f"{CK12_BASE}/physics/nuclear-physics/"},

    # ══════════════════════════════════════════════════════════════════════════
    # BIOLOGY  (ชีววิทยา)
    # ══════════════════════════════════════════════════════════════════════════

    # ม.4
    {"subject": "biology", "grade": "M4", "topic": "Cell Biology",
     "ck12_url": f"{CK12_BASE}/biology/cell-biology/"},
    {"subject": "biology", "grade": "M4", "topic": "Cell Structure and Function",
     "ck12_url": f"{CK12_BASE}/biology/cell-structure/"},
    {"subject": "biology", "grade": "M4", "topic": "Mitosis and Meiosis",
     "ck12_url": f"{CK12_BASE}/biology/cell-division/"},
    {"subject": "biology", "grade": "M4", "topic": "Biochemistry and Biomolecules",
     "ck12_url": f"{CK12_BASE}/biology/biochemistry/"},
    {"subject": "biology", "grade": "M4", "topic": "Enzymes",
     "ck12_url": f"{CK12_BASE}/biology/enzymes/"},
    {"subject": "biology", "grade": "M4", "topic": "Photosynthesis",
     "ck12_url": f"{CK12_BASE}/biology/photosynthesis/"},
    {"subject": "biology", "grade": "M4", "topic": "Cellular Respiration",
     "ck12_url": f"{CK12_BASE}/biology/cellular-respiration/"},

    # ม.5
    {"subject": "biology", "grade": "M5", "topic": "Mendelian Genetics",
     "ck12_url": f"{CK12_BASE}/biology/mendelian-genetics/"},
    {"subject": "biology", "grade": "M5", "topic": "DNA Structure and Replication",
     "ck12_url": f"{CK12_BASE}/biology/dna-structure/"},
    {"subject": "biology", "grade": "M5", "topic": "Gene Expression",
     "ck12_url": f"{CK12_BASE}/biology/gene-expression/"},
    {"subject": "biology", "grade": "M5", "topic": "Mutations and Genetic Disorders",
     "ck12_url": f"{CK12_BASE}/biology/mutations/"},
    {"subject": "biology", "grade": "M5", "topic": "Evolution and Natural Selection",
     "ck12_url": f"{CK12_BASE}/biology/evolution/"},

    # ม.6
    {"subject": "biology", "grade": "M6", "topic": "Plant Biology",
     "ck12_url": f"{CK12_BASE}/biology/plant-biology/"},
    {"subject": "biology", "grade": "M6", "topic": "Human Body Systems",
     "ck12_url": f"{CK12_BASE}/biology/human-biology/"},
    {"subject": "biology", "grade": "M6", "topic": "Nervous and Endocrine Systems",
     "ck12_url": f"{CK12_BASE}/biology/nervous-system/"},
    {"subject": "biology", "grade": "M6", "topic": "Immune System",
     "ck12_url": f"{CK12_BASE}/biology/immune-system/"},
    {"subject": "biology", "grade": "M6", "topic": "Ecology and Ecosystems",
     "ck12_url": f"{CK12_BASE}/biology/ecology/"},
    {"subject": "biology", "grade": "M6", "topic": "Biotechnology",
     "ck12_url": f"{CK12_BASE}/biology/biotechnology/"},
]

OPENSTAX_BASE = "https://openstax.org/books"

OPENSTAX_SOURCE_MAP: Dict[tuple[str, str, str], str] = {
    # Mathematics
    ("math", "M4", "Sets"): f"{OPENSTAX_BASE}/contemporary-mathematics/pages/1-4-set-operations-with-two-sets",
    ("math", "M4", "Logic and Reasoning"): f"{OPENSTAX_BASE}/contemporary-mathematics/pages/2-2-compound-statements",
    ("math", "M4", "Real Number System"): f"{OPENSTAX_BASE}/algebra-and-trigonometry-2e/pages/1-1-real-numbers-algebra-essentials",
    ("math", "M4", "Relations and Functions"): f"{OPENSTAX_BASE}/algebra-and-trigonometry-2e/pages/3-1-functions-and-function-notation",
    ("math", "M4", "Linear Equations and Inequalities"): f"{OPENSTAX_BASE}/algebra-and-trigonometry-2e/pages/2-2-linear-equations-in-one-variable",
    ("math", "M4", "Quadratic Equations"): f"{OPENSTAX_BASE}/algebra-and-trigonometry-2e/pages/2-5-quadratic-equations",
    ("math", "M4", "Exponential Functions"): f"{OPENSTAX_BASE}/algebra-and-trigonometry-2e/pages/6-1-exponential-functions",
    ("math", "M4", "Logarithms"): f"{OPENSTAX_BASE}/algebra-and-trigonometry-2e/pages/6-3-logarithmic-functions",
    ("math", "M5", "Trigonometric Functions"): f"{OPENSTAX_BASE}/algebra-and-trigonometry-2e/pages/7-2-right-triangle-trigonometry",
    ("math", "M5", "Trigonometric Identities"): f"{OPENSTAX_BASE}/algebra-and-trigonometry-2e/pages/9-1-verifying-trigonometric-identities-and-using-trigonometric-identities-to-simplify-trigonometric-expressions",
    ("math", "M5", "Arithmetic Sequences"): f"{OPENSTAX_BASE}/algebra-and-trigonometry-2e/pages/13-2-arithmetic-sequences",
    ("math", "M5", "Geometric Sequences"): f"{OPENSTAX_BASE}/algebra-and-trigonometry-2e/pages/13-3-geometric-sequences",
    ("math", "M5", "Statistics and Probability"): f"{OPENSTAX_BASE}/introductory-statistics-2e/pages/3-introduction",
    ("math", "M5", "Permutations and Combinations"): f"{OPENSTAX_BASE}/precalculus/pages/11-5-counting-principles",
    ("math", "M5", "Conic Sections"): f"{OPENSTAX_BASE}/algebra-and-trigonometry-2e/pages/12-introduction-to-analytic-geometry",
    ("math", "M6", "Matrices"): f"{OPENSTAX_BASE}/precalculus/pages/9-5-matrices-and-matrix-operations",
    ("math", "M6", "Vectors"): f"{OPENSTAX_BASE}/algebra-and-trigonometry-2e/pages/10-8-vectors",
    ("math", "M6", "Limits"): f"{OPENSTAX_BASE}/calculus-volume-1/pages/2-2-the-limit-of-a-function",
    ("math", "M6", "Derivatives"): f"{OPENSTAX_BASE}/calculus-volume-1/pages/3-1-defining-the-derivative",
    ("math", "M6", "Integrals"): f"{OPENSTAX_BASE}/calculus-volume-1/pages/5-2-the-definite-integral",

    # Chemistry
    ("chemistry", "M4", "Atomic Structure"): f"{OPENSTAX_BASE}/chemistry-2e/pages/2-2-evolution-of-atomic-theory",
    ("chemistry", "M4", "Electron Configuration"): f"{OPENSTAX_BASE}/chemistry-2e/pages/6-4-electronic-structure-of-atoms-electron-configurations",
    ("chemistry", "M4", "Periodic Table and Trends"): f"{OPENSTAX_BASE}/chemistry-2e/pages/6-5-periodic-variations-in-element-properties",
    ("chemistry", "M4", "Chemical Bonding"): f"{OPENSTAX_BASE}/chemistry-2e/pages/7-introduction",
    ("chemistry", "M4", "Ionic and Covalent Bonds"): f"{OPENSTAX_BASE}/chemistry-2e/pages/7-2-covalent-bonding",
    ("chemistry", "M4", "Intermolecular Forces"): f"{OPENSTAX_BASE}/chemistry-2e/pages/10-1-intermolecular-forces",
    ("chemistry", "M4", "Stoichiometry"): f"{OPENSTAX_BASE}/chemistry-2e/pages/4-3-reaction-stoichiometry",
    ("chemistry", "M5", "Gas Laws"): f"{OPENSTAX_BASE}/chemistry-2e/pages/9-2-relating-pressure-volume-amount-and-temperature-the-ideal-gas-law",
    ("chemistry", "M5", "Solutions and Concentration"): f"{OPENSTAX_BASE}/chemistry-2e/pages/11-3-solubility",
    ("chemistry", "M5", "Acids and Bases"): f"{OPENSTAX_BASE}/chemistry-2e/pages/14-2-ph-and-poh",
    ("chemistry", "M5", "pH and Buffers"): f"{OPENSTAX_BASE}/chemistry-2e/pages/14-6-buffers",
    ("chemistry", "M5", "Electrochemistry"): f"{OPENSTAX_BASE}/chemistry-2e/pages/17-introduction",
    ("chemistry", "M5", "Oxidation and Reduction"): f"{OPENSTAX_BASE}/chemistry-2e/pages/17-1-balancing-oxidation-reduction-reactions",
    ("chemistry", "M6", "Chemical Kinetics"): f"{OPENSTAX_BASE}/chemistry-2e/pages/12-introduction",
    ("chemistry", "M6", "Chemical Equilibrium"): f"{OPENSTAX_BASE}/chemistry-2e/pages/13-introduction",
    ("chemistry", "M6", "Thermochemistry"): f"{OPENSTAX_BASE}/chemistry-2e/pages/5-introduction",
    ("chemistry", "M6", "Organic Chemistry"): f"{OPENSTAX_BASE}/organic-chemistry/pages/1-introduction",
    ("chemistry", "M6", "Hydrocarbons"): f"{OPENSTAX_BASE}/organic-chemistry/pages/7-introduction",
    ("chemistry", "M6", "Functional Groups"): f"{OPENSTAX_BASE}/organic-chemistry/pages/3-1-functional-groups",
    ("chemistry", "M6", "Polymers"): f"{OPENSTAX_BASE}/chemistry-2e/pages/10-6-lattice-structures-in-crystalline-solids",

    # Physics
    ("physics", "M4", "Measurement and Units"): f"{OPENSTAX_BASE}/college-physics-2e/pages/1-2-physical-quantities-and-units",
    ("physics", "M4", "Kinematics 1D"): f"{OPENSTAX_BASE}/college-physics-2e/pages/2-introduction-to-one-dimensional-kinematics",
    ("physics", "M4", "Kinematics 2D"): f"{OPENSTAX_BASE}/college-physics-2e/pages/3-introduction-to-two-dimensional-kinematics",
    ("physics", "M4", "Newton's Laws of Motion"): f"{OPENSTAX_BASE}/college-physics-2e/pages/4-3-newtons-second-law-of-motion-concept-of-a-system",
    ("physics", "M4", "Circular Motion"): f"{OPENSTAX_BASE}/college-physics-2e/pages/6-introduction-to-uniform-circular-motion-and-gravitation",
    ("physics", "M4", "Work Energy and Power"): f"{OPENSTAX_BASE}/college-physics-2e/pages/7-introduction-to-work-energy-and-energy-resources",
    ("physics", "M4", "Momentum and Collisions"): f"{OPENSTAX_BASE}/college-physics-2e/pages/8-introduction-to-linear-momentum-and-collisions",
    ("physics", "M5", "Rotational Motion"): f"{OPENSTAX_BASE}/college-physics-2e/pages/10-introduction-to-rotational-motion-and-angular-momentum",
    ("physics", "M5", "Gravitation"): f"{OPENSTAX_BASE}/college-physics-2e/pages/6-5-newtons-universal-law-of-gravitation",
    ("physics", "M5", "Simple Harmonic Motion"): f"{OPENSTAX_BASE}/college-physics-2e/pages/16-3-simple-harmonic-motion-a-special-periodic-motion",
    ("physics", "M5", "Waves and Sound"): f"{OPENSTAX_BASE}/college-physics-2e/pages/16-introduction-to-oscillatory-motion-and-waves",
    ("physics", "M5", "Thermodynamics"): f"{OPENSTAX_BASE}/college-physics-2e/pages/15-introduction-to-thermodynamics",
    ("physics", "M5", "Heat Transfer"): f"{OPENSTAX_BASE}/college-physics-2e/pages/14-introduction-to-heat-and-heat-transfer-methods",
    ("physics", "M6", "Electrostatics"): f"{OPENSTAX_BASE}/college-physics-2e/pages/18-introduction-to-electric-charge-and-electric-field",
    ("physics", "M6", "Electric Current and Circuits"): f"{OPENSTAX_BASE}/college-physics-2e/pages/20-introduction-to-electric-current-resistance-and-ohms-law",
    ("physics", "M6", "Magnetism"): f"{OPENSTAX_BASE}/college-physics-2e/pages/22-introduction-to-magnetism",
    ("physics", "M6", "Electromagnetic Induction"): f"{OPENSTAX_BASE}/college-physics-2e/pages/23-introduction-to-electromagnetic-induction-ac-circuits-and-electrical-technologies",
    ("physics", "M6", "Light and Optics"): f"{OPENSTAX_BASE}/college-physics-2e/pages/25-introduction-to-geometric-optics",
    ("physics", "M6", "Quantum Physics"): f"{OPENSTAX_BASE}/college-physics-2e/pages/29-introduction-to-quantum-physics",
    ("physics", "M6", "Nuclear Physics"): f"{OPENSTAX_BASE}/college-physics-2e/pages/31-introduction-to-radioactivity-and-nuclear-physics",

    # Biology
    ("biology", "M4", "Cell Biology"): f"{OPENSTAX_BASE}/biology-2e/pages/4-introduction",
    ("biology", "M4", "Cell Structure and Function"): f"{OPENSTAX_BASE}/biology-2e/pages/4-4-eukaryotic-cells",
    ("biology", "M4", "Mitosis and Meiosis"): f"{OPENSTAX_BASE}/biology-2e/pages/10-introduction",
    ("biology", "M4", "Biochemistry and Biomolecules"): f"{OPENSTAX_BASE}/biology-2e/pages/3-introduction",
    ("biology", "M4", "Enzymes"): f"{OPENSTAX_BASE}/biology-2e/pages/6-5-enzymes",
    ("biology", "M4", "Photosynthesis"): f"{OPENSTAX_BASE}/biology-2e/pages/8-introduction",
    ("biology", "M4", "Cellular Respiration"): f"{OPENSTAX_BASE}/biology-2e/pages/7-introduction",
    ("biology", "M5", "Mendelian Genetics"): f"{OPENSTAX_BASE}/biology-2e/pages/12-introduction",
    ("biology", "M5", "DNA Structure and Replication"): f"{OPENSTAX_BASE}/biology-2e/pages/14-3-basics-of-dna-replication",
    ("biology", "M5", "Gene Expression"): f"{OPENSTAX_BASE}/biology-2e/pages/16-introduction",
    ("biology", "M5", "Mutations and Genetic Disorders"): f"{OPENSTAX_BASE}/biology-2e/pages/17-5-mutations",
    ("biology", "M5", "Evolution and Natural Selection"): f"{OPENSTAX_BASE}/biology-2e/pages/18-introduction",
    ("biology", "M6", "Plant Biology"): f"{OPENSTAX_BASE}/biology-2e/pages/30-introduction",
    ("biology", "M6", "Human Body Systems"): f"{OPENSTAX_BASE}/biology-2e/pages/33-introduction",
    ("biology", "M6", "Nervous and Endocrine Systems"): f"{OPENSTAX_BASE}/biology-2e/pages/35-introduction",
    ("biology", "M6", "Immune System"): f"{OPENSTAX_BASE}/biology-2e/pages/42-introduction",
    ("biology", "M6", "Ecology and Ecosystems"): f"{OPENSTAX_BASE}/biology-2e/pages/44-introduction",
    ("biology", "M6", "Biotechnology"): f"{OPENSTAX_BASE}/biology-2e/pages/17-introduction",
}

for _entry in CURRICULUM:
    _entry["source_name"] = "OpenStax"
    _entry["source_url"] = OPENSTAX_SOURCE_MAP.get(
        (_entry["subject"], _entry["grade"], _entry["topic"]),
        "",
    )

# ── Subject & Grade display metadata ─────────────────────────────────────────
SUBJECT_META = {
    "math":      {"display": "Mathematics", "display_th": "คณิตศาสตร์", "icon": "📐", "color": "#4F46E5"},
    "chemistry": {"display": "Chemistry",   "display_th": "เคมี",        "icon": "⚗️", "color": "#0891B2"},
    "physics":   {"display": "Physics",     "display_th": "ฟิสิกส์",     "icon": "⚡", "color": "#7C3AED"},
    "biology":   {"display": "Biology",     "display_th": "ชีววิทยา",    "icon": "🧬", "color": "#059669"},
}

GRADE_META = {
    "M4": {"display": "Mathayom 4 (Grade 10)", "display_th": "ม.4"},
    "M5": {"display": "Mathayom 5 (Grade 11)", "display_th": "ม.5"},
    "M6": {"display": "Mathayom 6 (Grade 12)", "display_th": "ม.6"},
}
