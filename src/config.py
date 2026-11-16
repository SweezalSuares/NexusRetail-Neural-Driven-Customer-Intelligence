
"""Central configuration: paths, time-split design and theme colours."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ecommerce_sales_34500.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
OUT_DIR = ROOT / "outputs"

SEED = 42

# ---- Customer-level (snapshot) design -------------------------------------
HORIZON_DAYS = 180
TRAIN_SNAPSHOTS = ["2024-03-14", "2024-06-13", "2024-09-12"]
TEST_SNAPSHOT = "2025-03-13"

# ---- Order-level design -----------------------------------------------------
NEXT_PURCHASE_CAP_DAYS = 120
ORDER_TRAIN_END = "2024-09-12"
ORDER_TEST_START = "2025-01-11"
RETURN_TRAIN_END = "2025-01-10"
PRIOR_RETURN_RATE = 0.055

CATEGORIES = [
    "Beauty", "Electronics", "Fashion", "Grocery",
    "Home", "Sports", "Toys"
]

PAYMENTS = [
    "COD", "Credit Card", "Debit Card",
    "PayPal", "UPI", "Wallet"
]

REGIONS = ["Central", "East", "North", "South", "West"]
GENDERS = ["Female", "Male", "Other"]


# ---- Theme: soft ivory, cocoa and rose ------------------------------------
CREAMY_BACKGROUND = "#FBF7EF"
PANEL_BACKGROUND = "#F3E6DA"
BORDER = "#E3D0C2"
CARD_BACKGROUND = "#FFFDF9"
COCOA_BROWN = "#3B2A24"
SOFT_COCOA = "#7A6458"
CHAMPAGNE_GOLD = "#C9A66B"
GOLD_SOFT = "#F1E5C8"
MUTED_ROSE = "#B76E79"
DEEP_ROSE = "#9C5560"
ROSE_SOFT = "#F3DEE0"
SAGE = "#5F7F68"
SAGE_SOFT = "#E3ECE4"
SIDEBAR_USER = "#F3E9DC"
SIDEBAR_SECONDARY = "#CDBBA9"
SIDEBAR_LINK = "#E9D8B4"
HERO_TEXT = "#E8DACB"
TEXT = "#3B2A24"

# Legacy aliases keep existing charts and cards on the updated palette.
DUSKY_BACKGROUND = CREAMY_BACKGROUND
SOFT_IVORY = CARD_BACKGROUND
BLUSH_BEIGE = PANEL_BACKGROUND
CHARCOAL = PANEL_BACKGROUND
CHARCOAL_LIGHT = CARD_BACKGROUND
CHARCOAL_MID = BORDER
TEAL = SAGE
TEAL_DARK = DEEP_ROSE
OLIVE = SAGE
MUTED = SOFT_COCOA
MINT = CHAMPAGNE_GOLD
MINT_LIGHT = GOLD_SOFT
MINT_DARK = SAGE
WARN = CHAMPAGNE_GOLD
DANGER = DEEP_ROSE

PALETTE = [
    DEEP_ROSE, CHAMPAGNE_GOLD, SAGE, COCOA_BROWN,
    MUTED_ROSE, "#B99579", "#D7C2AC",
]
