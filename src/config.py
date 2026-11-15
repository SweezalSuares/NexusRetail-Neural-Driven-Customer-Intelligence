
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


# ---- Theme: Dusky Botanical -----------------------------------------------
# Primary: Muted Teal
# Secondary: Olive
# Accent: Champagne Gold
# Background: Dusky Charcoal
# Cards: Botanical Charcoal
# Text: Soft Ivory

# Main backgrounds
DUSKY_BACKGROUND = "#252A29"
CHARCOAL = "#202523"
CHARCOAL_LIGHT = "#303A37"
CHARCOAL_MID = "#414B46"

# Brand colours
TEAL = "#4F8F83"
TEAL_DARK = "#386F65"

# Accent colours
OLIVE = "#A3A66A"
CHAMPAGNE_GOLD = "#D9B98C"
COCOA_BROWN = "#80644F"
BLUSH_BEIGE = "#B9A99A"
MUTED_ROSE = "#B58F89"

# Text colours
SOFT_IVORY = "#F4EFE5"
TEXT = "#FAF8F2"
MUTED = "#B9B5A9"

# Status colours
MINT = "#82B6A1"
MINT_LIGHT = "#A9C8B8"
MINT_DARK = "#4F8F83"

WARN = "#D9B98C"
DANGER = "#C98282"

# Chart colours
PALETTE = [
    "#4F8F83",  # Muted Teal
    "#A3A66A",  # Olive
    "#D9B98C",  # Champagne Gold
    "#B58F89",  # Muted Rose
    "#80644F",  # Cocoa Brown
    "#A9C8B8",  # Soft Sage
    "#E5D8C5",  # Warm Cream
]
