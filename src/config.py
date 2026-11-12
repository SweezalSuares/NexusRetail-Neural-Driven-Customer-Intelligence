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
# For every snapshot date we build features from orders <= snapshot and
# targets from the following HORIZON_DAYS. Train snapshots' target windows end
# before the test snapshot, so there is no temporal leakage.
HORIZON_DAYS = 180
TRAIN_SNAPSHOTS = ["2024-03-14", "2024-06-13", "2024-09-12"]
TEST_SNAPSHOT = "2025-03-13"

# ---- Order-level design -----------------------------------------------------
NEXT_PURCHASE_CAP_DAYS = 120          # censoring cap for "days to next purchase"
ORDER_TRAIN_END = "2024-09-12"        # next-product / next-purchase train cutoff
ORDER_TEST_START = "2025-01-11"       # embargo gap = max label window (120d)
RETURN_TRAIN_END = "2025-01-10"       # return label is known at order time
PRIOR_RETURN_RATE = 0.055             # smoothing prior for new customers

CATEGORIES = ["Beauty", "Electronics", "Fashion", "Grocery", "Home", "Sports", "Toys"]
PAYMENTS = ["COD", "Credit Card", "Debit Card", "PayPal", "UPI", "Wallet"]
REGIONS = ["Central", "East", "North", "South", "West"]
GENDERS = ["Female", "Male", "Other"]

# ---- Theme: Mint + Teal + Charcoal -------------------------------------------
MINT = "#3EDBB4"
MINT_LIGHT = "#B8F2E0"
MINT_DARK = "#1FA98A"
TEAL = "#0F8B8D"
TEAL_DARK = "#0B5D5F"
CHARCOAL = "#22272B"
CHARCOAL_LIGHT = "#2E353B"
CHARCOAL_MID = "#3A434A"
TEXT = "#E8F5F1"
MUTED = "#9DB2AB"
WARN = "#F2B84B"
DANGER = "#EF6F6C"
PALETTE = [MINT, TEAL, "#7BC8F6", "#F2B84B", "#B39DDB", "#EF6F6C", "#8FA3AD"]
