"""Data loading and feature engineering (customer snapshots + order-level)."""
import numpy as np
import pandas as pd

from .config import (CATEGORIES, DATA_PATH, GENDERS, HORIZON_DAYS, PAYMENTS,
                     PRIOR_RETURN_RATE, REGIONS)

LOG_COLS = ["total_spend", "avg_order_value", "max_order_value", "avg_price",
            "spend_last_90d", "spend_last_180d", "last_amount", "spend_so_far",
            "avg_shipping_cost", "price", "gross_amount", "prior_avg_spend"]
SIGNED_LOG_COLS = ["total_profit", "avg_profit"]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["order_date"])
    df = df.drop_duplicates("order_id")
    df = df.sort_values(["customer_id", "order_date", "order_id"]).reset_index(drop=True)
    df["is_returned"] = (df["returned"].str.strip().str.lower() == "yes").astype(int)
    df["is_discounted"] = (df["discount"] > 0).astype(int)
    df["gross_amount"] = df["price"] * df["quantity"]
    return df


def to_matrix(frame: pd.DataFrame, cols) -> np.ndarray:
    """Apply log transforms to heavy-tailed columns and return a float32 matrix."""
    X = frame[cols].astype(float).copy()
    for c in cols:
        if c in LOG_COLS:
            X[c] = np.log1p(X[c].clip(lower=0))
        elif c in SIGNED_LOG_COLS:
            X[c] = np.sign(X[c]) * np.log1p(X[c].abs())
    return X.fillna(0).to_numpy(dtype="float32")


def slog1p(y):
    y = np.asarray(y, dtype=float)
    return np.sign(y) * np.log1p(np.abs(y))


def inv_slog1p(z):
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.expm1(np.abs(z))


def _share(h, col, values, prefix):
    ct = pd.crosstab(h["customer_id"], h[col]).reindex(columns=values, fill_value=0)
    ct = ct.div(ct.sum(axis=1), axis=0)
    ct.columns = [f"{prefix}_{v.replace(' ', '_')}" for v in values]
    return ct


# ----------------------------------------------------------------------------
# Customer snapshot features / targets
# ----------------------------------------------------------------------------
def customer_features(df: pd.DataFrame, asof) -> pd.DataFrame:
    asof = pd.Timestamp(asof)
    h = df[df["order_date"] <= asof].copy()
    h["gap"] = h.groupby("customer_id")["order_date"].diff().dt.days
    g = h.groupby("customer_id")
    f = pd.DataFrame({"n_orders": g.size()})
    first, last = g["order_date"].min(), g["order_date"].max()
    f["tenure_days"] = (asof - first).dt.days
    f["recency_days"] = (asof - last).dt.days
    f["active_span_days"] = (last - first).dt.days
    f["orders_per_month"] = f["n_orders"] / (f["tenure_days"].clip(lower=30) / 30)
    f["avg_gap_days"] = g["gap"].mean().fillna(0)
    f["std_gap_days"] = g["gap"].std().fillna(0)
    f["has_repeat"] = (f["n_orders"] > 1).astype(int)
    gap_ref = f["avg_gap_days"].where(f["avg_gap_days"] > 0, 120.0)
    f["recency_ratio"] = (f["recency_days"] / gap_ref).clip(upper=20)

    f["total_spend"] = g["total_amount"].sum()
    f["avg_order_value"] = g["total_amount"].mean()
    f["max_order_value"] = g["total_amount"].max()
    f["avg_price"] = g["price"].mean()
    f["total_profit"] = g["profit_margin"].sum()
    f["avg_profit"] = g["profit_margin"].mean()
    f["profit_rate"] = (f["total_profit"] / f["total_spend"]).clip(-2, 1)
    f["total_items"] = g["quantity"].sum()
    f["avg_items"] = g["quantity"].mean()
    f["avg_discount"] = g["discount"].mean()
    f["discount_share"] = g["is_discounted"].mean()
    f["return_rate"] = g["is_returned"].mean()
    f["n_returns"] = g["is_returned"].sum()
    f["avg_delivery_days"] = g["delivery_time_days"].mean()
    f["avg_shipping_cost"] = g["shipping_cost"].mean()
    f["avg_age"] = g["customer_age"].mean()
    f["n_categories"] = g["category"].nunique()

    for w in (90, 180):
        r = h[h["order_date"] > asof - pd.Timedelta(days=w)].groupby("customer_id")
        f[f"orders_last_{w}d"] = r.size().reindex(f.index).fillna(0)
        f[f"spend_last_{w}d"] = r["total_amount"].sum().reindex(f.index).fillna(0)
    f["recent_order_share"] = f["orders_last_180d"] / f["n_orders"]

    cat = _share(h, "category", CATEGORIES, "cat")
    f = f.join(cat).join(_share(h, "payment_method", PAYMENTS, "pay"))
    f = f.join(_share(h, "region", REGIONS, "reg")).join(_share(h, "customer_gender", GENDERS, "gen"))
    f["top_category_share"] = cat.max(axis=1)

    lastrow = g.tail(1).set_index("customer_id")
    f["last_amount"] = lastrow["total_amount"]
    f["last_discount"] = lastrow["discount"]
    f["last_returned"] = lastrow["is_returned"]
    lc = pd.get_dummies(lastrow["category"]).reindex(columns=CATEGORIES, fill_value=0).astype(float)
    lc.columns = [f"last_cat_{c}" for c in CATEGORIES]
    f = f.join(lc)
    f.index.name = "customer_id"
    return f


def customer_targets(df: pd.DataFrame, asof, index, horizon=HORIZON_DAYS) -> pd.DataFrame:
    asof = pd.Timestamp(asof)
    end = asof + pd.Timedelta(days=horizon)
    fut = df[(df["order_date"] > asof) & (df["order_date"] <= end)]
    g = fut.groupby("customer_id")
    t = pd.DataFrame(index=index)
    t["future_orders"] = g.size().reindex(index).fillna(0)
    t["future_spend"] = g["total_amount"].sum().reindex(index).fillna(0.0)
    t["future_profit"] = g["profit_margin"].sum().reindex(index).fillna(0.0)
    t["churned"] = (t["future_orders"] == 0).astype(int)
    last_future = g["order_date"].max().reindex(index)
    t["active_days_future"] = (last_future - asof).dt.days.fillna(0)
    cat_spend = fut.pivot_table(index="customer_id", columns="category",
                                values="total_amount", aggfunc="sum").reindex(index)
    arr = cat_spend.fillna(-1.0).to_numpy()
    t["fav_category"] = pd.Series(cat_spend.columns.to_numpy()[arr.argmax(axis=1)], index=index).where(
        cat_spend.notna().any(axis=1))
    t["disc_share_future"] = g["is_discounted"].mean().reindex(index)
    t["discount_sensitive"] = (t["disc_share_future"] >= 0.5).astype(float).where(t["future_orders"] > 0)
    t["loyalty_tier"] = np.select([t["future_orders"] == 0, t["future_orders"] == 1], [0, 1], 2)
    return t


def build_customer_dataset(df, snapshot):
    f = customer_features(df, snapshot)
    t = customer_targets(df, snapshot, f.index)
    out = f.join(t)
    out["snapshot"] = pd.Timestamp(snapshot)
    return out.reset_index()


# ----------------------------------------------------------------------------
# Order-level features
# ----------------------------------------------------------------------------
def _cum_share(d, col, values):
    oh = pd.get_dummies(d[col]).reindex(columns=values, fill_value=0).astype(float)
    cum = oh.groupby(d["customer_id"].values).cumsum()
    n = (d.groupby("customer_id").cumcount() + 1).to_numpy()[:, None]
    return cum / n


def _onehot(s, values, prefix):
    oh = pd.get_dummies(s).reindex(columns=values, fill_value=0).astype(float)
    oh.columns = [f"{prefix}_{v.replace(' ', '_')}" for v in values]
    return oh


def order_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """State of the customer *after* each order (used for next-product and
    next-purchase models; every feature only uses orders up to and including i)."""
    d = df
    g = d.groupby("customer_id")
    n = g.cumcount() + 1
    first = g["order_date"].transform("min")
    prev_date = g["order_date"].shift()
    X = pd.DataFrame(index=d.index)
    X["n_orders"] = n
    X["days_since_first"] = (d["order_date"] - first).dt.days
    X["gap_last"] = (d["order_date"] - prev_date).dt.days.fillna(0)
    X["avg_gap"] = np.where(n > 1, X["days_since_first"] / (n - 1).clip(lower=1), 0.0)
    cum_spend = g["total_amount"].cumsum()
    X["spend_so_far"] = cum_spend
    X["avg_order_value"] = cum_spend / n
    X["profit_rate"] = (g["profit_margin"].cumsum() / cum_spend).clip(-2, 1)
    X["return_rate"] = g["is_returned"].cumsum() / n
    X["avg_discount"] = g["discount"].cumsum() / n
    X["discount_share"] = g["is_discounted"].cumsum() / n
    X["last_amount"] = d["total_amount"]
    X["last_qty"] = d["quantity"]
    X["last_discount"] = d["discount"]
    X["customer_age"] = d["customer_age"]
    X["avg_delivery_days"] = g["delivery_time_days"].cumsum() / n
    cs = _cum_share(d, "category", CATEGORIES)
    cs.columns = [f"cat_{c}" for c in CATEGORIES]
    X = X.join(cs)
    X = X.join(_onehot(d["category"], CATEGORIES, "last_cat"))
    prev_cat = g["category"].shift()
    X = X.join(_onehot(prev_cat, CATEGORIES, "prev_cat"))
    m = d["order_date"].dt.month
    X["month_sin"] = np.sin(2 * np.pi * m / 12)
    X["month_cos"] = np.cos(2 * np.pi * m / 12)
    X["dow"] = d["order_date"].dt.dayofweek
    return X


def return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features known when an order is placed: the order itself + the customer's
    *prior* history (strictly before this order)."""
    d = df
    g = d.groupby("customer_id")
    n_prior = g.cumcount()
    X = pd.DataFrame(index=d.index)
    X["price"] = d["price"]
    X["gross_amount"] = d["gross_amount"]
    X["quantity"] = d["quantity"]
    X["discount"] = d["discount"]
    X["delivery_time_days"] = d["delivery_time_days"]
    X["shipping_cost"] = d["shipping_cost"]
    X["shipping_ratio"] = (d["shipping_cost"] / d["total_amount"].clip(lower=1)).clip(upper=5)
    X["customer_age"] = d["customer_age"]
    X["month_sin"] = np.sin(2 * np.pi * d["order_date"].dt.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * d["order_date"].dt.month / 12)
    X = X.join(_onehot(d["category"], CATEGORIES, "cat"))
    X = X.join(_onehot(d["payment_method"], PAYMENTS, "pay"))
    X = X.join(_onehot(d["region"], REGIONS, "reg"))
    X = X.join(_onehot(d["customer_gender"], GENDERS, "gen"))
    k = 3.0
    prior_ret = g["is_returned"].cumsum() - d["is_returned"]
    X["n_prior_orders"] = n_prior
    X["prior_return_rate"] = (prior_ret + k * PRIOR_RETURN_RATE) / (n_prior + k)
    denom = n_prior.where(n_prior > 0)
    X["prior_avg_discount"] = ((g["discount"].cumsum() - d["discount"]) / denom).fillna(0)
    X["prior_avg_spend"] = ((g["total_amount"].cumsum() - d["total_amount"]) / denom).fillna(0)
    prev = g["order_date"].shift()
    X["has_prev"] = prev.notna().astype(int)
    X["days_since_prev"] = (d["order_date"] - prev).dt.days.fillna(0)
    return X
