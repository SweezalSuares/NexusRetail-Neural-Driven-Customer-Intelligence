"""End-to-end ML pipeline: build datasets -> train 11 MLPs -> evaluate -> score all customers."""
import json
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import (build_customer_dataset, customer_features, load_data,
                       order_state_features, return_features, to_matrix)
from .models import predict, train_and_evaluate
from .tasks import LOYALTY_LABELS, TASKS


# Small, regularised MLPs: the data is noisy, so large nets only memorise noise.
HYPER = {"default": ((64, 32), 0.4), "return_risk": ((16, 8), 0.3)}


def signal_level(m):
    """Plain-language verdict on whether the MLP beats chance / the simple baseline."""
    if m["task"] == "binary":
        a, (lo, hi) = m["roc_auc"], m["roc_auc_ci"]
        lvl = "None" if lo <= 0.5 else "Useful" if a >= 0.65 else "Weak"
        return lvl, f"ROC-AUC {a:.3f} (95% CI {lo:.2f}-{hi:.2f}; 0.50 = coin flip)"
    if m["task"] == "multiclass":
        best = max(m["majority_accuracy"], m["baseline_accuracy"])
        lift = m["accuracy"] - best
        lvl = "Useful" if lift >= 0.05 else "Weak" if lift >= 0.02 else "None"
        return lvl, f"accuracy {m['accuracy']:.1%} vs {best:.1%} best naive/baseline"
    r = m["r2_log"]
    lvl = "Useful" if r >= 0.15 else "Weak" if r >= 0.05 else "None"
    return lvl, f"R² on log scale {r:.3f} (0 = predict the average)"


def _customer_frames(df):
    train = pd.concat([build_customer_dataset(df, s) for s in C.TRAIN_SNAPSHOTS], ignore_index=True)
    test = build_customer_dataset(df, C.TEST_SNAPSHOT)
    feat_cols = [c for c in customer_features(df, C.TEST_SNAPSHOT).columns]
    return train, test, feat_cols


def _order_frames(df):
    S, R = order_state_features(df), return_features(df)
    g = df.groupby("customer_id")
    next_date, next_cat = g["order_date"].shift(-1), g["category"].shift(-1)
    gap_next = (next_date - df["order_date"]).dt.days
    end = df["order_date"].max()
    cap = C.NEXT_PURCHASE_CAP_DAYS
    cat_idx = {c: i for i, c in enumerate(C.CATEGORIES)}
    od = df["order_date"]

    m_ret_tr, m_ret_te = od <= C.RETURN_TRAIN_END, od > C.RETURN_TRAIN_END
    test_start = pd.Timestamp(C.ORDER_TEST_START)
    m_nx_tr = (od <= C.ORDER_TRAIN_END) & next_date.notna() & (next_date < test_start)
    m_nx_te = (od >= test_start) & next_date.notna()
    elig = od <= end - pd.Timedelta(days=cap)
    y_days = gap_next.fillna(cap).clip(upper=cap)
    m_np_tr, m_np_te = (od <= C.ORDER_TRAIN_END) & elig, (od >= test_start) & elig
    return dict(S=S, R=R, next_cat=next_cat.map(cat_idx), y_days=y_days,
                masks=dict(ret=(m_ret_tr, m_ret_te), nx=(m_nx_tr, m_nx_te), np=(m_np_tr, m_np_te)))


def train_all(df=None, verbose=True, seed=C.SEED):
    df = load_data() if df is None else df
    C.MODELS_DIR.mkdir(exist_ok=True, parents=True)
    C.REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    tr, te, fcols = _customer_frames(df)
    of = _order_frames(df)
    groups_c = tr["customer_id"].values
    all_metrics, all_curves = {}, {}

    cust_specs = {
        "churn": (None, "churned", None),
        "revenue_loss": (lambda d: d["churned"] == 0, "future_spend", None),
        "spending": (None, "future_spend", None),
        "profit": (None, "future_profit", None),
        "fav_category": (lambda d: d["future_orders"] > 0, "fav_category", C.CATEGORIES),
        "active_duration": (None, "active_days_future", None),
        "discount": (lambda d: d["future_orders"] > 0, "discount_sensitive", None),
        "loyalty": (None, "loyalty_tier", LOYALTY_LABELS),
    }

    def run(task, key, Xtr, ytr, Xte, yte, groups, cols, classes=None):
        t0 = time.time()
        hidden, dropout = HYPER.get(key, HYPER["default"])
        model, bundle, metrics, curves = train_and_evaluate(
            key, task["type"], Xtr, ytr, Xte, yte, groups, to_matrix, cols, classes, hidden, dropout,
            seed=seed)
        metrics["signal"], metrics["signal_note"] = signal_level(metrics)
        model.save(C.MODELS_DIR / f"{key}.keras")
        joblib.dump(bundle, C.MODELS_DIR / f"{key}_bundle.joblib")
        metrics.update(id=task["id"], title=task["title"], level=task["level"], seconds=round(time.time() - t0, 1))
        all_metrics[key], all_curves[key] = metrics, curves
        if verbose:
            show = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}
            print(f"[{task['id']:>2}] {task['title']:<38} n_train={metrics['n_train']:<6} {show}", flush=True)

    for task in TASKS:
        key = task["key"]
        if key in cust_specs:
            sel, ycol, classes = cust_specs[key]
            a = tr if sel is None else tr[sel(tr)]
            b = te if sel is None else te[sel(te)]
            ya, yb = a[ycol], b[ycol]
            if key == "fav_category":
                idx = {c: i for i, c in enumerate(C.CATEGORIES)}
                ya, yb = ya.map(idx), yb.map(idx)
            run(task, key, a, ya.values, b, yb.values, a["customer_id"].values, fcols, classes)
        elif key == "return_risk":
            m_tr, m_te = of["masks"]["ret"]
            R, y = of["R"], df["is_returned"]
            run(task, key, R[m_tr], y[m_tr].values, R[m_te], y[m_te].values,
                df.loc[m_tr, "customer_id"].values, list(R.columns))
        elif key == "next_product":
            m_tr, m_te = of["masks"]["nx"]
            S, y = of["S"], of["next_cat"]
            run(task, key, S[m_tr], y[m_tr].astype(int).values, S[m_te], y[m_te].astype(int).values,
                df.loc[m_tr, "customer_id"].values, list(S.columns), C.CATEGORIES)
        elif key == "next_purchase":
            m_tr, m_te = of["masks"]["np"]
            S, y = of["S"], of["y_days"]
            run(task, key, S[m_tr], y[m_tr].values, S[m_te], y[m_te].values,
                df.loc[m_tr, "customer_id"].values, list(S.columns))

    meta = dict(end_date=str(df["order_date"].max().date()), start_date=str(df["order_date"].min().date()),
                horizon_days=C.HORIZON_DAYS, train_snapshots=C.TRAIN_SNAPSHOTS, test_snapshot=C.TEST_SNAPSHOT,
                next_purchase_cap=C.NEXT_PURCHASE_CAP_DAYS, n_orders=int(len(df)),
                n_customers=int(df["customer_id"].nunique()), seed=int(seed))
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(dict(meta=meta, metrics=all_metrics), indent=2))
    (C.REPORTS_DIR / "curves.json").write_text(json.dumps(all_curves))
    return all_metrics


def load_models():
    import keras
    out = {}
    for t in TASKS:
        k = t["key"]
        out[k] = (keras.models.load_model(C.MODELS_DIR / f"{k}.keras"), joblib.load(C.MODELS_DIR / f"{k}_bundle.joblib"))
    return out


def _run(models, key, frame):
    m, b = models[key]
    return predict(m, b, to_matrix(frame, b["features"]))


def score_customers(df, models):
    """Score every customer as of the last date in the data (forecast = next 180 days)."""
    end = df["order_date"].max()
    f = customer_features(df, end)
    o = pd.DataFrame(index=f.index)
    last_date = df.groupby("customer_id")["order_date"].max()
    o["last_order_date"] = last_date.reindex(o.index)
    for c in ["n_orders", "tenure_days", "recency_days", "total_spend", "total_profit", "avg_order_value", "return_rate"]:
        o[c] = f[c]

    o["churn_prob"] = _run(models, "churn", f)
    o["spend_if_retained"] = _run(models, "revenue_loss", f).clip(min=0)
    o["revenue_at_risk"] = o["churn_prob"] * o["spend_if_retained"]
    o["predicted_spend_180d"] = _run(models, "spending", f).clip(min=0)
    o["predicted_profit_180d"] = _run(models, "profit", f)
    P = _run(models, "fav_category", f)
    o["fav_category"] = np.array(C.CATEGORIES)[P.argmax(1)]
    o["fav_category_conf"] = P.max(1)
    top3 = np.argsort(-P, axis=1)[:, :3]
    o["fav_top3"] = [", ".join(C.CATEGORIES[i] for i in r) for r in top3]
    o["expected_active_days"] = _run(models, "active_duration", f).clip(0, C.HORIZON_DAYS)
    o["discount_sensitivity_prob"] = _run(models, "discount", f)
    L = _run(models, "loyalty", f)
    for i, name in enumerate(LOYALTY_LABELS):
        o[f"loyalty_p_{name.lower()}"] = L[:, i]
    o["loyalty_score"] = (L[:, 1] * 0.5 + L[:, 2]) * 100
    o["loyalty_tier"] = np.array(LOYALTY_LABELS)[L.argmax(1)]

    last_idx = df.groupby("customer_id").tail(1).index
    S = order_state_features(df).loc[last_idx]
    S.index = df.loc[last_idx, "customer_id"].values
    PN = _run(models, "next_product", S)
    o["next_product"] = pd.Series(np.array(C.CATEGORIES)[PN.argmax(1)], index=S.index).reindex(o.index)
    o["next_product_conf"] = pd.Series(PN.max(1), index=S.index).reindex(o.index)
    o["next_top3"] = pd.Series([", ".join(C.CATEGORIES[i] for i in r) for r in np.argsort(-PN, axis=1)[:, :3]],
                               index=S.index).reindex(o.index)
    days = np.clip(_run(models, "next_purchase", S), 0, C.NEXT_PURCHASE_CAP_DAYS)
    o["next_purchase_days"] = pd.Series(days, index=S.index).reindex(o.index)
    o["next_purchase_date"] = o["last_order_date"] + pd.to_timedelta(o["next_purchase_days"].round(), unit="D")

    pr = _run(models, "return_risk", return_features(df))
    o["return_propensity"] = pd.Series(pr, index=df["customer_id"].values).groupby(level=0).mean().reindex(o.index)

    q_lo, q_hi = o["churn_prob"].quantile([0.4, 0.7])
    o["churn_risk"] = np.where(o["churn_prob"] >= q_hi, "High", np.where(o["churn_prob"] >= q_lo, "Medium", "Low"))
    o = o.reset_index()
    C.OUT_DIR.mkdir(exist_ok=True, parents=True)
    o.to_csv(C.OUT_DIR / "customer_scores.csv", index=False)
    return o, dict(churn_q_low=float(q_lo), churn_q_high=float(q_hi))
