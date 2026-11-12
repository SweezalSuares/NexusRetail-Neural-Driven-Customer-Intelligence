"""Exploratory data analysis: cleaning checks, summary tables and themed figures."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config as C

plt.rcParams.update({
    "figure.facecolor": C.CHARCOAL, "axes.facecolor": C.CHARCOAL_LIGHT, "savefig.facecolor": C.CHARCOAL,
    "text.color": C.TEXT, "axes.labelcolor": C.TEXT, "xtick.color": C.MUTED, "ytick.color": C.MUTED,
    "axes.edgecolor": C.CHARCOAL_MID, "grid.color": C.CHARCOAL_MID, "axes.grid": True, "grid.alpha": 0.5,
    "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.prop_cycle": plt.cycler(color=C.PALETTE),
})
CMAP = LinearSegmentedColormap.from_list("mint_teal", [C.CHARCOAL_LIGHT, C.TEAL, C.MINT])


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"{name}.png", dpi=130)
    plt.close(fig)


def data_quality(df):
    return dict(
        rows=int(len(df)), columns=int(df.shape[1]), duplicates=int(df.duplicated().sum()),
        missing_values=int(df.isna().sum().sum()), customers=int(df["customer_id"].nunique()),
        products=int(df["product_id"].nunique()),
        date_range=[str(df["order_date"].min().date()), str(df["order_date"].max().date())],
        return_rate=float(df["is_returned"].mean()), discounted_share=float(df["is_discounted"].mean()),
        total_revenue=float(df["total_amount"].sum()), total_profit=float(df["profit_margin"].sum()),
        total_matches_formula=float(np.isclose(df["price"] * df["quantity"] * (1 - df["discount"]),
                                               df["total_amount"], atol=0.02).mean()),
        customers_with_changing_age=float((df.groupby("customer_id")["customer_age"].nunique() > 1).mean()),
        negative_profit_orders=float((df["profit_margin"] < 0).mean()),
    )


def run_eda(df):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = dict(quality=data_quality(df))
    d = df.copy()
    d["month"] = d["order_date"].dt.to_period("M").dt.to_timestamp()

    m = d.groupby("month").agg(revenue=("total_amount", "sum"), profit=("profit_margin", "sum"), orders=("order_id", "count"))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(m.index, m["revenue"], color=C.MINT, lw=2.4, label="Revenue")
    ax.plot(m.index, m["profit"], color=C.TEAL, lw=2.4, label="Profit")
    ax.set_title("Monthly revenue and profit"); ax.legend(frameon=False)
    _save(fig, "01_monthly_revenue_profit")

    cat = d.groupby("category").agg(revenue=("total_amount", "sum"), profit=("profit_margin", "sum"),
                                    ret=("is_returned", "mean"), disc=("discount", "mean"))
    cat["margin"] = cat["profit"] / cat["revenue"]
    summary["category"] = cat.round(4).reset_index().to_dict("records")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cat["revenue"].sort_values().plot.barh(ax=axes[0], color=C.MINT); axes[0].set_title("Revenue by category")
    cat["margin"].sort_values().plot.barh(ax=axes[1], color=C.TEAL); axes[1].set_title("Profit margin by category")
    cat["ret"].sort_values().plot.barh(ax=axes[2], color=C.WARN); axes[2].set_title("Return rate by category")
    _save(fig, "02_category_performance")

    per = d.groupby("customer_id").agg(orders=("order_id", "count"), spend=("total_amount", "sum"),
                                       first=("order_date", "min"), last=("order_date", "max"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    per["orders"].value_counts().sort_index().plot.bar(ax=axes[0], color=C.MINT); axes[0].set_title("Orders per customer")
    axes[1].hist(np.log10(per["spend"]), bins=40, color=C.TEAL); axes[1].set_title("Customer lifetime spend (log10)")
    gaps = d.sort_values("order_date").groupby("customer_id")["order_date"].diff().dt.days.dropna()
    axes[2].hist(gaps.clip(upper=400), bins=40, color=C.MINT_DARK); axes[2].set_title("Days between orders")
    _save(fig, "03_customer_behaviour")
    summary["customer"] = dict(orders_mean=float(per["orders"].mean()), orders_median=float(per["orders"].median()),
                               single_order_share=float((per["orders"] == 1).mean()),
                               gap_median_days=float(gaps.median()), top10pct_revenue_share=float(
                                   per["spend"].nlargest(int(len(per) * 0.1)).sum() / per["spend"].sum()))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    d["discount"].round(2).value_counts().sort_index().plot.bar(ax=axes[0], color=C.MINT); axes[0].set_title("Discount levels")
    d.groupby("payment_method")["total_amount"].sum().sort_values().plot.barh(ax=axes[1], color=C.TEAL); axes[1].set_title("Revenue by payment method")
    d.groupby("region")["total_amount"].sum().sort_values().plot.barh(ax=axes[2], color=C.MINT_DARK); axes[2].set_title("Revenue by region")
    _save(fig, "04_discount_payment_region")

    d["age_group"] = pd.cut(d["customer_age"], [17, 25, 35, 45, 55, 70], labels=["18-25", "26-35", "36-45", "46-55", "56+"])
    ag = d.groupby("age_group", observed=True).agg(aov=("total_amount", "mean"), ret=("is_returned", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ag["aov"].plot.bar(ax=axes[0], color=C.MINT, rot=0); axes[0].set_title("Avg order value by age group")
    d.groupby("delivery_time_days")["is_returned"].mean().plot(ax=axes[1], color=C.WARN, marker="o"); axes[1].set_title("Return rate by delivery days")
    _save(fig, "05_age_delivery")

    num = d[["price", "discount", "quantity", "delivery_time_days", "total_amount", "shipping_cost",
             "profit_margin", "customer_age", "is_returned"]].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(num, cmap=CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(num))); ax.set_xticklabels(num.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(num))); ax.set_yticklabels(num.columns); ax.grid(False)
    for i in range(len(num)):
        for j in range(len(num)):
            ax.text(j, i, f"{num.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7, color=C.TEXT)
    ax.set_title("Correlation matrix"); fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, "06_correlation")

    fig, ax = plt.subplots(figsize=(8, 4))
    clip = d["total_amount"].clip(upper=d["total_amount"].quantile(0.98))
    ax.hist(clip, bins=50, color=C.MINT); ax.set_title("Order value distribution (clipped at 98th pct)")
    _save(fig, "07_order_value")

    m.reset_index().assign(month=lambda x: x["month"].dt.strftime("%Y-%m")).to_csv(C.REPORTS_DIR / "monthly_summary.csv", index=False)
    (C.REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"EDA done: {len(list(C.FIG_DIR.glob('*.png')))} figures in {C.FIG_DIR}")
    return summary
