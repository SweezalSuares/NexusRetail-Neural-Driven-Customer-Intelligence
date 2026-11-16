"""Customer Intelligence dashboard (Streamlit). Run:  streamlit run app/streamlit_app.py"""
import json
import sys
from datetime import date
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import accounts
import retraining
import ui
from src import config as C
from src.features import build_customer_dataset, customer_features, load_data, to_matrix
from src.tasks import LOYALTY_LABELS, TASK_BY_KEY, TASKS

CUR = "$"  # change to "₹" etc. if your data uses another currency

st.set_page_config(page_title="NexusRetail", page_icon="🌿", layout="wide",
                   initial_sidebar_state="expanded")
ui.setup_plotly()
ui.inject_css()


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading data…")
def get_df(data_version="0"):
    return load_data(include_live=True)


@st.cache_data
def get_reference_df():
    return load_data(include_live=False)


def versioned_path(root, version, filename):
    if version:
        return root / "versions" / version / filename
    return root / filename


@st.cache_data
def get_scores(model_version=""):
    path = (C.OUT_DIR / "score_versions" / model_version / "customer_scores.csv"
            if model_version else C.OUT_DIR / "customer_scores.csv")
    return pd.read_csv(path, parse_dates=["last_order_date", "next_purchase_date"])


@st.cache_data
def get_metrics(model_version=""):
    return json.loads(versioned_path(C.REPORTS_DIR, model_version, "metrics.json").read_text())


@st.cache_data
def get_curves(model_version=""):
    return json.loads(versioned_path(C.REPORTS_DIR, model_version, "curves.json").read_text())


@st.cache_data
def get_eda():
    return json.loads((C.REPORTS_DIR / "eda_summary.json").read_text())


@st.cache_data
def get_latest_features(data_version="0"):
    df = get_df(data_version)
    return customer_features(df, df["order_date"].max())


@st.cache_data
def get_diagnostics(data_version="0"):
    df = get_df(data_version)
    if df["customer_id"].astype(str).str.startswith("LOCAL-").any():
        test_snapshot = (df["order_date"].max().normalize() - pd.Timedelta(days=C.HORIZON_DAYS)).date()
    else:
        test_snapshot = C.TEST_SNAPSHOT
    te = build_customer_dataset(df, test_snapshot)
    d = df.sort_values(["customer_id", "order_date"])
    gap = d.groupby("customer_id")["order_date"].diff().dt.days.dropna()
    prev_cat = d.groupby("customer_id")["category"].shift()
    base = (df["category"].value_counts(normalize=True) ** 2).sum()
    return dict(
        corr_orders=float(te["future_orders"].corr(te["n_orders"])),
        gap_cv=float(gap.std() / gap.mean()),
        cat_repeat=float((d["category"] == prev_cat)[prev_cat.notna()].mean()), cat_chance=float(base),
        age_changes=float((df.groupby("customer_id")["customer_age"].nunique() > 1).mean()),
    )


@st.cache_resource(show_spinner="Loading neural networks…")
def get_models(model_version=""):
    from src.pipeline import load_models
    model_dir = C.MODELS_DIR / "versions" / model_version if model_version else C.MODELS_DIR
    return load_models(model_dir)


def money(x):
    return f"{CUR}{x:,.0f}"


def page_auth():
    st.markdown('<div class="auth-shell"><div class="auth-brand">NexusRetail</div>'
                '<p>Customer intelligence, thoughtfully designed.</p></div>', unsafe_allow_html=True)
    sign_in, customer_signup, admin_signup = st.tabs(
        ["Sign in", "Create customer account", "Admin registration"]
    )
    with sign_in:
        with st.form("sign_in"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            user = accounts.authenticate(email, password)
            if user is None:
                st.error("Email or password is incorrect.")
            else:
                st.session_state["account"] = user
                st.rerun()
    with customer_signup:
        with st.form("customer_signup"):
            name = st.text_input("Your name", key="customer_name")
            email = st.text_input("Email address", key="customer_email")
            password = st.text_input("Password (at least 10 characters)", type="password",
                                     key="customer_password")
            confirm = st.text_input("Confirm password", type="password", key="customer_confirm")
            profile = st.columns(3)
            age = profile[0].slider("Age", 18, 100, 35, key="customer_age")
            gender = profile[1].selectbox("Gender", C.GENDERS, key="customer_gender")
            region = profile[2].selectbox("Region", C.REGIONS, key="customer_region")
            submitted = st.form_submit_button("Create customer account", type="primary", width="stretch")
        if submitted:
            if password != confirm:
                st.error("The passwords do not match.")
            else:
                try:
                    st.session_state["account"] = accounts.register_user(
                        email, name, password, "customer", age=age, gender=gender, region=region
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()
    with admin_signup:
        st.caption("Admin accounts require an invite code configured by the application owner.")
        with st.form("admin_signup"):
            name = st.text_input("Admin name", key="admin_name")
            email = st.text_input("Admin email", key="admin_email")
            password = st.text_input("Password (at least 10 characters)", type="password",
                                     key="admin_password")
            confirm = st.text_input("Confirm password", type="password", key="admin_confirm")
            invite = st.text_input("Admin invite code", type="password", key="admin_invite")
            submitted = st.form_submit_button("Create admin account", type="primary", width="stretch")
        if submitted:
            if password != confirm:
                st.error("The passwords do not match.")
            else:
                try:
                    st.session_state["account"] = accounts.register_user(
                        email, name, password, "admin", invite
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.rerun()
    st.caption("Prototype accounts and orders are stored locally on this machine. No payment is processed.")


accounts.initialize_db()
if "account" not in st.session_state:
    page_auth()
    ui.footer()
    st.stop()


account = st.session_state["account"]
MODEL_VERSION = accounts.get_state("model_version")
DATA_VERSION = accounts.get_state("data_version", default="0")
retraining.maybe_start_retraining()
if account["role"] == "admin":
    try:
        DF = get_df(DATA_VERSION)
        SCORES = get_scores(MODEL_VERSION)
        METRICS = get_metrics(MODEL_VERSION)
    except FileNotFoundError:
        st.error("Model outputs not found. Run `python run_pipeline.py` first, then reload this page.")
        st.stop()
    META, MET = METRICS["meta"], METRICS["metrics"]
    END = DF["order_date"].max()
    LEVEL = {k: v["signal"] for k, v in MET.items()}


# ----------------------------------------------------------------------------- pages
def page_overview():
    ui.hero("NexusRetail",
            "Neural Driven Customer Intelligence")
    c = st.columns(6)
    ui.kpi(c[0], "Revenue", money(DF.total_amount.sum()), f"{META['start_date']} → {META['end_date']}")
    ui.kpi(c[1], "Profit", money(DF.profit_margin.sum()), f"{DF.profit_margin.sum() / DF.total_amount.sum():.1%} margin")
    ui.kpi(c[2], "Orders", f"{len(DF):,}", f"AOV {money(DF.total_amount.mean())}")
    ui.kpi(c[3], "Customers", f"{DF.customer_id.nunique():,}", f"{len(DF) / DF.customer_id.nunique():.1f} orders each")
    ui.kpi(c[4], "Return rate", f"{DF.is_returned.mean():.1%}", f"{int(DF.is_returned.sum()):,} returns")
    ui.kpi(c[5], "Discounted orders", f"{DF.is_discounted.mean():.0%}", f"avg {DF.discount.mean():.1%} off")
    st.write("")
    n_none = sum(v == "None" for v in LEVEL.values())
    ui.callout(f"<b>Honest model report:</b> on a strict temporal hold-out, {n_none} of 11 models perform at chance "
               "level (they do not beat a naive baseline). Diagnostics on the <b>Model Lab</b> page show why: in this "
               "dataset a customer's past behaviour does not predict their future behaviour. Every prediction in the "
               "app carries a reliability badge so it is never over-read.")
    m = DF.assign(month=DF.order_date.dt.to_period("M").dt.to_timestamp()).groupby("month").agg(
        Revenue=("total_amount", "sum"), Profit=("profit_margin", "sum")).reset_index()
    a, b = st.columns([1.6, 1])
    fig = go.Figure()
    fig.add_scatter(x=m.month, y=m.Revenue, name="Revenue", line=dict(color=C.MINT, width=3), fill="tozeroy",
                    fillcolor="rgba(217,189,133,.10)")
    fig.add_scatter(x=m.month, y=m.Profit, name="Profit", line=dict(color=C.TEAL, width=3))
    fig.update_layout(title="Monthly revenue & profit", height=340, legend=dict(orientation="h", y=1.12))
    a.plotly_chart(fig, width="stretch")
    cat = DF.groupby("category").agg(Revenue=("total_amount", "sum")).reset_index()
    b.plotly_chart(px.pie(cat, names="category", values="Revenue", hole=.55, title="Revenue by category",
                          color_discrete_sequence=C.PALETTE).update_layout(height=340), width="stretch")
    st.subheader("Model reliability at a glance")
    cols = st.columns(4)
    for i, t in enumerate(TASKS):
        lvl = LEVEL[t["key"]]
        cols[i % 4].markdown(f'<div class="card"><h4>{t["icon"]} #{t["id"]} {t["title"]}</h4>'
                             f'{ui.signal_badge(lvl)}<div class="note">{MET[t["key"]]["signal_note"]}</div></div>',
                             unsafe_allow_html=True)


def page_eda():
    ui.hero("🔎 EDA Explorer", "Slice the data and explore revenue, products, customers, discounts and returns.")
    with st.expander("Filters", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        dr = f1.date_input("Order dates", (DF.order_date.min().date(), END.date()),
                           min_value=DF.order_date.min().date(), max_value=END.date())
        cats = f2.multiselect("Category", C.CATEGORIES, default=C.CATEGORIES)
        regs = f3.multiselect("Region", C.REGIONS, default=C.REGIONS)
        pays = f4.multiselect("Payment method", C.PAYMENTS, default=C.PAYMENTS)
    d0, d1 = (dr if isinstance(dr, tuple) and len(dr) == 2 else (DF.order_date.min().date(), END.date()))
    f = DF[(DF.order_date.dt.date >= d0) & (DF.order_date.dt.date <= d1) & DF.category.isin(cats)
           & DF.region.isin(regs) & DF.payment_method.isin(pays)]
    if f.empty:
        st.warning("No orders match these filters.")
        return
    c = st.columns(5)
    ui.kpi(c[0], "Orders", f"{len(f):,}")
    ui.kpi(c[1], "Revenue", money(f.total_amount.sum()))
    ui.kpi(c[2], "Profit", money(f.profit_margin.sum()))
    ui.kpi(c[3], "Customers", f"{f.customer_id.nunique():,}")
    ui.kpi(c[4], "Return rate", f"{f.is_returned.mean():.1%}")
    st.write("")
    t1, t2, t3, t4, t5, t6 = st.tabs(["Trends", "Products", "Customers", "Discounts & returns", "Correlations", "Data quality"])
    with t1:
        metric = st.radio("Metric", ["total_amount", "profit_margin", "order_id"], horizontal=True,
                          format_func=lambda x: {"total_amount": "Revenue", "profit_margin": "Profit", "order_id": "Orders"}[x])
        agg = "count" if metric == "order_id" else "sum"
        g = f.assign(month=f.order_date.dt.to_period("M").dt.to_timestamp()).groupby("month")[metric].agg(agg).reset_index()
        st.plotly_chart(px.area(g, x="month", y=metric, title="Monthly trend", color_discrete_sequence=[C.MINT]), width="stretch")
        a, b = st.columns(2)
        dw = f.groupby(f.order_date.dt.day_name())[metric].agg(agg).reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index()
        a.plotly_chart(px.bar(dw, x="order_date", y=metric, title="By weekday", color_discrete_sequence=[C.TEAL]), width="stretch")
        hm = f.assign(m=f.order_date.dt.month).pivot_table(index="category", columns="m", values=metric, aggfunc=agg)
        b.plotly_chart(px.imshow(hm, aspect="auto", title="Category × month", color_continuous_scale=[C.CHARCOAL_LIGHT, C.TEAL, C.MINT]), width="stretch")
    with t2:
        cat = f.groupby("category").agg(Revenue=("total_amount", "sum"), Profit=("profit_margin", "sum"),
                                        Orders=("order_id", "count"), Return=("is_returned", "mean")).reset_index()
        cat["Margin"] = cat.Profit / cat.Revenue
        a, b = st.columns(2)
        a.plotly_chart(px.bar(cat.sort_values("Revenue"), x="Revenue", y="category", orientation="h", title="Revenue",
                              color_discrete_sequence=[C.MINT]), width="stretch")
        b.plotly_chart(px.bar(cat.sort_values("Margin"), x="Margin", y="category", orientation="h",
                              title="Profit margin (Grocery loses money)", color_discrete_sequence=[C.TEAL]), width="stretch")
        st.plotly_chart(px.box(f, x="category", y="price", log_y=True, title="Price distribution (log scale)",
                               color="category", color_discrete_sequence=C.PALETTE), width="stretch")
    with t3:
        per = f.groupby("customer_id").agg(orders=("order_id", "count"), spend=("total_amount", "sum")).reset_index()
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(per, x="orders", title="Orders per customer", color_discrete_sequence=[C.MINT]), width="stretch")
        b.plotly_chart(px.histogram(per, x="spend", log_y=True, nbins=50, title="Customer spend (log count)",
                                    color_discrete_sequence=[C.TEAL]), width="stretch")
        s = per.spend.sort_values(ascending=False).cumsum() / per.spend.sum()
        par = pd.DataFrame({"customers": np.arange(1, len(s) + 1) / len(s), "revenue": s.values})
        a.plotly_chart(px.line(par, x="customers", y="revenue", title="Pareto: share of revenue by top customers",
                               color_discrete_sequence=[C.MINT]), width="stretch")
        b.plotly_chart(px.histogram(f, x="customer_age", color="customer_gender", barmode="overlay", opacity=.7,
                                    title="Age by gender", color_discrete_sequence=C.PALETTE), width="stretch")
    with t4:
        a, b = st.columns(2)
        dl = f.groupby(f.discount.round(2)).agg(qty=("quantity", "mean"), ret=("is_returned", "mean"), n=("order_id", "count")).reset_index()
        a.plotly_chart(px.bar(dl, x="discount", y="qty", title="Avg quantity by discount level",
                              color_discrete_sequence=[C.MINT]), width="stretch")
        b.plotly_chart(px.bar(f.groupby("category").is_returned.mean().reset_index(), x="category", y="is_returned",
                              title="Return rate by category", color_discrete_sequence=[C.WARN]), width="stretch")
        a.plotly_chart(px.line(f.groupby("delivery_time_days").is_returned.mean().reset_index(), x="delivery_time_days",
                               y="is_returned", markers=True, title="Return rate by delivery days",
                               color_discrete_sequence=[C.TEAL]), width="stretch")
        b.plotly_chart(px.bar(f.groupby("payment_method").is_returned.mean().reset_index(), x="payment_method", y="is_returned",
                              title="Return rate by payment method", color_discrete_sequence=[C.MINT_DARK]), width="stretch")
    with t5:
        num = f[["price", "discount", "quantity", "delivery_time_days", "total_amount", "shipping_cost", "profit_margin",
                 "customer_age", "is_returned"]].corr()
        st.plotly_chart(px.imshow(num, text_auto=".2f", aspect="auto", zmin=-1, zmax=1, title="Correlation matrix",
                                  color_continuous_scale=[C.CHARCOAL_LIGHT, C.TEAL, C.MINT]).update_layout(height=520), width="stretch")
    with t6:
        q = get_eda()["quality"]
        c = st.columns(4)
        ui.kpi(c[0], "Missing values", f"{q['missing_values']}")
        ui.kpi(c[1], "Duplicate rows", f"{q['duplicates']}")
        ui.kpi(c[2], "total = price×qty×(1-disc)", f"{q['total_matches_formula']:.0%}")
        ui.kpi(c[3], "Loss-making orders", f"{q['negative_profit_orders']:.1%}")
        ui.callout(f"<b>Data note:</b> {q['customers_with_changing_age']:.0%} of customers appear with more than one age, "
                   "so age, gender and region are not stable customer attributes in this dataset.")
        st.dataframe(f.head(500), width="stretch")
        st.download_button("⬇️ Download filtered data (CSV)", f.to_csv(index=False).encode(), "filtered_orders.csv", "text/csv")


def _pick_random():
    st.session_state["cust"] = SCORES.customer_id.sample(1).iloc[0]
    st.toast("Random customer selected", icon="🎲")


def _pick_risky():
    st.session_state["cust"] = SCORES.sort_values("revenue_at_risk", ascending=False).customer_id.iloc[0]
    st.toast("Highest revenue-at-risk customer selected", icon="🔥")


def page_customer():
    ui.hero("👤 Customer 360", "All 11 predictions for one customer, with reliability badges and history.")
    ids = SCORES.customer_id.sort_values().tolist()
    st.session_state.setdefault("cust", ids[0])
    a, b, c = st.columns([3, 1, 1.4])
    a.selectbox("Customer ID", ids, key="cust", help="Type to search")
    b.button("🎲 Random", on_click=_pick_random, width="stretch")
    c.button("🔥 Top revenue at risk", on_click=_pick_risky, width="stretch")
    r = SCORES[SCORES.customer_id == st.session_state["cust"]].iloc[0]
    hist = DF[DF.customer_id == r.customer_id]
    k = st.columns(5)
    ui.kpi(k[0], "Orders", f"{int(r.n_orders)}")
    ui.kpi(k[1], "Lifetime spend", money(r.total_spend))
    ui.kpi(k[2], "Lifetime profit", money(r.total_profit))
    ui.kpi(k[3], "Last order", r.last_order_date.strftime("%d %b %Y"), f"{int(r.recency_days)} days ago")
    k[4].markdown(f'<div class="kpi"><div class="label">Churn risk band</div><div class="value">'
                  f'{ui.risk_badge(r.churn_risk)}</div><div class="sub">relative to all customers</div></div>', unsafe_allow_html=True)
    st.write("")
    st.subheader("Predictions for the next 180 days")
    overdue = r.next_purchase_date < END
    cards = [
        ("churn", f"{r.churn_prob:.0%}", f"Risk band: {r.churn_risk}"),
        ("revenue_loss", money(r.revenue_at_risk), f"Expected loss = P(churn) × {money(r.spend_if_retained)} if retained"),
        ("spending", money(r.predicted_spend_180d), "Predicted order value"),
        ("profit", money(r.predicted_profit_180d), "Predicted profit contribution"),
        ("fav_category", r.fav_category, f"Top-3: {r.fav_top3} · confidence {r.fav_category_conf:.0%}"),
        ("return_risk", f"{r.return_propensity:.1%}", f"Avg across their orders · dataset avg {DF.is_returned.mean():.1%}"),
        ("active_duration", f"{r.expected_active_days:.0f} days", "Expected further active days"),
        ("next_product", r.next_product, f"Top-3: {r.next_top3}"),
        ("discount", f"{r.discount_sensitivity_prob:.0%}",
         "Discount-sensitive" if r.discount_sensitivity_prob >= .5 else "Not discount-sensitive"),
        ("loyalty", f"{r.loyalty_tier} · {r.loyalty_score:.0f}/100", "Loyalty tier and score"),
        ("next_purchase", r.next_purchase_date.strftime("%d %b %Y"),
         f"in ~{r.next_purchase_days:.0f} days after last order" + (" (date already passed)" if overdue else "")),
    ]
    cols = st.columns(4)
    for i, (key, value, note) in enumerate(cards):
        t = TASK_BY_KEY[key]
        ui.pred_card(cols[i % 4], t["icon"], f"#{t['id']} {t['title']}", t["question"], value, note, LEVEL[key])
    ui.callout("Badges show how much each model can be trusted on unseen future data. “At chance level” means the "
               "value is indicative only (it is close to the dataset average).")
    a, b = st.columns(2)
    a.plotly_chart(ui.gauge(r.churn_prob * 100, "Churn probability",
                            color=C.DANGER if r.churn_risk == "High" else C.WARN if r.churn_risk == "Medium" else C.MINT), width="stretch")
    lo = pd.DataFrame({"tier": LOYALTY_LABELS, "p": [r.loyalty_p_low, r.loyalty_p_medium, r.loyalty_p_high]})
    b.plotly_chart(px.bar(lo, x="tier", y="p", title="Loyalty tier probabilities", color="tier",
                          color_discrete_sequence=[C.MUTED, C.TEAL, C.MINT]).update_layout(
        showlegend=False, height=260, yaxis_tickformat=".0%"), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.scatter(hist, x="order_date", y="total_amount", color="category", size="quantity",
                              title="Order history", color_discrete_sequence=C.PALETTE), width="stretch")
    mix = hist.groupby("category").total_amount.sum().reset_index()
    b.plotly_chart(px.pie(mix, names="category", values="total_amount", hole=.5, title="Spend by category",
                          color_discrete_sequence=C.PALETTE), width="stretch")
    with st.expander("Order history table"):
        st.dataframe(hist.drop(columns=["is_returned", "is_discounted", "gross_amount"]), width="stretch")
    st.download_button("⬇️ Download this customer's predictions", r.to_frame().T.to_csv(index=False).encode(),
                       f"{r.customer_id}_predictions.csv", "text/csv")


def page_customer_portal():
    account = st.session_state["account"]
    customer_id = account["customer_id"]
    profile_age = account.get("age") or 35
    profile_gender = account.get("gender") if account.get("gender") in C.GENDERS else "Other"
    profile_region = account.get("region") if account.get("region") in C.REGIONS else "Central"
    ui.hero(f"Welcome, {escape(account['display_name'])}",
            "Your orders, shopping and personal spending at a glance.")
    st.caption(f"Customer account · {customer_id}")

    st.subheader("Place an order")
    st.caption("Demo orders are recorded for your history and the retail analytics. No payment is processed.")
    with st.form("place_order"):
        a, b, c, d = st.columns(4)
        product = a.text_input("Product name", placeholder="For example, Everyday tote")
        category = b.selectbox("Category", C.CATEGORIES)
        quantity = c.number_input("Quantity", min_value=1, max_value=25, value=1)
        unit_price = d.number_input("Unit price", min_value=0.01, max_value=3000.0, value=80.0, step=1.0)
        a, b, c = st.columns(3)
        discount = a.select_slider("Discount", [0, .05, .1, .15, .2, .3], value=0.0,
                                   format_func=lambda value: f"{value:.0%}")
        payment = b.selectbox("Payment method", C.PAYMENTS)
        c.caption(f"Profile: age {profile_age} · {profile_gender} · {profile_region}")
        submitted = st.form_submit_button("Place demo order", type="primary")
    if submitted:
        try:
            reference = get_reference_df()
            category_history = reference[reference.category == category]
            category_revenue = category_history.total_amount.sum()
            profit_rate = (category_history.profit_margin.sum() / category_revenue
                           if category_revenue else 0.1)
            delivery_days = max(1, int(category_history.delivery_time_days.median())) if not category_history.empty else 5
            shipping_cost = float(category_history.shipping_cost.mean()) if not category_history.empty else 6.0
            amount = unit_price * quantity * (1 - discount)
            order_id = accounts.create_order(
                customer_id, product, category, unit_price, quantity, discount, payment, date.today(),
                delivery_time_days=delivery_days, shipping_cost=shipping_cost,
                profit_margin=amount * profit_rate, age=profile_age, gender=profile_gender,
                region=profile_region,
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            retraining.maybe_start_retraining()
            st.success(f"Order {order_id} has been saved to your local order history.")
            st.rerun()

    history = pd.DataFrame(accounts.get_orders(customer_id))
    st.subheader("Your analytics")
    if history.empty:
        for col, label, value, note in zip(
            st.columns(3), ["Orders", "Lifetime spend", "Average order"], ["0", money(0), money(0)],
            ["No orders yet", "Your saved purchases", "Your order average"],
        ):
            ui.kpi(col, label, value, note)
        ui.callout("Your previous orders and personal analytics will appear here after your first demo order.")
        return

    history["order_date"] = pd.to_datetime(history["order_date"])
    columns = st.columns(4)
    ui.kpi(columns[0], "Orders", f"{len(history):,}")
    ui.kpi(columns[1], "Lifetime spend", money(history.total_amount.sum()))
    ui.kpi(columns[2], "Average order", money(history.total_amount.mean()))
    favorite = history.groupby("category").total_amount.sum().idxmax()
    ui.kpi(columns[3], "Top category", favorite, "by your recorded spend")
    monthly = history.assign(month=history.order_date.dt.to_period("M").dt.to_timestamp())
    trend = monthly.groupby("month", as_index=False).total_amount.sum()
    mix = history.groupby("category", as_index=False).total_amount.sum()
    left, right = st.columns(2)
    left.plotly_chart(px.area(trend, x="month", y="total_amount", title="Your monthly spend",
                              color_discrete_sequence=[C.CHAMPAGNE_GOLD]), width="stretch")
    right.plotly_chart(px.pie(mix, names="category", values="total_amount", hole=.55,
                              title="Your spending by category", color_discrete_sequence=C.PALETTE),
                      width="stretch")
    st.subheader("Previous orders")
    visible = history.drop(columns=["returned"]).rename(columns={
        "order_id": "Order", "product_name": "Product", "category": "Category",
        "unit_price": "Unit price", "quantity": "Qty", "discount": "Discount",
        "payment_method": "Payment", "order_date": "Date", "total_amount": "Total",
        "return_date": "Return date",
    })
    visible["Status"] = np.where(history.returned.astype(bool), "Returned", "Not returned")
    st.dataframe(visible, width="stretch", hide_index=True, column_config={
        "Unit price": st.column_config.NumberColumn(format=f"{CUR}%.2f"),
        "Discount": st.column_config.NumberColumn(format="percent"),
        "Total": st.column_config.NumberColumn(format=f"{CUR}%.2f"),
        "Date": st.column_config.DateColumn(),
        "Return date": st.column_config.DateColumn(),
    })
    with st.expander("Return an order"):
        eligible = history[history.returned == 0]
        if eligible.empty:
            st.caption("All your orders are already marked returned.")
        for row in eligible.itertuples(index=False):
            if st.button(f"Return {row.product_name} · {row.order_id}", key=f"return_{row.order_id}"):
                try:
                    accounts.mark_order_returned(customer_id, row.order_id)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    retraining.maybe_start_retraining()
                    st.success(f"Order {row.order_id} marked as returned.")
                    st.rerun()
    st.download_button("Download your order history", visible.to_csv(index=False).encode(),
                       "my_orders.csv", "text/csv")


def page_segments():
    ui.hero(" Segments & Actions", "Prioritise retention using churn risk, loyalty and revenue at risk.")
    a, b, c = st.columns(3)
    risk = a.multiselect("Churn risk band", ["High", "Medium", "Low"], default=["High", "Medium"])
    tiers = b.multiselect("Loyalty tier", LOYALTY_LABELS, default=LOYALTY_LABELS)
    min_spend = c.slider("Min lifetime spend", 0, int(SCORES.total_spend.quantile(.99)), 0, step=50)
    v = SCORES[SCORES.churn_risk.isin(risk) & SCORES.loyalty_tier.isin(tiers) & (SCORES.total_spend >= min_spend)]
    if v.empty:
        st.warning("No customers match these filters.")
        return
    k = st.columns(4)
    ui.kpi(k[0], "Customers in view", f"{len(v):,}", f"{len(v) / len(SCORES):.0%} of base")
    ui.kpi(k[1], "Revenue at risk", money(v.revenue_at_risk.sum()), "Σ P(churn) × value if retained")
    ui.kpi(k[2], "Avg churn probability", f"{v.churn_prob.mean():.0%}")
    ui.kpi(k[3], "Predicted spend (180d)", money(v.predicted_spend_180d.sum()))
    st.write("")
    a, b = st.columns(2)
    ct = pd.crosstab(SCORES.churn_risk, SCORES.loyalty_tier).reindex(index=["High", "Medium", "Low"], columns=LOYALTY_LABELS, fill_value=0)
    a.plotly_chart(px.imshow(ct, text_auto=True, title="Churn risk × loyalty tier (all customers)", aspect="auto",
                             color_continuous_scale=[C.CHARCOAL_LIGHT, C.TEAL, C.MINT]), width="stretch")
    s = v.sample(min(len(v), 1500), random_state=1)
    b.plotly_chart(px.scatter(s, x="total_spend", y="churn_prob", size="revenue_at_risk", color="loyalty_tier", log_x=True,
                              title="Value vs churn probability", color_discrete_sequence=[C.MUTED, C.TEAL, C.MINT],
                              category_orders={"loyalty_tier": LOYALTY_LABELS}), width="stretch")
    n = st.slider("Show top N customers by revenue at risk", 10, 200, 25, step=5)
    top = v.sort_values("revenue_at_risk", ascending=False).head(n)
    show = top[["customer_id", "churn_risk", "churn_prob", "revenue_at_risk", "total_spend", "loyalty_tier", "fav_category",
                "next_product", "next_purchase_date"]]
    st.dataframe(show, width="stretch", hide_index=True, column_config={
        "churn_prob": st.column_config.ProgressColumn("Churn prob", format="percent", min_value=0, max_value=1),
        "revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format=f"{CUR}%.0f"),
        "total_spend": st.column_config.NumberColumn("Lifetime spend", format=f"{CUR}%.0f"),
        "next_purchase_date": st.column_config.DateColumn("Next purchase")})
    st.download_button("⬇️ Download segment (CSV)", v.to_csv(index=False).encode(), "segment_customers.csv", "text/csv")
    with st.expander("Suggested playbook"):
        st.markdown("- **High risk + High loyalty:** personal outreach, loyalty reward.\n"
                    "- **High risk + Low loyalty:** win-back discount on their preferred category.\n"
                    "- **Low risk + High loyalty:** early access, referral programme.\n"
                    "- **Discount-sensitive:** time offers around the predicted next-purchase date.")


def page_orders():
    ui.hero("Orders & Returns", "Customer purchases and returns shared with the retail admin.")
    status = accounts.get_state("training_status", default="waiting")
    matured = accounts.get_matured_order_count()
    trained = int(accounts.get_state("last_matured_count", default="0"))
    new_matured = max(0, matured - trained)
    return_labels = accounts.get_matured_return_count()
    trained_returns = int(accounts.get_state("last_return_event_count", default="0"))
    new_returns = max(0, return_labels - trained_returns)
    stats = st.columns(4)
    ui.kpi(stats[0], "Training status", status.title())
    ui.kpi(stats[1], "All demo orders", f"{len(accounts.get_admin_orders()):,}")
    ui.kpi(stats[2], "Matured local orders", f"{matured:,}")
    ui.kpi(stats[3], "New training data", f"{new_matured:,} orders · {new_returns:,} returns",
           "100 of either trigger a background retrain")
    if status == "running":
        ui.callout("The all-model training batch is running. Tasks with enough labels will be retrained; others "
                   "retain their previous model. The active version remains available until the batch completes.")
    elif status == "queued":
        ui.callout("A full model retrain is queued.")
    elif status == "failed":
        ui.callout(f"Automatic retraining failed: {escape(accounts.get_state('training_error'))}")
    elif status == "complete":
        retained = sum(m.get("training_status") == "retained_previous" for m in MET.values())
        detail = f" · {retained} models retained their prior version due to insufficient labels" if retained else ""
        ui.callout(
            f"Latest automatic training completed · model version "
            f"{accounts.get_state('model_version')[:12]}{detail}.",
            ok=retained == 0,
        )
    st.caption("All-model training starts after 100 new orders mature (180 days) or 100 returned orders have mature "
               "180-day outcomes. Returns are saved immediately and become eligible when their outcomes mature.")

    orders = pd.DataFrame(accounts.get_admin_orders())
    if orders.empty:
        st.info("Customer orders will appear here after customers place demo orders.")
        return
    orders["Status"] = np.where(orders.returned.astype(bool), "Returned", "Not returned")
    orders = orders.drop(columns=["returned"]).rename(columns={
        "order_id": "Order", "customer_id": "Customer ID", "display_name": "Customer",
        "email": "Email", "product_name": "Product", "category": "Category",
        "unit_price": "Unit price", "quantity": "Qty", "discount": "Discount",
        "payment_method": "Payment", "order_date": "Date", "total_amount": "Total",
        "return_date": "Return date",
    })
    st.dataframe(orders, width="stretch", hide_index=True, column_config={
        "Unit price": st.column_config.NumberColumn(format=f"{CUR}%.2f"),
        "Discount": st.column_config.NumberColumn(format="percent"),
        "Total": st.column_config.NumberColumn(format=f"{CUR}%.2f"),
        "Date": st.column_config.DateColumn(),
        "Return date": st.column_config.DateColumn(),
    })
    st.download_button("Download all customer orders", orders.to_csv(index=False).encode(),
                       "customer_orders.csv", "text/csv")


def _return_row(bundle, p):
    row = dict.fromkeys(bundle["features"], 0.0)
    gross = p["price"] * p["qty"]
    total = gross * (1 - p["disc"])
    m = p["month"]
    row.update(price=p["price"], gross_amount=gross, quantity=p["qty"], discount=p["disc"], delivery_time_days=p["deliv"],
               shipping_cost=p["ship"], shipping_ratio=min(p["ship"] / max(total, 1), 5), customer_age=p["age"],
               month_sin=np.sin(2 * np.pi * m / 12), month_cos=np.cos(2 * np.pi * m / 12), n_prior_orders=p["n_prior"],
               prior_return_rate=(p["past_ret"] * p["n_prior"] + 3 * C.PRIOR_RETURN_RATE) / (p["n_prior"] + 3),
               prior_avg_discount=p["disc"] if p["n_prior"] else 0.0,
               prior_avg_spend=total if p["n_prior"] else 0.0, has_prev=int(p["n_prior"] > 0),
               days_since_prev=90 if p["n_prior"] else 0)
    for pre, val in (("cat", p["cat"]), ("pay", p["pay"]), ("reg", p["reg"]), ("gen", p["gen"])):
        row[f"{pre}_{val.replace(' ', '_')}"] = 1.0
    return pd.DataFrame([row])[bundle["features"]]


def _first_order_customer_row(p, columns):
    spend = p["price"] * p["qty"] * (1 - p["disc"])
    profit = spend * p["margin"]
    row = dict.fromkeys(columns, 0.0)
    row.update(
        n_orders=1, orders_per_month=1, total_spend=spend, avg_order_value=spend,
        max_order_value=spend, avg_price=p["price"], total_profit=profit, avg_profit=profit,
        profit_rate=p["margin"], total_items=p["qty"], avg_items=p["qty"],
        avg_discount=p["disc"], discount_share=float(p["disc"] > 0),
        avg_delivery_days=p["deliv"], avg_shipping_cost=p["ship"], avg_age=p["age"],
        n_categories=1, orders_last_90d=1, spend_last_90d=spend,
        orders_last_180d=1, spend_last_180d=spend, recent_order_share=1,
        top_category_share=1, last_amount=spend, last_discount=p["disc"],
    )
    for prefix, selected in (
        ("cat", p["cat"]),
        ("last_cat", p["cat"]),
        ("pay", p["pay"]),
        ("reg", p["reg"]),
        ("gen", p["gen"]),
    ):
        row[f"{prefix}_{selected.replace(' ', '_')}"] = 1.0
    return pd.DataFrame([row])[columns]


def _first_order_state_row(p, columns):
    spend = p["price"] * p["qty"] * (1 - p["disc"])
    row = dict.fromkeys(columns, 0.0)
    row.update(
        n_orders=1, spend_so_far=spend, avg_order_value=spend, profit_rate=p["margin"],
        avg_discount=p["disc"], discount_share=float(p["disc"] > 0), last_amount=spend,
        last_qty=p["qty"], last_discount=p["disc"], customer_age=p["age"],
        month_sin=np.sin(2 * np.pi * p["date"].month / 12),
        month_cos=np.cos(2 * np.pi * p["date"].month / 12), dow=p["date"].weekday(),
    )
    row[f"cat_{p['cat']}"] = 1.0
    row[f"last_cat_{p['cat']}"] = 1.0
    return pd.DataFrame([row])[columns]


def _predict_one(models, key, frame):
    from src.models import predict

    model, bundle = models[key]
    return predict(model, bundle, to_matrix(frame, bundle["features"]))


def page_whatif():
    ui.hero(" What-if Lab", "Change inputs and watch the neural networks respond.")
    t1, t2, t3 = st.tabs(["↩️ Order return-risk simulator", "👤 Customer scenario", "➕ New customer"])
    with t1:
        with st.form("ret"):
            a, b, c, d = st.columns(4)
            cat = a.selectbox("Category", C.CATEGORIES, index=2)
            price = b.number_input("Unit price", 1.0, 3000.0, 80.0)
            qty = c.slider("Quantity", 1, 5, 1)
            disc = d.select_slider("Discount", [0, .05, .1, .15, .2, .3], value=0.0, format_func=lambda x: f"{x:.0%}")
            a, b, c, d = st.columns(4)
            pay = a.selectbox("Payment", C.PAYMENTS)
            reg = b.selectbox("Region", C.REGIONS)
            deliv = c.slider("Delivery days", 3, 13, 5)
            ship = d.number_input("Shipping cost", 0.0, 20.0, 6.0)
            a, b, c, d = st.columns(4)
            age = a.slider("Customer age", 18, 69, 40)
            gen = b.selectbox("Gender", C.GENDERS)
            n_prior = c.slider("Previous orders", 0, 12, 3)
            past_ret = d.slider("Past return rate", 0.0, 1.0, 0.0, step=.05)
            month = st.slider("Order month", 1, 12, 6)
            go_btn = st.form_submit_button("Predict return risk", type="primary")
        if go_btn:
            with st.spinner("Running the neural network…"):
                from src.models import predict
                model, bundle = get_models(MODEL_VERSION)["return_risk"]
                x = _return_row(bundle, dict(cat=cat, price=price, qty=qty, disc=disc, pay=pay, reg=reg, deliv=deliv, ship=ship,
                                             age=age, gen=gen, n_prior=n_prior, past_ret=past_ret, month=month))
                p = float(predict(model, bundle, to_matrix(x, bundle["features"]))[0])
            base = DF[DF.category == cat].is_returned.mean()
            a, b = st.columns(2)
            a.plotly_chart(ui.gauge(p * 100, "Predicted return probability", maxv=30, color=C.WARN), width="stretch")
            b.markdown(f'<div class="card"><h4>Compared with history</h4><div class="big">{p / DF.is_returned.mean():.1f}×</div>'
                       f'<div class="note">vs dataset average {DF.is_returned.mean():.1%}; the historical return rate for {cat} '
                       f'is {base:.1%}. Reliability: {ui.signal_badge(LEVEL["return_risk"])}</div></div>', unsafe_allow_html=True)
            st.toast("Prediction ready", icon="✅")
    with t2:
        F = get_latest_features(DATA_VERSION)
        cid = st.selectbox("Customer", SCORES.customer_id.tolist(), key="wi_c")
        row = F.loc[cid].copy()
        a, b, c = st.columns(3)
        n_o = a.slider("Number of orders", 1, 30, int(row.n_orders))
        recency_max = max(700, int(F.recency_days.max()))
        spend_max = max(1, int(F.total_spend.max()))
        rec = b.slider("Days since last order", 0, recency_max, int(row.recency_days))
        spend = c.slider("Lifetime spend", 1, spend_max, max(1, int(row.total_spend)))
        a, b, c = st.columns(3)
        o90 = a.slider("Orders in last 90 days", 0, 10, int(row.orders_last_90d))
        dshare = b.slider("Share of discounted orders", 0.0, 1.0, float(row.discount_share), step=.05)
        rr = c.slider("Return rate", 0.0, 1.0, float(row.return_rate), step=.05)
        new = row.copy()
        new[["n_orders", "recency_days", "total_spend", "orders_last_90d", "discount_share", "return_rate"]] = [n_o, rec, spend, o90, dshare, rr]
        new["avg_order_value"] = spend / max(n_o, 1)
        from src.models import predict
        M = get_models(MODEL_VERSION)

        def run(key, r):
            m, bdl = M[key]
            return predict(m, bdl, to_matrix(pd.DataFrame([r]), bdl["features"]))
        base_c, new_c = float(run("churn", row)[0]), float(run("churn", new)[0])
        base_s, new_s = float(run("spending", row)[0]), float(run("spending", new)[0])
        lo = run("loyalty", new)[0]
        k = st.columns(3)
        ui.kpi(k[0], "Churn probability", f"{new_c:.1%}", f"{new_c - base_c:+.1%} vs current")
        ui.kpi(k[1], "Predicted spend (180d)", money(max(new_s, 0)), f"{new_s - base_s:+,.0f} vs current")
        ui.kpi(k[2], "Most likely loyalty tier", LOYALTY_LABELS[int(np.argmax(lo))], f"P(High) {lo[2]:.0%}")
        ui.callout("Derived columns are only partly recomputed, so treat this as an approximate sensitivity check. "
                   f"Churn model reliability: {ui.signal_badge(LEVEL['churn'])}")
    with t3:
        st.markdown("Enter a new customer's profile and an assumed first order to generate all 11 predictions.")
        with st.form("new_customer"):
            a, b, c, d = st.columns(4)
            customer_id = a.text_input("Customer ID", "NEW-001")
            age = b.slider("Customer age", 18, 69, 35, key="new_age")
            gender = c.selectbox("Gender", C.GENDERS, key="new_gender")
            region = d.selectbox("Region", C.REGIONS, key="new_region")
            a, b, c, d = st.columns(4)
            category = a.selectbox("First-order category", C.CATEGORIES, key="new_category")
            payment = b.selectbox("Payment method", C.PAYMENTS, key="new_payment")
            price = c.number_input("Unit price", min_value=0.01, max_value=3000.0, value=80.0, step=1.0,
                                   key="new_price")
            quantity = d.slider("Quantity", 1, 5, 1, key="new_quantity")
            a, b, c, d = st.columns(4)
            discount = a.select_slider("Discount", [0, .05, .1, .15, .2, .3], value=0.0,
                                       format_func=lambda x: f"{x:.0%}", key="new_discount")
            delivery = b.slider("Delivery days", 3, 13, 5, key="new_delivery")
            shipping = c.number_input("Shipping cost", min_value=0.0, max_value=20.0, value=6.0,
                                      key="new_shipping")
            default_margin = float(np.clip(DF.profit_margin.sum() / DF.total_amount.sum() * 100, -100, 100))
            margin = d.number_input("Estimated profit margin (%)", min_value=-100.0, max_value=100.0,
                                    value=default_margin, step=1.0, key="new_margin")
            first_order_date = st.date_input("Expected first-order date", value=END.date(), key="new_order_date")
            submitted = st.form_submit_button("Predict all customer features", type="primary")
        if submitted:
            p = dict(
                price=price, qty=quantity, disc=discount, margin=margin / 100, age=age, gen=gender,
                reg=region, cat=category, pay=payment, deliv=delivery, ship=shipping, date=first_order_date,
            )
            models = get_models(MODEL_VERSION)
            customer_row = _first_order_customer_row(p, models["churn"][1]["features"])
            order_row = _first_order_state_row(p, models["next_product"][1]["features"])
            churn = float(_predict_one(models, "churn", customer_row)[0])
            retained_revenue = max(0.0, float(_predict_one(models, "revenue_loss", customer_row)[0]))
            spending = max(0.0, float(_predict_one(models, "spending", customer_row)[0]))
            profit = float(_predict_one(models, "profit", customer_row)[0])
            favorite = _predict_one(models, "fav_category", customer_row)[0]
            return_row = _return_row(models["return_risk"][1], dict(
                price=price, qty=quantity, disc=discount, month=first_order_date.month,
                cat=category, pay=payment, reg=region, gen=gender, deliv=delivery, ship=shipping,
                age=age, n_prior=0, past_ret=0.0,
            ))
            return_risk = float(_predict_one(models, "return_risk", return_row)[0])
            active_days = float(np.clip(_predict_one(models, "active_duration", customer_row)[0], 0, C.HORIZON_DAYS))
            next_product = _predict_one(models, "next_product", order_row)[0]
            discount_probability = float(_predict_one(models, "discount", customer_row)[0])
            loyalty = _predict_one(models, "loyalty", customer_row)[0]
            next_days = float(np.clip(_predict_one(models, "next_purchase", order_row)[0],
                                      0, C.NEXT_PURCHASE_CAP_DAYS))
            next_date = pd.Timestamp(first_order_date) + pd.to_timedelta(round(next_days), unit="D")
            results = {
                "churn": (f"{churn:.1%}", "Probability of no purchase in the next 180 days"),
                "revenue_loss": (money(churn * retained_revenue),
                                 f"Expected lost revenue; retained-customer value {money(retained_revenue)}"),
                "spending": (money(spending), "Predicted spend in the next 180 days"),
                "profit": (money(profit), "Predicted profit in the next 180 days"),
                "fav_category": (C.CATEGORIES[int(np.argmax(favorite))],
                                 f"Most likely repeat category · confidence {np.max(favorite):.1%}"),
                "return_risk": (f"{return_risk:.1%}", "Return / refund probability for the assumed first order"),
                "active_duration": (f"{active_days:.0f} days", "Expected active days over the 180-day horizon"),
                "next_product": (C.CATEGORIES[int(np.argmax(next_product))],
                                 f"Next category recommendation · confidence {np.max(next_product):.1%}"),
                "discount": (f"{discount_probability:.1%}", "Probability of discount-sensitive purchasing"),
                "loyalty": (LOYALTY_LABELS[int(np.argmax(loyalty))], f"P(High) {loyalty[2]:.1%}"),
                "next_purchase": (next_date.strftime("%d %b %Y"), f"Predicted in about {next_days:.0f} days"),
            }
            st.subheader(f"Predictions for {customer_id.strip() or 'new customer'}")
            cols = st.columns(4)
            for i, task in enumerate(TASKS):
                value, note = results[task["key"]]
                ui.pred_card(cols[i % 4], task["icon"], f"#{task['id']} {task['title']}",
                             task["question"], value, note, LEVEL[task["key"]])
            ui.callout("This is an in-session cold-start prediction based on the profile and assumed first order; "
                       "it does not save a customer record or change the training data. Category recommendations "
                       "are used because this dataset cannot support reliable item-level recommendations.")


def page_models():
    ui.hero(" Model Lab", "Architecture, training curves, evaluation and honest comparison with baselines.")
    rows = []
    for t in TASKS:
        m = MET[t["key"]]
        if t["type"] == "binary":
            head, base = f"AUC {m['roc_auc']:.3f}", f"AUC {m['baseline_roc_auc']:.3f}"
        elif t["type"] == "multiclass":
            head, base = f"Acc {m['accuracy']:.1%}", f"Acc {max(m['majority_accuracy'], m['baseline_accuracy']):.1%}"
        else:
            head, base = f"MAE {m['mae']:.1f} · R²log {m['r2_log']:.3f}", f"MAE {m['baseline_mae']:.1f}"
        rows.append({"#": t["id"], "Task": t["title"], "Level": t["level"], "Type": t["type"], "MLP": head,
                     "Baseline": base, "Reliability": {"None": "○ chance", "Weak": "◐ weak", "Useful": "✓ useful"}[m["signal"]],
                     "Training status": m.get("training_status", "baseline"),
                     "Train / Test": f"{m['n_train'] + m['n_val']:,} / {m['n_test']:,}",
                     "Params": f"{m['params']:,}"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.subheader("Why most models are close to chance")
    dg = get_diagnostics(DATA_VERSION)
    k = st.columns(4)
    ui.kpi(k[0], "Past vs future orders", f"r = {dg['corr_orders']:.3f}", "0 = no relationship")
    ui.kpi(k[1], "Gap between orders CV", f"{dg['gap_cv']:.2f}", "≈1 = memoryless / random")
    ui.kpi(k[2], "Same category twice in a row", f"{dg['cat_repeat']:.1%}", f"chance = {dg['cat_chance']:.1%}")
    ui.kpi(k[3], "Customers whose age changes", f"{dg['age_changes']:.0%}", "attributes not stable")
    ui.callout("Gradient-boosted trees reach the same ~0.50 AUC on churn as the MLP, so this is a property of the data "
               "(orders look randomly assigned to customer IDs), not of the network. Return risk is the exception: it "
               "depends on the product category, so it shows a weak but real signal. Swap in a real transaction dataset "
               "(e.g. Online Retail II) and rerun <code>python run_pipeline.py</code> to see the same 11 models learn real behaviour.")
    st.subheader("Inspect a model")
    key = st.selectbox("Model", [t["key"] for t in TASKS], format_func=lambda k: f"#{TASK_BY_KEY[k]['id']} {TASK_BY_KEY[k]['title']}")
    t, m, cv = TASK_BY_KEY[key], MET[key], get_curves(MODEL_VERSION)[key]
    st.markdown(f"**{t['question']}**  \nTarget: {t['target']}")
    model_dir = C.MODELS_DIR / "versions" / MODEL_VERSION if MODEL_VERSION else C.MODELS_DIR
    bundle_hidden = " → ".join(str(h) for h in __import__("joblib").load(model_dir / f"{key}_bundle.joblib")["hidden"])
    st.markdown(f"Architecture: **Input({m['n_features']}) → Dense({bundle_hidden}) with BatchNorm + ReLU + Dropout → "
                f"{ {'binary': 'Sigmoid(1)', 'multiclass': 'Softmax', 'regression': 'Linear(1)'}[t['type']] }** · "
                f"{m['params']:,} parameters · stopped after {m['epochs_run']} epochs · {ui.signal_badge(m['signal'])}",
                unsafe_allow_html=True)
    a, b = st.columns(2)
    h = pd.DataFrame({"epoch": range(1, len(cv["history"]["loss"]) + 1), "train": cv["history"]["loss"], "validation": cv["history"]["val_loss"]})
    a.plotly_chart(px.line(h, x="epoch", y=["train", "validation"], title="Training curve (loss)",
                           color_discrete_sequence=[C.MINT, C.WARN]), width="stretch")
    if t["type"] == "binary":
        fig = go.Figure()
        fig.add_scatter(x=cv["roc"]["fpr"], y=cv["roc"]["tpr"], name=f"MLP (AUC {m['roc_auc']:.3f})", line=dict(color=C.MINT, width=3))
        fig.add_scatter(x=cv["roc_baseline"]["fpr"], y=cv["roc_baseline"]["tpr"], name=f"Logistic (AUC {m['baseline_roc_auc']:.3f})",
                        line=dict(color=C.TEAL, dash="dash"))
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color=C.MUTED, dash="dot"))
        fig.update_layout(title="ROC curve on hold-out", xaxis_title="False positive rate", yaxis_title="True positive rate")
        b.plotly_chart(fig, width="stretch")
        cm = np.array(cv["confusion"])
        c = st.columns(5)
        for col, (lab, val) in zip(c, [("ROC-AUC", m["roc_auc"]), ("PR-AUC", m["pr_auc"]), ("F1", m["f1"]),
                                       ("Precision", m["precision"]), ("Recall", m["recall"])]):
            ui.kpi(col, lab, f"{val:.3f}")
        st.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred 0", "Pred 1"], y=["Actual 0", "Actual 1"], title="Confusion matrix",
                                  color_continuous_scale=[C.CHARCOAL_LIGHT, C.TEAL, C.MINT]).update_layout(height=320), width="stretch")
    elif t["type"] == "multiclass":
        names = cv["class_names"]
        cm = np.array(cv["confusion"])
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        b.plotly_chart(px.imshow(cmn, text_auto=".0%", x=names, y=names, title="Confusion matrix (row-normalised)",
                                 color_continuous_scale=[C.CHARCOAL_LIGHT, C.TEAL, C.MINT]), width="stretch")
        comp = pd.DataFrame({"model": ["MLP", "Logistic reg.", "Majority class", "Random guess"],
                             "accuracy": [m["accuracy"], m["baseline_accuracy"], m["majority_accuracy"], m["random_accuracy"]]})
        st.plotly_chart(px.bar(comp, x="model", y="accuracy", title="Accuracy vs baselines", color_discrete_sequence=[C.MINT]).update_layout(
            yaxis_tickformat=".0%", height=300), width="stretch")
        st.caption(f"Top-3 accuracy: {m['top3_accuracy']:.1%} · macro-F1: {m['macro_f1']:.3f}")
    else:
        sc = pd.DataFrame(cv["scatter"])
        fig = px.scatter(sc, x="actual", y="pred", title="Actual vs predicted (hold-out sample)", opacity=.6,
                         color_discrete_sequence=[C.MINT])
        mx = float(max(sc.actual.max(), sc.pred.max()))
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=C.MUTED, dash="dot"), name="perfect")
        b.plotly_chart(fig, width="stretch")
        c = st.columns(4)
        ui.kpi(c[0], "MAE", f"{m['mae']:.2f}", f"Ridge {m['baseline_mae']:.2f}")
        ui.kpi(c[1], "RMSE", f"{m['rmse']:.2f}")
        ui.kpi(c[2], "R² (log scale)", f"{m['r2_log']:.3f}", f"Ridge {m['baseline_r2_log']:.3f}")
        ui.kpi(c[3], "Always-predict-mean MAE", f"{m['mean_mae']:.2f}")
    with st.expander("How the train / test split avoids leakage"):
        st.markdown(f"- Customer models use snapshots **{', '.join(META['train_snapshots'])}** for training and "
                    f"**{META['test_snapshot']}** as the hold-out; features use orders ≤ snapshot, targets use the next "
                    f"{META['horizon_days']} days, so no training label overlaps the test period.\n"
                    f"- Order-level models use training through **{META.get('order_train_end', C.ORDER_TRAIN_END)}** "
                    f"and test from **{META.get('order_test_start', C.ORDER_TEST_START)}**, with an embargo gap.\n"
                    f"- Return-risk hold-out runs **{META.get('return_test_start', C.ORDER_TEST_START)}** through "
                    f"**{META.get('return_test_end', META['end_date'])}**; live-order evaluation excludes orders "
                    "without mature 180-day outcomes.\n"
                    "- 15% of training customers are held out for early stopping. The decision threshold is tuned on that "
                    "validation set only.")


def page_about():
    account = st.session_state["account"]
    if account["role"] == "customer":
        ui.hero("About your customer space", "A simple place to shop and understand your purchases.")
        st.markdown("""
### What you can do
- Place demo orders and review your previous orders.
- Mark an order returned and see the updated status in your order history.
- Explore your spending over time and by product category.
- Download a copy of your locally saved order history.

Your profile, orders, and return status are stored in the shared local app database and are visible to retail admins.
This prototype does not connect to a real catalogue, inventory, or payment service.
""")
        return

    ui.hero("ℹ About NexusRetail", "Retail operations and neural-driven customer intelligence.")
    st.markdown("""
### Retail admin features
- **Overview:** revenue, profit, orders, customer counts, returns, discounts, trends, and model reliability.
- **Orders & Returns:** view and download customer order/return records and automatic training status.
- **EDA Explorer:** filter and explore product, customer, discount, return, and data-quality patterns.
- **Customer 360:** customer history and all 11 predictions, including spending, profit, churn, returns, loyalty,
  active duration, recommendation, discount sensitivity, and next purchase.
- **Segments & Actions:** prioritize customers by churn risk, loyalty, and expected revenue at risk.
- **What-if Lab:** simulate return risk, adjust existing-customer assumptions, and score a new customer's first order.
- **Model Lab:** inspect hold-out metrics, training curves, baselines, diagnostics, and model details.
- **About:** methodology, limitations, and setup.

### Pipeline and methodology
**Pipeline:** raw orders → cleaning & EDA → customer snapshot / order-level features → 11 Keras MLPs → temporal evaluation →
scoring of all customers → this dashboard.

**Network:** `Input → [Dense → BatchNorm → ReLU → Dropout] × N → Sigmoid | Softmax | Linear`, Adam optimiser, L2 regularisation,
early stopping and learning-rate decay. Regression targets use a signed log transform.

**Notes on the 11 predictions**
- *Revenue lost if a customer leaves* = churn probability × the revenue the customer would generate if retained (a separate MLP).
- *Frequently bought items* and *next product* are predicted at **category** level, because 24,912 product IDs appear on only
  34,500 orders (almost every product is bought once, so item-level recommendation is not learnable).
- *Return risk* is scored per order (features known at purchase time) and averaged for customer view. Live training
  excludes orders without mature 180-day outcomes, so recent unreturned orders are not mislabeled as negatives.
- All forecasts are for the 180 days after the last date in the data.
- Customer orders are included in training data. A background batch attempts all 11 models after 100 new local orders
  have mature 180-day outcome windows, or 100 returned orders have mature 180-day outcome windows. A task without
  enough valid labels retains its previous model; the active version stays available until the batch completes.

**Reproduce:** `pip install -r requirements.txt` → `python run_pipeline.py` → `streamlit run app/streamlit_app.py`
""")


account = st.session_state["account"]
if account["role"] == "admin":
    pages = {
        "Overview": page_overview, "EDA Explorer": page_eda, "Customer 360": page_customer,
        "Segments & Actions": page_segments, "Orders & Returns": page_orders, "What-if Lab": page_whatif,
        "Model Lab": page_models, "About": page_about,
    }
else:
    pages = {"My shop": page_customer_portal, "About": page_about}

if st.session_state.get("nav") not in pages:
    st.session_state["nav"] = next(iter(pages))

with st.sidebar:
    st.markdown("## NexusRetail")
    st.markdown(f'<div class="sidebar-user">{escape(account["display_name"])}</div>', unsafe_allow_html=True)
    st.caption(account["role"].title())
    page = st.radio("Navigate", list(pages), key="nav", label_visibility="collapsed")
    st.divider()
    if account["role"] == "admin":
        st.caption(f"Data: {META['start_date']} → {META['end_date']}")
        st.caption(f"{META['n_orders']:,} orders · {META['n_customers']:,} customers")
        st.markdown(f"{ui.signal_badge('Useful')} {ui.signal_badge('Weak')} {ui.signal_badge('None')}",
                    unsafe_allow_html=True)
        st.caption("Reliability of each model on unseen data")
    if st.button("Sign out", width="stretch"):
        del st.session_state["account"]
        st.session_state.pop("nav", None)
        st.rerun()

pages[page]()
ui.footer()
