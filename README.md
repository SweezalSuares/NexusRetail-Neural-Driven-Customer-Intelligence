# Customer Intelligence with ANN / Multilayer Perceptrons

End-to-end project on `ecommerce_sales_34500.csv` (34,500 orders, 7,903 customers, Sep 2023 – Sep 2025):
EDA → feature engineering → **11 Keras MLP models** → temporal evaluation → customer scoring → **Streamlit dashboard**
(Champagne gold, cocoa brown, blush beige, soft ivory, and muted rose theme).

## Quick start (Windows / macOS / Linux)

```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Trained models and scores are already included, so the app runs immediately. To rebuild everything from the CSV
(about 2 minutes on a laptop CPU): `python run_pipeline.py`
(`--skip-eda` to skip figures, `--skip-train` to only re-score). Windows shortcuts: `run_pipeline.bat`, `run_app.bat`.

## Accounts and dashboard roles

The app opens with sign-in and registration for customer and retail-admin accounts. Customer accounts can place
demo orders, review their own locally saved order history, and view personal spending analytics. The admin role
retains the existing Overview, EDA Explorer, Customer 360, Segments & Actions, What-if Lab, Model Lab, and About
pages, including all 11 existing predictions.

Customer accounts can register directly. Admin registration is invite-code protected: set
`NEXUS_ADMIN_INVITE_CODE` before starting Streamlit, then provide that code on the Admin registration tab.
For example, in PowerShell:

```powershell
$env:NEXUS_ADMIN_INVITE_CODE = "your-private-invite-code"
streamlit run app/streamlit_app.py
```

Accounts and demo orders are stored in the ignored local file `outputs/app_data.sqlite3` (override its path with
`NEXUS_APP_DB`). Customer registrations receive a new local customer ID; they are not automatically linked to a
customer in the historical training dataset. Customers can mark their orders returned from order history; order and
return status are shared with the admin **Orders & Returns** page.

The order records are appended to model input without changing the source CSV. All 11 models retrain in a background
process after **100 new local orders are at least 180 days old**, or 100 returned orders have mature 180-day outcome
windows. Training uses updated temporal splits and stores a versioned model, reports, and customer scores; the current
version stays active until the batch is complete. Tasks without sufficient training/hold-out labels keep their
previous model and are identified in Model Lab. Training runs are triggered while the app is being used. Demo orders
use estimated fulfilment and profit fields.
There is no live catalogue, inventory reservation, or payment processing. Do not expose this local prototype as a
production authentication or commerce service.

## The 11 predictions

| # | Prediction | Level | MLP head | Target (what the network learns) |
|---|-----------|-------|----------|----------------------------------|
| 1 | Churn | customer | sigmoid | no order in the next 180 days |
| 2 | Revenue lost if customer leaves | customer | linear | revenue in next 180 days *if retained*; expected loss = P(churn) × this |
| 3 | Customer spending | customer | linear | order value in next 180 days |
| 4 | Customer profit | customer | linear | profit in next 180 days |
| 5 | Frequently bought items | customer | softmax | top-spend category in next 180 days |
| 6 | Return / refund risk | order | sigmoid | `returned = Yes` (features known at order time) |
| 7 | Active duration | customer | linear | days until last order in next 180 days |
| 8 | Next product recommendation | order | softmax | category of the next order |
| 9 | Discount sensitivity | customer | sigmoid | ≥ 50% of next-180-day orders discounted |
| 10 | Customer loyalty | customer | softmax | tier from next-180-day orders: Low (0) / Medium (1) / High (2+) |
| 11 | Next purchase | order | linear | days to next order (capped at 120) |

Recommendations are at **category** level: 24,912 product IDs appear on only 34,500 orders, so item-level
recommendation cannot be learned from this file.

The **What-if Lab → New customer** form takes a customer profile and an assumed first order, then displays all
11 predictions. It is a cold-start, in-session estimate: it does not persist a customer record or modify the source
data, and the next-product recommendation is at category level.

## Network

`Input → [Dense → BatchNorm → ReLU → Dropout] × 2–3 → Sigmoid | Softmax | Linear`, Adam, L2 regularisation,
early stopping, learning-rate decay. Small nets (64-32, 16-8 for returns) because the data is noisy.

## Evaluation design (no leakage)

* Customer models: features from orders ≤ snapshot date, targets from the next 180 days. Train snapshots
  2024-03-14 / 2024-06-13 / 2024-09-12, **hold-out snapshot 2025-03-13** (its window ends before the data does).
* Order models: split by date with an embargo gap; return-history features use prior orders. For live records,
  return-risk training and hold-out include only orders whose 180-day outcome window has matured, so recent
  unreturned orders are not treated as negative labels.
* Every model is compared with a Logistic / Ridge baseline and naive predictors; ROC-AUC has a bootstrap 95% CI.

## Results — please read

| # | Model | Hold-out result | Baseline | Verdict |
|---|-------|-----------------|----------|---------|
| 1 | Churn | ROC-AUC 0.502 (CI 0.49–0.52) | 0.502 | no signal |
| 2–4, 7, 11 | Regressions | R² (log) ≈ 0 or negative | Ridge ≈ same | no signal |
| 5, 8, 10 | Multiclass | accuracy ≈ majority / chance | — | no signal |
| 6 | Return risk | ROC-AUC 0.574 (CI 0.55–0.60) | LogReg 0.599 | weak but real |
| 9 | Discount sensitivity | ROC-AUC 0.511 | 0.509 | no signal |

**10 of 11 models do not beat a naive baseline.** This is a property of the dataset, not a bug: gradient-boosted
trees also score AUC 0.50 on churn, a customer's past order count has correlation 0.003 with their future order
count, gaps between orders are memoryless (CV 0.92), a customer repeats a category less often than chance, and 94% of
customers show more than one age. Orders appear to be randomly assigned to customer IDs. The only real signal is
order-level: return rate depends on category (Grocery 1.3% vs Fashion 8.3%).

The pipeline is correct and general: point `DATA_PATH` in `src/config.py` at a real transaction dataset with the same
columns (e.g. Online Retail II mapped to this schema) and rerun to get meaningful models. The dashboard shows a
reliability badge next to every prediction so weak models are never over-read.

## Project structure

```
app/streamlit_app.py   role-based customer shop and 7-page retail admin dashboard; What-if includes an
                       all-predictions new-customer cold-start form
app/accounts.py        local SQLite accounts, password hashing, customer orders, returns, and training state
app/retraining.py      background, threshold-triggered training and versioned model activation
app/ui.py              theme CSS, KPI cards, badges, Plotly template
src/features.py        customer snapshot + order-level features and targets
src/models.py          Keras MLP builder, training, evaluation, baselines
src/pipeline.py        trains 11 models, scores customers
src/eda.py             EDA figures and summary
run_pipeline.py        one-command pipeline
models/                11 trained .keras models + scalers
outputs/               customer_scores.csv (all predictions per customer)
reports/               metrics.json, curves.json, eda_summary.json, figures/
```

Dashboard currency symbol: change `CUR` at the top of `app/streamlit_app.py`.
