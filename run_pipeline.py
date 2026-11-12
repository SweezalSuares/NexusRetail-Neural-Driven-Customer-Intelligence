"""Run the whole project: EDA -> feature building -> training -> evaluation -> scoring.

Usage:  python run_pipeline.py            (everything)
        python run_pipeline.py --skip-eda (skip EDA figures)
"""
import argparse
import json
from src import config as C
from src.eda import run_eda
from src.features import load_data
from src.pipeline import load_models, score_customers, train_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="reuse saved models, only re-score")
    args = ap.parse_args()

    df = load_data()
    print(f"Loaded {len(df):,} orders from {df['customer_id'].nunique():,} customers")
    if not args.skip_eda:
        print("== EDA =="); run_eda(df)
    if not args.skip_train:
        print("== Training 11 MLP models =="); train_all(df)
    print("== Scoring customers ==")
    scores, extra = score_customers(df, load_models())
    mpath = C.REPORTS_DIR / "metrics.json"
    m = json.loads(mpath.read_text()); m["meta"].update(extra); mpath.write_text(json.dumps(m, indent=2))
    print(f"Saved {len(scores):,} customer scores to {C.OUT_DIR / 'customer_scores.csv'}")
    print("Done. Launch the app with:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
