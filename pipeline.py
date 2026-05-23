"""
F1 Winner Predictor - Master Pipeline
End-to-end workflow: collect → engineer → EDA → train → predict

Modes:
  full        Collect real FastF1 data + full pipeline
  append      Add new years to existing dataset, then retrain
  synthetic   Generate synthetic 2015-2024 data (no internet needed)
  train       Re-engineer + retrain on existing data
  predict     Live prediction only (model already trained)
  demo        Synthetic data + train + predict (instant, no internet)

Examples:
  python pipeline.py --mode full --years 2018 2019 2020 2021 2022 2023 2024
  python pipeline.py --mode append --years 2018 2019 2020
  python pipeline.py --mode synthetic --years 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024
  python pipeline.py --mode predict --year 2025 --round 9
"""

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)


def banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║          🏎️   F1 RACE WINNER PREDICTOR  🏁               ║
║      XGBoost + LightGBM + Logistic Regression            ║
║      FastF1 API  |  2015–2024 Training Data              ║
╚══════════════════════════════════════════════════════════╝
""")


# ── Pipeline steps ─────────────────────────────────────────────────────────

def step_collect_data(years, force=False):
    from data_collector import build_historical_dataset, append_years_to_existing
    print(f"\n{'═'*60}")
    print(f"  STEP 1: DATA COLLECTION  (years={years})")
    print(f"{'═'*60}")
    raw_path = os.path.join(DATA_DIR, "historical_data.csv")

    if os.path.exists(raw_path) and not force:
        existing = pd.read_csv(raw_path)
        existing_years = sorted(set(int(y) for y in existing["Year"].unique()))
        wanted = sorted(set(int(y) for y in years))
        missing = [y for y in wanted if y not in existing_years]
        if not missing:
            print(f"  ✅ All years {wanted} already cached ({len(existing):,} rows). Use --force to re-fetch.")
            return existing
        print(f"  ⚠️  Years {missing} not in cache → fetching incrementally...")
        return append_years_to_existing(missing, existing_path="data/historical_data.csv")

    return build_historical_dataset(years=years, save_path="data/historical_data.csv")


def step_collect_synthetic(years):
    from synthetic_data import generate_dataset
    print(f"\n{'═'*60}")
    print(f"  STEP 1 (SYNTHETIC): GENERATING {len(years)}-SEASON DATASET")
    print(f"  Years: {years}")
    print(f"{'═'*60}")
    return generate_dataset(years=years, save_path="data/historical_data.csv")


def step_feature_engineering(df):
    from feature_engineering import build_features, get_feature_columns
    print(f"\n{'═'*60}")
    print(f"  STEP 2: FEATURE ENGINEERING")
    print(f"{'═'*60}")
    out_path = os.path.join(DATA_DIR, "engineered_data.csv")
    engineered = build_features(df)
    engineered.to_csv(out_path, index=False)
    feat_cols = get_feature_columns()
    available = [c for c in feat_cols if c in engineered.columns]
    total_races = engineered[["Year", "Round"]].drop_duplicates().shape[0]
    print(f"  ✅ {len(engineered):,} records | {total_races} races | {len(available)} features")
    print(f"  📁 Saved: {out_path}")
    return engineered, feat_cols


def step_eda(df, feat_cols):
    from eda import run_full_eda
    print(f"\n{'═'*60}")
    print(f"  STEP 3: EXPLORATORY DATA ANALYSIS")
    print(f"{'═'*60}")
    run_full_eda(df, feat_cols)


def step_train(df, feat_cols):
    from model_trainer import train_and_evaluate
    print(f"\n{'═'*60}")
    print(f"  STEP 4: MODEL TRAINING")
    print(f"{'═'*60}")
    return train_and_evaluate(df, feat_cols)


def step_predict(year, round_number, hist_df=None):
    from live_predictor import predict_race
    print(f"\n{'═'*60}")
    print(f"  STEP 5: LIVE RACE PREDICTION  ({year} R{round_number})")
    print(f"{'═'*60}")
    return predict_race(year, round_number, hist_df=hist_df)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="F1 Race Winner Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "append", "synthetic", "train", "predict", "demo"],
        default="demo",
    )
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=[2021, 2022, 2023, 2024],
        help="Years to collect/generate data for",
    )
    parser.add_argument("--year",  type=int, default=2025, help="Year to predict")
    parser.add_argument("--round", type=int, default=8,    dest="round_number",
                        help="Round number to predict")
    parser.add_argument("--force",  action="store_true", help="Force re-fetch even if cached")
    parser.add_argument("--no-eda", action="store_true", help="Skip EDA plots")
    args = parser.parse_args()

    banner()
    start = time.time()

    # ── DEMO ──────────────────────────────────────────────────────────────
    if args.mode == "demo":
        print("  🎮 DEMO MODE — 10 seasons of synthetic data (no internet needed)")
        years_demo = list(range(2015, 2025))
        df_raw = step_collect_synthetic(years_demo)
        df_eng, feat_cols = step_feature_engineering(df_raw)
        if not args.no_eda:
            step_eda(df_eng, feat_cols)
        step_train(df_eng, feat_cols)
        step_predict(args.year, args.round_number, hist_df=df_eng)

    # ── SYNTHETIC ──────────────────────────────────────────────────────────
    elif args.mode == "synthetic":
        print(f"  🔢 SYNTHETIC MODE — generating {len(args.years)} seasons of data")
        df_raw = step_collect_synthetic(args.years)
        df_eng, feat_cols = step_feature_engineering(df_raw)
        if not args.no_eda:
            step_eda(df_eng, feat_cols)
        step_train(df_eng, feat_cols)
        step_predict(args.year, args.round_number, hist_df=df_eng)

    # ── FULL (real FastF1 data) ────────────────────────────────────────────
    elif args.mode == "full":
        df_raw = step_collect_data(args.years, force=args.force)
        df_eng, feat_cols = step_feature_engineering(df_raw)
        if not args.no_eda:
            step_eda(df_eng, feat_cols)
        step_train(df_eng, feat_cols)
        step_predict(args.year, args.round_number, hist_df=df_eng)

    # ── APPEND (add more years to existing dataset) ────────────────────────
    elif args.mode == "append":
        print(f"  ➕ APPEND MODE — adding years {args.years} to existing dataset")
        raw_path = os.path.join(DATA_DIR, "historical_data.csv")
        if not os.path.exists(raw_path):
            print("  ⚠️  No existing dataset found. Running full collection instead.")
            df_raw = step_collect_data(args.years)
        else:
            existing = pd.read_csv(raw_path)
            existing_years = sorted(set(int(y) for y in existing["Year"].unique()))
            new_years = [y for y in args.years if y not in existing_years]
            if not new_years:
                print(f"  ✅ All requested years {args.years} already present: {existing_years}")
                df_raw = existing
            else:
                from data_collector import build_historical_dataset
                new_data = build_historical_dataset(years=new_years, save_path="data/_tmp.csv")
                df_raw = pd.concat([existing, new_data], ignore_index=True)
                df_raw.drop_duplicates(subset=["Year", "Round", "Abbreviation"], inplace=True)
                df_raw.to_csv(raw_path, index=False)
                print(f"  ✅ Appended {len(new_data):,} new rows → total {len(df_raw):,}")

        df_eng, feat_cols = step_feature_engineering(df_raw)
        if not args.no_eda:
            step_eda(df_eng, feat_cols)
        step_train(df_eng, feat_cols)
        step_predict(args.year, args.round_number, hist_df=df_eng)

    # ── TRAIN (retrain on existing data) ──────────────────────────────────
    elif args.mode == "train":
        raw_path = os.path.join(DATA_DIR, "historical_data.csv")
        if not os.path.exists(raw_path):
            print("❌ No data found. Run --mode full or --mode synthetic first.")
            sys.exit(1)
        df_raw = pd.read_csv(raw_path)
        years_in = sorted(set(int(y) for y in df_raw["Year"].unique()))
        print(f"  📂 Loaded {len(df_raw):,} rows from years {years_in}")
        df_eng, feat_cols = step_feature_engineering(df_raw)
        if not args.no_eda:
            step_eda(df_eng, feat_cols)
        step_train(df_eng, feat_cols)
        step_predict(args.year, args.round_number, hist_df=df_eng)

    # ── PREDICT (just predict) ─────────────────────────────────────────────
    elif args.mode == "predict":
        eng_path = os.path.join(DATA_DIR, "engineered_data.csv")
        hist_df = pd.read_csv(eng_path) if os.path.exists(eng_path) else None
        if hist_df is not None:
            years_in = sorted(set(int(y) for y in hist_df["Year"].unique()))
            print(f"  📂 Using engineered data from years {years_in}")
        step_predict(args.year, args.round_number, hist_df=hist_df)

    elapsed = time.time() - start
    print(f"\n{'═'*60}")
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()