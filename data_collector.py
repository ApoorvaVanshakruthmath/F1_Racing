"""
F1 Winner Predictor - Data Collection Module
Pulls race results, qualifying & practice from FastF1 (2018–2024).
Falls back to Jolpica/Ergast for pre-2018 if needed.
"""

import fastf1
import pandas as pd
import numpy as np
import os
import warnings
import logging
import time

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ── per-year round count (for progress reporting) ──────────────────────────
ROUNDS_PER_YEAR = {
    2015: 19, 2016: 21, 2017: 20, 2018: 21,
    2019: 21, 2020: 17, 2021: 22, 2022: 22,
    2023: 22, 2024: 24,
}


def _safe_lap_seconds(x):
    try:
        if pd.notna(x) and hasattr(x, "total_seconds"):
            return x.total_seconds()
        return float(x) if pd.notna(x) else np.nan
    except Exception:
        return np.nan


def fetch_race_results(year: int, round_number: int) -> pd.DataFrame:
    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(telemetry=False, weather=False, messages=False)
        r = session.results
        if r is None or r.empty:
            return pd.DataFrame()
        df = r[["DriverNumber", "Abbreviation", "FullName", "TeamName",
                "GridPosition", "Position", "Points", "Status", "Time"]].copy()
        df["Year"]      = year
        df["Round"]     = round_number
        df["EventName"] = session.event["EventName"]
        df["Circuit"]   = session.event["Location"]
        return df
    except Exception as e:
        logger.debug(f"Race {year} R{round_number}: {e}")
        return pd.DataFrame()


def fetch_qualifying_results(year: int, round_number: int) -> pd.DataFrame:
    try:
        session = fastf1.get_session(year, round_number, "Q")
        session.load(telemetry=False, weather=False, messages=False)
        r = session.results
        if r is None or r.empty:
            return pd.DataFrame()
        r = r.copy()
        r.rename(columns={"Position": "QualifyingPosition"}, inplace=True)
        for col in ["Q1", "Q2", "Q3"]:
            r[f"{col}_seconds"] = r[col].apply(_safe_lap_seconds) if col in r.columns else np.nan
        r["Year"]  = year
        r["Round"] = round_number
        keep = ["DriverNumber", "Abbreviation", "QualifyingPosition",
                "Q1_seconds", "Q2_seconds", "Q3_seconds", "Year", "Round"]
        return r[[c for c in keep if c in r.columns]]
    except Exception as e:
        logger.debug(f"Quali {year} R{round_number}: {e}")
        return pd.DataFrame()


def fetch_practice_results(year: int, round_number: int) -> pd.DataFrame:
    frames = []
    for fp in ["FP1", "FP2", "FP3"]:
        try:
            s = fastf1.get_session(year, round_number, fp)
            s.load(telemetry=False, weather=False, messages=False)
            laps = s.laps.pick_quicklaps() if hasattr(s, "laps") else pd.DataFrame()
            if laps is None or laps.empty:
                continue
            best = (laps.groupby("Driver")["LapTime"].min().reset_index()
                    .rename(columns={"Driver": "Abbreviation", "LapTime": f"{fp}_best"}))
            best[f"{fp}_best_sec"] = best[f"{fp}_best"].apply(_safe_lap_seconds)
            frames.append(best[["Abbreviation", f"{fp}_best_sec"]])
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()
    from functools import reduce
    merged = reduce(lambda a, b: pd.merge(a, b, on="Abbreviation", how="outer"), frames)
    merged["Year"]  = year
    merged["Round"] = round_number
    return merged


def fetch_weather_data(year: int, round_number: int) -> dict:
    try:
        s = fastf1.get_session(year, round_number, "R")
        s.load(telemetry=False, weather=True, messages=False)
        w = s.weather_data
        if w is not None and not w.empty:
            return {
                "AirTemp_mean":   w["AirTemp"].mean(),
                "TrackTemp_mean": w["TrackTemp"].mean(),
                "Humidity_mean":  w["Humidity"].mean(),
                "WindSpeed_mean": w["WindSpeed"].mean(),
                "Rainfall":       int(w["Rainfall"].any()),
            }
    except Exception:
        pass
    return {}


def build_historical_dataset(
    years: list = None,
    save_path: str = "data/historical_data.csv",
) -> pd.DataFrame:
    """
    Collect race + quali + practice data for all requested years via FastF1.
    If a round fails silently, it is skipped and collection continues.
    """
    if years is None:
        years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

    save_path = os.path.join(os.path.dirname(__file__), save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_records = []
    total_rounds = sum(ROUNDS_PER_YEAR.get(y, 22) for y in years)
    done = 0

    for year in sorted(years):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            rounds = schedule["RoundNumber"].tolist()
        except Exception as e:
            logger.warning(f"Could not get schedule for {year}: {e}")
            rounds = list(range(1, ROUNDS_PER_YEAR.get(year, 22) + 1))

        logger.info(f"\n📅 {year} — {len(rounds)} rounds")

        for rnd in rounds:
            done += 1
            pct = done / total_rounds * 100
            logger.info(f"  [{pct:5.1f}%] {year} Round {rnd:02d}")

            race = fetch_race_results(year, rnd)
            if race.empty:
                logger.debug(f"    Skipping — no race results")
                continue

            quali    = fetch_qualifying_results(year, rnd)
            practice = fetch_practice_results(year, rnd)
            weather  = fetch_weather_data(year, rnd)

            df = race.copy()
            if not quali.empty:
                df = df.merge(
                    quali.drop(columns=["Year", "Round"], errors="ignore"),
                    on=["DriverNumber", "Abbreviation"], how="left"
                )
            if not practice.empty:
                df = df.merge(
                    practice.drop(columns=["Year", "Round"], errors="ignore"),
                    on="Abbreviation", how="left"
                )
            for k, v in weather.items():
                df[k] = v

            all_records.append(df)

    if not all_records:
        raise RuntimeError("No data collected! Check FastF1 connectivity.")

    full_df = pd.concat(all_records, ignore_index=True)
    full_df.to_csv(save_path, index=False)
    logger.info(f"\n✅ Saved {len(full_df):,} records ({full_df['Year'].nunique()} seasons, "
                f"{full_df[['Year','Round']].drop_duplicates().shape[0]} races) → {save_path}")
    return full_df


def append_years_to_existing(
    new_years: list,
    existing_path: str = "data/historical_data.csv",
) -> pd.DataFrame:
    """
    Add new years to an existing dataset without re-fetching what's already there.
    Deduplicates by (Year, Round, Abbreviation).
    """
    existing_path = os.path.join(os.path.dirname(__file__), existing_path)
    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path)
        already_have = set(existing["Year"].unique().tolist())
        fetch_years = [y for y in new_years if int(y) not in already_have]
        if not fetch_years:
            logger.info(f"All years {new_years} already in dataset. Nothing to fetch.")
            return existing
        logger.info(f"Fetching new years: {fetch_years}")
        new_df = build_historical_dataset(years=fetch_years, save_path="data/_tmp_new.csv")
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["Year", "Round", "Abbreviation"], inplace=True)
        combined.to_csv(existing_path, index=False)
        logger.info(f"✅ Combined dataset: {len(combined):,} records")
        return combined
    else:
        return build_historical_dataset(years=new_years, save_path=existing_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        years = [int(y) for y in sys.argv[1:]]
    else:
        years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    print(f"Collecting data for: {years}")
    df = build_historical_dataset(years=years)
    print(f"Final dataset: {df.shape}")