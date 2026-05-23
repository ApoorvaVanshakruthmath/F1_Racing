"""
F1 Winner Predictor - Feature Engineering Module
Creates ML-ready features from raw F1 data:
- Driver/team historical performance stats
- Circuit-specific win rates
- Qualifying gap features
- Momentum & form indicators
- Weather interaction features
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── Driver & Team encoding ──────────────────────────────────────────────────

DRIVER_EXP = {
    "VER": 10, "HAM": 18, "LEC": 7, "SAI": 9, "NOR": 6,
    "PIA": 3,  "RUS": 6,  "ALO": 22, "STR": 8, "GAS": 8,
    "OCO": 7,  "HUL": 15, "TSU": 5,  "RIC": 12, "BOT": 12,
    "ZHO": 3,  "MAG": 10, "ALB": 6,  "SAR": 2,  "LAW": 2,
    "BEA": 1,  "ANT": 1,  "HAD": 1,  "DOO": 1,  "BOR": 1,
}

TEAM_TIER = {
    "Red Bull Racing": 1, "Ferrari": 1, "McLaren": 1,
    "Mercedes": 2, "Aston Martin": 2,
    "Alpine": 3, "Williams": 3, "RB": 3,
    "Kick Sauber": 4, "Haas F1 Team": 4,
}


def clean_position(pos):
    """Convert position to numeric, NaN for DNFs."""
    try:
        p = float(pos)
        return p if 1 <= p <= 20 else np.nan
    except (TypeError, ValueError):
        return np.nan


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add driver experience and team tier features."""
    df = df.copy()
    df["Position_clean"] = df["Position"].apply(clean_position)
    df["GridPosition_clean"] = pd.to_numeric(df["GridPosition"], errors="coerce")
    df["DriverExp"] = df["Abbreviation"].map(DRIVER_EXP).fillna(3)
    df["TeamTier"] = df["TeamName"].map(TEAM_TIER).fillna(3)
    df["IsTopTeam"] = (df["TeamTier"] == 1).astype(int)
    return df


def add_qualifying_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer qualifying-based features per race."""
    df = df.copy()
    if "QualifyingPosition" in df.columns:
        df["QualifyingPosition"] = pd.to_numeric(df["QualifyingPosition"], errors="coerce")
    else:
        df["QualifyingPosition"] = df["GridPosition_clean"]

    # Best Q time (Q3 > Q2 > Q1 fallback)
    for col in ["Q3_seconds", "Q2_seconds", "Q1_seconds"]:
        if col not in df.columns:
            df[col] = np.nan

    df["BestQTime"] = df["Q3_seconds"].fillna(df["Q2_seconds"]).fillna(df["Q1_seconds"])

    # Gap to pole per race
    def gap_to_pole(group):
        pole_time = group["BestQTime"].min()
        group["GapToPole_sec"] = group["BestQTime"] - pole_time
        group["GapToPole_pct"] = group["GapToPole_sec"] / pole_time * 100
        return group

    df = df.groupby(["Year", "Round"], group_keys=False).apply(gap_to_pole)

    # Front row / top 3 / top 10 flags
    df["IsPolePosition"] = (df["QualifyingPosition"] == 1).astype(int)
    df["IsFrontRow"] = (df["QualifyingPosition"] <= 2).astype(int)
    df["IsTop3Grid"] = (df["QualifyingPosition"] <= 3).astype(int)
    df["IsTop10Grid"] = (df["QualifyingPosition"] <= 10).astype(int)
    df["GridPositionInv"] = 21 - df["QualifyingPosition"].fillna(20)  # higher = better start

    return df


def add_practice_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer practice session features."""
    df = df.copy()
    for col in ["FP1_best_sec", "FP2_best_sec", "FP3_best_sec"]:
        if col not in df.columns:
            df[col] = np.nan

    # Best practice time across sessions
    df["BestPracticeTime"] = df[["FP1_best_sec", "FP2_best_sec", "FP3_best_sec"]].min(axis=1)

    # Practice pace rank per race
    def rank_practice(group):
        group["PracticePaceRank"] = group["BestPracticeTime"].rank(ascending=True, method="min")
        best_time = group["BestPracticeTime"].min()
        group["PracticeGapToBest"] = group["BestPracticeTime"] - best_time
        return group

    df = df.groupby(["Year", "Round"], group_keys=False).apply(rank_practice)
    df["IsTopPracticePace"] = (df["PracticePaceRank"] <= 3).astype(int)
    return df


def add_driver_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling form features — recent race performance per driver."""
    df = df.sort_values(["Abbreviation", "Year", "Round"]).copy()

    # Points scored (0 for DNF)
    df["PointsClean"] = pd.to_numeric(df["Points"], errors="coerce").fillna(0)
    df["DNF"] = df["Status"].apply(
        lambda x: 0 if str(x).strip() in ["Finished", "+1 Lap", "+2 Laps"] else 1
    )
    df["Win"] = (df["Position_clean"] == 1).astype(int)
    df["Podium"] = (df["Position_clean"] <= 3).astype(int)
    df["PointsFinish"] = (df["Position_clean"] <= 10).astype(int)

    for col, feat in [
        ("PointsClean", "AvgPoints_L5"),
        ("Win", "WinRate_L5"),
        ("Podium", "PodiumRate_L5"),
        ("DNF", "DNFRate_L5"),
    ]:
        df[feat] = (
            df.groupby("Abbreviation")[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Positions gained/lost from grid
    df["PositionsGained"] = df["GridPosition_clean"] - df["Position_clean"]
    df["AvgPositionsGained_L5"] = (
        df.groupby("Abbreviation")["PositionsGained"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # Season cumulative stats
    df["SeasonWins"] = df.groupby(["Abbreviation", "Year"])["Win"].cumsum().shift(1).fillna(0)
    df["SeasonPoints"] = df.groupby(["Abbreviation", "Year"])["PointsClean"].cumsum().shift(1).fillna(0)

    return df


def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add circuit-specific win rate per driver."""
    df = df.copy()
    if "Circuit" not in df.columns:
        df["Circuit"] = "Unknown"

    circuit_wins = (
        df.groupby(["Abbreviation", "Circuit"])["Win"]
        .sum()
        .reset_index()
        .rename(columns={"Win": "CircuitWins"})
    )
    circuit_starts = (
        df.groupby(["Abbreviation", "Circuit"])["Win"]
        .count()
        .reset_index()
        .rename(columns={"Win": "CircuitStarts"})
    )
    circuit_stats = circuit_wins.merge(circuit_starts, on=["Abbreviation", "Circuit"])
    circuit_stats["CircuitWinRate"] = circuit_stats["CircuitWins"] / circuit_stats["CircuitStarts"]

    df = df.merge(circuit_stats[["Abbreviation", "Circuit", "CircuitWinRate", "CircuitWins"]],
                  on=["Abbreviation", "Circuit"], how="left")
    df["CircuitWinRate"] = df["CircuitWinRate"].fillna(0)
    df["CircuitWins"] = df["CircuitWins"].fillna(0)
    return df


def add_team_form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling team-level performance features."""
    df = df.sort_values(["TeamName", "Year", "Round"]).copy()
    team_round = (
        df.groupby(["TeamName", "Year", "Round"])["PointsClean"]
        .sum()
        .reset_index()
        .rename(columns={"PointsClean": "TeamRoundPoints"})
    )
    team_round["TeamAvgPoints_L5"] = (
        team_round.groupby("TeamName")["TeamRoundPoints"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    df = df.merge(team_round[["TeamName", "Year", "Round", "TeamAvgPoints_L5"]],
                  on=["TeamName", "Year", "Round"], how="left")
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Process and encode weather features."""
    df = df.copy()
    for col in ["AirTemp_mean", "TrackTemp_mean", "Humidity_mean", "WindSpeed_mean", "Rainfall"]:
        if col not in df.columns:
            df[col] = np.nan

    df["Rainfall"] = df["Rainfall"].fillna(0).astype(int)
    df["IsWetRace"] = df["Rainfall"]
    df["TrackTemp_mean"] = df["TrackTemp_mean"].fillna(df["TrackTemp_mean"].median())
    df["AirTemp_mean"] = df["AirTemp_mean"].fillna(df["AirTemp_mean"].median())
    # Hot track = tire deg risk
    df["HighTireStress"] = (df["TrackTemp_mean"] > 45).astype(int)
    return df


def add_championship_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Add championship standings gap features."""
    df = df.copy()
    season_pts = df.groupby(["Year", "Round", "Abbreviation"])["SeasonPoints"].max().reset_index()

    def championship_gap(group):
        max_pts = group["SeasonPoints"].max()
        group["ChampionshipGap"] = max_pts - group["SeasonPoints"]
        group["IsChampionshipLeader"] = (group["SeasonPoints"] == max_pts).astype(int)
        return group

    season_pts = season_pts.groupby(["Year", "Round"], group_keys=False).apply(championship_gap)
    df = df.merge(
        season_pts[["Year", "Round", "Abbreviation", "ChampionshipGap", "IsChampionshipLeader"]],
        on=["Year", "Round", "Abbreviation"], how="left"
    )
    df["ChampionshipGap"] = df["ChampionshipGap"].fillna(200)
    df["IsChampionshipLeader"] = df["IsChampionshipLeader"].fillna(0)
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary winner target and position target."""
    df = df.copy()
    df["Winner"] = (df["Position_clean"] == 1).astype(int)
    df["Top3"] = (df["Position_clean"] <= 3).astype(int)
    return df


def get_feature_columns() -> list:
    """Return the final list of feature columns used for modeling."""
    return [
        # Qualifying
        "QualifyingPosition", "GridPositionInv", "GapToPole_sec", "GapToPole_pct",
        "IsPolePosition", "IsFrontRow", "IsTop3Grid", "IsTop10Grid", "BestQTime",
        # Practice
        "PracticePaceRank", "PracticeGapToBest", "IsTopPracticePace",
        # Driver form
        "AvgPoints_L5", "WinRate_L5", "PodiumRate_L5", "DNFRate_L5",
        "AvgPositionsGained_L5", "SeasonWins", "SeasonPoints",
        # Circuit
        "CircuitWinRate", "CircuitWins",
        # Team
        "TeamTier", "IsTopTeam", "TeamAvgPoints_L5",
        # Driver
        "DriverExp",
        # Weather
        "IsWetRace", "HighTireStress", "TrackTemp_mean", "AirTemp_mean",
        # Championship
        "ChampionshipGap", "IsChampionshipLeader",
    ]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_basic_features(df)
    df = add_qualifying_features(df)
    df = add_practice_features(df)
    df = add_driver_form_features(df)
    df = add_circuit_features(df)
    df = add_team_form_features(df)
    df = add_weather_features(df)
    df = add_championship_pressure(df)
    df = create_target(df)
    return df


if __name__ == "__main__":
    import os
    raw_path = os.path.join(os.path.dirname(__file__), "data/historical_data.csv")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        engineered = build_features(df)
        out_path = raw_path.replace("historical_data.csv", "engineered_data.csv")
        engineered.to_csv(out_path, index=False)
        print(f"Engineered dataset saved: {engineered.shape}")
        print(engineered[get_feature_columns() + ["Winner"]].describe())
    else:
        print("Run data_collector.py first to build the historical dataset.")
