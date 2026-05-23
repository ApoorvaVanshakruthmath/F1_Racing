"""
F1 Synthetic Data Generator
Produces a statistically realistic dataset for 2015-2024 (10 seasons, ~22 races each = ~4,400 rows)
using real-world-calibrated win rates, DNF rates, and team performance tiers.
Used when FastF1 cannot fetch older seasons.
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)

# ── Real-world calibrated win weights per era ──────────────────────────────
# Each dict maps driver abbr → relative win probability weight
# (not sum-to-1; normalised at runtime per race)

ERA_ROSTERS = {
    # 2015-2016: Mercedes dominant, Ferrari chasing
    2015: {
        "HAM": 10.0, "ROS": 8.0, "VET": 5.0, "RAI": 2.0, "BOT": 0.5,
        "MAS": 0.3, "PER": 0.3, "HUL": 0.2, "SAI": 0.1, "GRO": 0.1,
        "MAG": 0.1, "ALO": 1.0, "BUT": 0.2, "NAS": 0.1, "ERI": 0.1,
        "RIC": 1.5, "VES": 0.0, "KVY": 0.3, "SAI2": 0.1, "NAL": 0.1,
    },
    2016: {
        "HAM": 9.0, "ROS": 9.0, "VET": 4.0, "RAI": 1.5, "BOT": 0.5,
        "RIC": 2.0, "VES": 1.0, "PER": 0.3, "HUL": 0.2, "GRO": 0.1,
        "MAG": 0.1, "BUT": 0.2, "NAS": 0.1, "ERI": 0.1, "PAL": 0.1,
        "SAI": 0.2, "KVY": 0.3, "GUT": 0.05, "WEH": 0.05, "HAR": 0.05,
    },
    # 2017-2018: Mercedes vs Ferrari, Red Bull rising
    2017: {
        "HAM": 9.0, "BOT": 3.0, "VET": 8.0, "RAI": 2.0, "RIC": 3.0,
        "VES": 1.5, "PER": 0.3, "OCO": 0.1, "HUL": 0.2, "SAI": 0.3,
        "GRO": 0.1, "MAG": 0.2, "ALO": 0.5, "VAN": 0.1, "STR": 0.1,
        "WEH": 0.05, "ERI": 0.1, "PAL": 0.1, "KVY": 0.2, "LAN": 0.0,
    },
    2018: {
        "HAM": 10.0, "BOT": 3.0, "VET": 8.0, "RAI": 2.5, "RIC": 3.0,
        "VES": 2.5, "PER": 0.4, "OCO": 0.2, "HUL": 0.3, "MAG": 0.2,
        "ALO": 0.5, "VAN": 0.1, "STR": 0.1, "GAS": 0.1, "LEC": 0.3,
        "HARt": 0.05, "ERI": 0.1, "SIR": 0.05, "SAI": 0.2, "GRO": 0.1,
    },
    # 2019: Mercedes dominant again
    2019: {
        "HAM": 11.0, "BOT": 5.0, "VET": 4.0, "LEC": 2.0, "VES": 3.0,
        "RIC": 1.0, "NOR": 0.3, "SAI": 0.3, "HUL": 0.2, "RAI": 1.0,
        "GIO": 0.1, "PER": 0.4, "STR": 0.1, "GRO": 0.1, "MAG": 0.1,
        "RUS": 0.0, "KUB": 0.1, "GAS": 0.1, "OCO": 0.1, "ALB": 0.1,
    },
    # 2020: Mercedes dominant, Red Bull chasing
    2020: {
        "HAM": 12.0, "BOT": 4.0, "VES": 3.5, "ALB": 0.5, "LEC": 1.5,
        "VET": 1.0, "SAI": 0.5, "PER": 1.5, "NOR": 0.3, "RIC": 0.5,
        "OCO": 0.5, "STR": 0.3, "HUL": 0.2, "GRO": 0.1, "MAG": 0.1,
        "RUS": 0.5, "LAT": 0.0, "GIO": 0.1, "RAI": 0.3, "GAS": 0.1,
    },
    # 2021: VER vs HAM, Red Bull competitive
    2021: {
        "VER": 10.0, "HAM": 9.5, "BOT": 2.0, "PER": 2.5, "LEC": 1.5,
        "SAI": 1.0, "NOR": 1.0, "RIC": 0.5, "ALO": 0.5, "GAS": 0.3,
        "STR": 0.2, "VET": 0.5, "OCO": 0.3, "RUS": 0.3, "TSU": 0.1,
        "LAT": 0.0, "GIO": 0.1, "RAI": 0.3, "MAS2": 0.1, "SCH": 0.1,
    },
    # 2022: VER dominant, Ferrari challenging
    2022: {
        "VER": 12.0, "LEC": 5.0, "SAI": 2.5, "PER": 4.0, "HAM": 2.0,
        "RUS": 2.5, "NOR": 0.5, "ALO": 0.5, "OCO": 0.2, "BOT": 0.5,
        "VET": 0.3, "STR": 0.1, "GAS": 0.2, "TSU": 0.1, "RIC": 0.3,
        "ZHO": 0.0, "MAG": 0.2, "HUL": 0.2, "SCH": 0.1, "LAT": 0.0,
    },
    # 2023: VER record-breaking season
    2023: {
        "VER": 15.0, "PER": 3.5, "ALO": 2.0, "HAM": 1.5, "RUS": 1.0,
        "SAI": 1.5, "LEC": 1.0, "NOR": 0.5, "STR": 0.3, "GAS": 0.3,
        "PIA": 0.5, "OCO": 0.2, "BOT": 0.2, "HUL": 0.2, "TSU": 0.1,
        "ZHO": 0.0, "MAG": 0.1, "SAR": 0.0, "ALB": 0.2, "RIC": 0.3,
    },
    # 2024: Red Bull fading, McLaren/Ferrari rising
    2024: {
        "VER": 8.0, "NOR": 7.0, "LEC": 5.0, "PIA": 5.0, "SAI": 3.0,
        "HAM": 2.5, "RUS": 2.5, "PER": 1.5, "ALO": 0.8, "STR": 0.3,
        "GAS": 0.3, "OCO": 0.3, "TSU": 0.3, "RIC": 0.3, "HUL": 0.2,
        "BOT": 0.2, "MAG": 0.2, "ZHO": 0.1, "ALB": 0.3, "SAR": 0.0,
    },
}

# Team for each driver (per era)
DRIVER_TEAM = {
    # 2015-2016
    "HAM": "Mercedes", "ROS": "Mercedes",
    "VET": "Ferrari",  "RAI": "Ferrari",
    "RIC": "Red Bull Racing", "VES": "Red Bull Racing", "KVY": "Red Bull Racing",
    "BOT": "Williams", "MAS": "Williams",
    "ALO": "McLaren", "BUT": "McLaren",
    "PER": "Force India", "HUL": "Force India",
    "SAI": "Toro Rosso", "SAI2": "Toro Rosso",
    "GRO": "Lotus", "MAG": "Haas F1 Team",
    "NAS": "Lotus", "ERI": "Sauber", "NAL": "Manor",
    "PAL": "Renault", "GUT": "Haas F1 Team", "WEH": "Manor", "HAR": "Manor",
    # 2017+
    "OCO": "Force India", "STR": "Williams", "VAN": "Williams",
    "LEC": "Ferrari", "NOR": "McLaren", "PIA": "McLaren",
    "RUS": "Mercedes", "ALB": "Williams",
    "ANT": "Mercedes", "HAD": "Racing Bulls",
    "GAS": "Alpine", "TSU": "RB",
    "LAT": "Williams", "GIO": "Alfa Romeo",
    "ZHO": "Kick Sauber", "SAR": "Williams",
    "SCH": "Haas F1 Team", "MAS2": "Aston Martin",
    "HUL": "Haas F1 Team", "BOT": "Kick Sauber",
    "LAN": "McLaren", "KUB": "Williams",
    "HARt": "Haas F1 Team", "SIR": "Sauber",
}

TEAM_TIER_MAP = {
    "Red Bull Racing": 1, "Ferrari": 1, "McLaren": 1, "Mercedes": 1,
    "Aston Martin": 2, "Alpine": 2, "Renault": 2,
    "Williams": 3, "Force India": 2, "Racing Point": 2,
    "Toro Rosso": 3, "RB": 3, "Racing Bulls": 3,
    "Haas F1 Team": 4, "Kick Sauber": 4, "Sauber": 4,
    "Alfa Romeo": 3, "Lotus": 3, "Manor": 4,
}

CIRCUITS_BY_ERA = {
    "early": [  # 2015-2018
        "Bahrain", "Australia", "China", "Russia", "Spain", "Monaco",
        "Canada", "Azerbaijan", "Austria", "Britain", "Hungary",
        "Belgium", "Italy", "Singapore", "Malaysia", "Japan",
        "United States", "Mexico", "Brazil", "Abu Dhabi",
    ],
    "late": [  # 2019-2024
        "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami",
        "Monaco", "Canada", "Spain", "Austria", "Britain", "Hungary",
        "Belgium", "Netherlands", "Italy", "Azerbaijan", "Singapore",
        "United States", "Mexico", "Brazil", "Las Vegas", "Abu Dhabi",
    ],
}

POINTS_MAP = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}


def generate_dataset(
    years: list = None,
    save_path: str = "data/historical_data.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a full multi-season synthetic dataset."""
    if years is None:
        years = list(range(2015, 2025))

    np.random.seed(seed)
    save_path = os.path.join(BASE_DIR, save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_records = []

    for year in sorted(years):
        roster_weights = ERA_ROSTERS.get(year, ERA_ROSTERS[2024])
        drivers = list(roster_weights.keys())
        weights_raw = np.array([roster_weights[d] for d in drivers], dtype=float)

        n_rounds = {2015:19, 2016:21, 2017:20, 2018:21, 2019:21,
                    2020:17, 2021:22, 2022:22, 2023:22, 2024:24}.get(year, 22)

        circuits = (CIRCUITS_BY_ERA["early"] if year <= 2018
                    else CIRCUITS_BY_ERA["late"])

        season_points = {d: 0.0 for d in drivers}
        season_wins   = {d: 0   for d in drivers}
        driver_history = {d: [] for d in drivers}  # list of finishing positions

        print(f"  {year}: {n_rounds} rounds, {len(drivers)} drivers")

        for rnd in range(1, n_rounds + 1):
            circuit = circuits[(rnd - 1) % len(circuits)]
            is_wet = np.random.random() < 0.12
            track_temp = np.random.normal(40, 9)
            air_temp   = track_temp - np.random.uniform(5, 15)
            humidity   = np.random.uniform(30, 80)

            # Add small random noise to weights each race
            w = weights_raw * np.exp(np.random.normal(0, 0.15, size=len(drivers)))
            w = np.abs(w)
            w = np.maximum(w, 1e-6)  # prevent exact zeros

            # Wet race scrambler — boost experienced drivers
            if is_wet:
                for i, d in enumerate(drivers):
                    if d in ("HAM", "VES", "ALO", "VER"):
                        w[i] *= 1.4
                    elif d in ("LEC", "NOR", "VET", "RIC"):
                        w[i] *= 1.15

            # Form boost: if driver won last 2 races, boost by 15%
            for i, d in enumerate(drivers):
                last3 = driver_history[d][-3:]
                if last3.count(1) >= 2:
                    w[i] *= 1.15
                elif last3.count(1) == 0 and len(last3) == 3:
                    w[i] *= 0.95

            w /= w.sum()

            # Race finishing order
            finishing_order = np.random.choice(len(drivers), size=len(drivers), replace=False, p=w)

            # Qualifying order (correlated ~70% with race weights, 30% random)
            q_w = 0.70 * w + 0.30 * (np.ones(len(drivers)) / len(drivers))
            q_w /= q_w.sum()
            quali_order = np.random.choice(len(drivers), size=len(drivers), replace=False, p=q_w)

            pole_time = 85.0 + np.random.normal(0, 2.0)  # varies by circuit

            for pos_race, didx in enumerate(finishing_order):
                d = drivers[didx]
                team = DRIVER_TEAM.get(d, "Unknown")
                grid_pos = int(np.where(quali_order == didx)[0][0]) + 1

                # Q times (only top 15 have Q2, top 10 have Q3)
                q3s = (pole_time + (grid_pos - 1) * 0.12 + np.random.normal(0, 0.04)
                       if grid_pos <= 10 else np.nan)
                q2s = (pole_time + 0.6 + np.random.normal(0, 0.05)
                       if 10 < grid_pos <= 15 else np.nan)
                q1s = (pole_time + 1.2 + np.random.normal(0, 0.06)
                       if grid_pos > 15 else np.nan)

                fp1 = (q3s or q2s or q1s or pole_time + 1) + np.random.uniform(0.3, 1.2)
                fp2 = (q3s or q2s or q1s or pole_time + 1) + np.random.uniform(0.1, 0.8)
                fp3 = (q3s or q2s or q1s or pole_time + 1) + np.random.uniform(0.05, 0.5)

                dnf = np.random.random() < 0.045
                finish_pos = pos_race + 1 if not dnf else 20
                pts = POINTS_MAP.get(finish_pos, 0) if not dnf else 0
                status = "Retired" if dnf else ("Finished" if finish_pos <= 18 else "+1 Lap")

                driver_history[d].append(finish_pos)
                season_points[d] += pts
                if finish_pos == 1:
                    season_wins[d] += 1

                # Rolling last-5 stats (using history before this race)
                prev = driver_history[d][:-1][-5:]
                avg_pts_l5 = np.mean([POINTS_MAP.get(p, 0) for p in prev]) if prev else 0
                win_rate_l5 = sum(1 for p in prev if p == 1) / max(len(prev), 1)
                pod_rate_l5 = sum(1 for p in prev if p <= 3) / max(len(prev), 1)
                dnf_rate_l5 = sum(1 for p in prev if p == 20) / max(len(prev), 1)

                all_records.append({
                    "Year":               year,
                    "Round":              rnd,
                    "EventName":          f"{circuit} Grand Prix",
                    "Circuit":            circuit,
                    "DriverNumber":       didx + 1,
                    "Abbreviation":       d,
                    "FullName":           d,
                    "TeamName":           team,
                    "GridPosition":       grid_pos,
                    "Position":           str(finish_pos) if not dnf else "R",
                    "Points":             pts,
                    "Status":             status,
                    "QualifyingPosition": grid_pos,
                    "Q1_seconds":         q1s,
                    "Q2_seconds":         q2s,
                    "Q3_seconds":         q3s,
                    "FP1_best_sec":       fp1,
                    "FP2_best_sec":       fp2,
                    "FP3_best_sec":       fp3,
                    "AirTemp_mean":       air_temp,
                    "TrackTemp_mean":     track_temp,
                    "Humidity_mean":      humidity,
                    "WindSpeed_mean":     np.random.uniform(0, 25),
                    "Rainfall":           int(is_wet),
                    "PointsClean":        pts,
                    "DNF":                int(dnf),
                    "AvgPoints_L5":       avg_pts_l5,
                    "WinRate_L5":         win_rate_l5,
                    "PodiumRate_L5":      pod_rate_l5,
                    "DNFRate_L5":         dnf_rate_l5,
                    "SeasonWins":         season_wins[d],
                    "SeasonPoints":       season_points[d],
                })

    df = pd.DataFrame(all_records)
    df.to_csv(save_path, index=False)
    print(f"\n✅ Synthetic dataset: {len(df):,} records | "
          f"{df['Year'].nunique()} seasons | "
          f"{df[['Year','Round']].drop_duplicates().shape[0]} races")
    return df


if __name__ == "__main__":
    import sys
    years = [int(y) for y in sys.argv[1:]] if len(sys.argv) > 1 else list(range(2015, 2025))
    df = generate_dataset(years=years)
    print(df[["Year", "Abbreviation", "TeamName", "Position", "Points"]].head(10))
