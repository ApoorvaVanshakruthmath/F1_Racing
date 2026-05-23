"""
F1 Winner Predictor - Exploratory Data Analysis Module
Generates key visualizations and statistical insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#e10600",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "grid.color": "#333355",
    "grid.alpha": 0.5,
    "font.family": "monospace",
})

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_win_rate_by_grid(df: pd.DataFrame):
    """Win rate by starting grid position."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("F1 Win Rate Analysis", color="#e10600", fontsize=16, fontweight="bold")

    # Win rate by grid position (top 10)
    grid_win = df[df["GridPosition_clean"] <= 10].groupby("GridPosition_clean")["Winner"].mean() * 100
    ax = axes[0]
    bars = ax.bar(grid_win.index, grid_win.values, color=["#e10600" if i == 1 else "#ff6b35" if i <= 3 else "#4a90d9"
                                                            for i in grid_win.index])
    ax.set_title("Win % by Grid Position", color="white")
    ax.set_xlabel("Grid Position")
    ax.set_ylabel("Win Rate (%)")
    for bar, val in zip(bars, grid_win.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=8, color="white")

    # Podium rate by grid position
    podium_rate = df[df["GridPosition_clean"] <= 10].groupby("GridPosition_clean")["Top3"].mean() * 100
    ax2 = axes[1]
    ax2.bar(podium_rate.index, podium_rate.values,
            color=["#ffd700" if i <= 3 else "#c0c0c0" for i in podium_rate.index])
    ax2.set_title("Podium Rate (Top 3) by Grid Position", color="white")
    ax2.set_xlabel("Grid Position")
    ax2.set_ylabel("Podium Rate (%)")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "win_rate_grid.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_driver_performance(df: pd.DataFrame):
    """Driver win rates and average points in the dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Driver Performance Overview", color="#e10600", fontsize=16, fontweight="bold")

    driver_stats = df.groupby("Abbreviation").agg(
        TotalWins=("Winner", "sum"),
        TotalRaces=("Winner", "count"),
        AvgPoints=("PointsClean", "mean"),
    ).reset_index()
    driver_stats["WinRate"] = driver_stats["TotalWins"] / driver_stats["TotalRaces"] * 100
    driver_stats = driver_stats[driver_stats["TotalRaces"] >= 5].sort_values("WinRate", ascending=False).head(15)

    # Win rate
    ax = axes[0]
    colors = ["#e10600" if r > 20 else "#ff6b35" if r > 10 else "#4a90d9" for r in driver_stats["WinRate"]]
    ax.barh(driver_stats["Abbreviation"], driver_stats["WinRate"], color=colors)
    ax.set_title("Win Rate by Driver (%)", color="white")
    ax.set_xlabel("Win Rate (%)")
    ax.invert_yaxis()

    # Avg points
    ax2 = axes[1]
    driver_pts = df.groupby("Abbreviation")["PointsClean"].mean().sort_values(ascending=False).head(15)
    ax2.barh(driver_pts.index, driver_pts.values,
             color=["#ffd700" if v > 15 else "#c0c0c0" for v in driver_pts.values])
    ax2.set_title("Average Points per Race", color="white")
    ax2.set_xlabel("Average Points")
    ax2.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "driver_performance.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: list):
    """Feature correlation heatmap."""
    available = [c for c in feature_cols if c in df.columns]
    corr_df = df[available + ["Winner"]].dropna()
    corr = corr_df.corr()

    fig, ax = plt.subplots(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax, cmap="RdYlGn", center=0,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.5, linecolor="#0f0f1a",
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", color="#e10600", fontsize=14, pad=20)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_feature_importance_preview(df: pd.DataFrame, feature_cols: list):
    """Quick correlation-based feature importance with target."""
    available = [c for c in feature_cols if c in df.columns]
    corr = df[available + ["Winner"]].dropna().corr()["Winner"].drop("Winner")
    corr_sorted = corr.abs().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ["#e10600" if v > 0.2 else "#ff6b35" if v > 0.1 else "#4a90d9" for v in corr_sorted]
    ax.barh(corr_sorted.index, corr_sorted.values, color=colors)
    ax.set_title("Feature Correlation with Winner (|r|)", color="#e10600", fontsize=14)
    ax.set_xlabel("|Correlation|")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance_preview.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_team_performance(df: pd.DataFrame):
    """Team performance overview."""
    team_stats = df.groupby("TeamName").agg(
        Wins=("Winner", "sum"),
        Races=("Winner", "count"),
        AvgPoints=("PointsClean", "mean"),
    ).reset_index()
    team_stats = team_stats[team_stats["Races"] >= 10].sort_values("AvgPoints", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#e10600", "#dc0000", "#ff8700", "#00d2be", "#1e3a5f",
              "#358c75", "#0067ff", "#900000", "#ffffff", "#006f62"][:len(team_stats)]
    bars = ax.bar(team_stats["TeamName"], team_stats["AvgPoints"], color=colors[:len(team_stats)])
    ax.set_title("Average Points per Race by Team", color="#e10600", fontsize=14)
    ax.set_xticklabels(team_stats["TeamName"], rotation=30, ha="right")
    for bar, wins in zip(bars, team_stats["Wins"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{int(wins)}W", ha="center", fontsize=8, color="white")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "team_performance.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_wet_vs_dry(df: pd.DataFrame):
    """Wet vs dry race win rate comparison."""
    if "IsWetRace" not in df.columns:
        return
    wet_dry = df.groupby(["Abbreviation", "IsWetRace"])["Winner"].mean().unstack(fill_value=0)
    wet_dry.columns = ["Dry", "Wet"]
    wet_dry = wet_dry[(wet_dry["Dry"] > 0.05) | (wet_dry["Wet"] > 0.05)].sort_values("Dry", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(wet_dry))
    w = 0.35
    ax.bar(x - w/2, wet_dry["Dry"] * 100, w, label="Dry", color="#4a90d9")
    ax.bar(x + w/2, wet_dry["Wet"] * 100, w, label="Wet", color="#00d2be")
    ax.set_xticks(x)
    ax.set_xticklabels(wet_dry.index, rotation=30)
    ax.set_title("Win Rate: Wet vs Dry Races by Driver", color="#e10600", fontsize=14)
    ax.set_ylabel("Win Rate (%)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "wet_vs_dry.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def run_full_eda(df: pd.DataFrame, feature_cols: list):
    """Run all EDA plots."""
    print("\n🏎️  Running Full EDA...\n")
    plot_win_rate_by_grid(df)
    plot_driver_performance(df)
    plot_team_performance(df)
    plot_wet_vs_dry(df)
    plot_correlation_heatmap(df, feature_cols)
    plot_feature_importance_preview(df, feature_cols)
    print(f"\n✅ All plots saved to: {PLOTS_DIR}")

    # Summary stats
    print("\n── Dataset Summary ──────────────────────────")
    print(f"Total records : {len(df):,}")
    print(f"Seasons       : {sorted(df['Year'].unique())}")
    print(f"Drivers       : {df['Abbreviation'].nunique()}")
    print(f"Teams         : {df['TeamName'].nunique()}")
    print(f"Circuits      : {df['Circuit'].nunique() if 'Circuit' in df.columns else 'N/A'}")
    print(f"Total winners : {df['Winner'].sum():,}")
    print(f"Wet races     : {df['IsWetRace'].sum() if 'IsWetRace' in df.columns else 'N/A'}")
    print("─────────────────────────────────────────────")


if __name__ == "__main__":
    from feature_engineering import get_feature_columns
    path = os.path.join(os.path.dirname(__file__), "data/engineered_data.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        run_full_eda(df, get_feature_columns())
    else:
        print("Run feature_engineering.py first.")
