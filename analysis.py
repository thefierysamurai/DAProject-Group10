#EV Charging Duration Analysis
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DURATION = "Charging Duration (hours)"
CHARGER = "Charger Type"
ENERGY = "Energy Consumed (kWh)"
ORDER = ["Level 1", "Level 2", "DC Fast Charger"]
CORR_VARS = [
    "Energy Consumed (kWh)", "Charging Rate (kW)", "Charging Cost (USD)",
    "Battery Capacity (kWh)", "Temperature (°C)", "State of Charge (Start %)",
    "State of Charge (End %)", "Distance Driven (since last charge) (km)"
]

def main():
    parser = argparse.ArgumentParser(description="Analyze EV charging duration data.")
    parser.add_argument("--data", type=Path, default=Path("ev_charging_patterns.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if not args.data.exists():
        raise FileNotFoundError("Download the Kaggle CSV and pass its path using --data.")

    out = args.output
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip()
    df[DURATION] = pd.to_numeric(df[DURATION], errors="coerce")
    df[CHARGER] = df[CHARGER].astype("string").str.strip()
    analysis = df.dropna(subset=[DURATION, CHARGER]).copy()

    # Data quality and missing values
    missing = df.isna().sum()
    missing_summary = pd.DataFrame({
        "Variable": missing.index,
        "Missing values": missing.values,
        "Missing percent": (missing.values / len(df) * 100).round(2)
    })
    missing_summary = missing_summary[missing_summary["Missing values"] > 0]
    missing_summary.to_csv(out / "missing_values_summary.csv", index=False)
    quality = pd.DataFrame([
        ["Records", len(df)], ["Variables", df.shape[1]],
        ["Exact duplicate rows", int(df.duplicated().sum())],
        ["Rows with any missing value", int(df.isna().any(axis=1).sum())],
        ["Rows used in main comparison", len(analysis)],
        ["Missing duration values", int(df[DURATION].isna().sum())],
        ["Missing charger type values", int(df[CHARGER].isna().sum())]
    ], columns=["Check", "Result"])
    quality.to_csv(out / "data_quality_summary.csv", index=False)

    # Descriptive statistics by charger type
    summary = analysis.groupby(CHARGER)[DURATION].agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).reindex(ORDER).rename(columns={
        "count":"n", "mean":"mean_hours", "median":"median_hours",
        "std":"sd_hours", "min":"min_hours", "max":"max_hours"
    })
    summary.round(4).to_csv(out / "duration_summary_by_charger.csv")

    # Z-score example, using sample standard deviation
    x = analysis[DURATION]
    mean_x, sd_x = x.mean(), x.std(ddof=1)
    first_x = float(x.iloc[0])
    z = (first_x - mean_x) / sd_x
    pd.DataFrame([
        ["Duration mean (hours)", mean_x],
        ["Duration sample SD (hours)", sd_x],
        ["First analysed duration (hours)", first_x],
        ["First analysed duration z-score", z]
    ], columns=["Measure", "Value"]).to_csv(out / "z_score_summary.csv", index=False)

    # One-way ANOVA and Kruskal-Wallis test
    groups = [analysis.loc[analysis[CHARGER] == name, DURATION].dropna().to_numpy()
              for name in ORDER]
    if any(len(g) == 0 for g in groups):
        raise ValueError("A charger category is empty; check the CSV category labels.")
    a = stats.f_oneway(*groups)
    k = stats.kruskal(*groups)
    tests = pd.DataFrame([
        ["One-way ANOVA", a.statistic, a.pvalue,
         "Reject H0" if a.pvalue < .05 else "Fail to reject H0"],
        ["Kruskal-Wallis", k.statistic, k.pvalue,
         "Reject H0" if k.pvalue < .05 else "Fail to reject H0"]
    ], columns=["Test", "Statistic", "p_value", "Decision_at_0.05"])
    tests.to_csv(out / "hypothesis_test_results.csv", index=False)

    # Pearson correlations with duration; pairwise complete rows only
    rows = []
    for col in CORR_VARS:
        if col in df.columns:
            pair = df[[col, DURATION]].dropna()
            r = pair[col].corr(pair[DURATION]) if len(pair) >= 2 else np.nan
            rows.append([col, r, len(pair)])
    pd.DataFrame(rows, columns=[
        "Variable", "Pearson_r_with_duration", "Complete_pairs"
    ]).to_csv(out / "duration_correlations.csv", index=False)

    # Figure 1: missing values
    miss = missing[missing > 0].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(miss):
        miss.plot(kind="bar", ax=ax)
        ax.set_ylabel("Missing records")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.text(.5, .5, "No missing values found", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Missing values by variable")
    fig.tight_layout()
    fig.savefig(figdir / "missing_values.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: boxplot of duration by charger category
    values = [analysis.loc[analysis[CHARGER] == name, DURATION].dropna().to_numpy()
              for name in ORDER]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(values, tick_labels=ORDER)
    ax.set_title("Charging duration by charger category")
    ax.set_ylabel("Charging duration (hours)")
    ax.set_xlabel("Charger type")
    ax.tick_params(axis="x", rotation=8)
    fig.tight_layout()
    fig.savefig(figdir / "duration_boxplot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: energy consumed vs duration; incomplete pairs omitted from this chart
    scatter = df[[ENERGY, DURATION, CHARGER]].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ORDER:
        s = scatter[scatter[CHARGER] == name]
        ax.scatter(s[ENERGY], s[DURATION], label=name, alpha=.6, s=14)
    ax.set_title("Energy consumed vs. charging duration")
    ax.set_xlabel("Energy consumed (kWh)")
    ax.set_ylabel("Charging duration (hours)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "energy_duration_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\\nEV CHARGING DURATION ANALYSIS")
    print(f"Records: {len(df)} | Variables: {df.shape[1]}")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")
    print(f"Rows used for main comparison: {len(analysis)}")
    print("\\nMissing values:\\n", missing_summary.to_string(index=False))
    print("\\nDuration summary by charger (hours):\\n", summary.round(3).to_string())
    print(f"\\nZ-score example: mean={mean_x:.3f}, sample SD={sd_x:.3f}, "
          f"first duration={first_x:.3f}, z={z:.3f}")
    print("\\nHypothesis tests (alpha=0.05):\\n", tests.to_string(index=False))
    print("\\nPearson correlations with duration:\\n",
          pd.read_csv(out / "duration_correlations.csv").round(3).to_string(index=False))
    print(f"\\nEnergy-duration scatterplot complete pairs: {len(scatter)}")
    print(f"\\nOutputs saved in: {out.resolve()}")

if __name__ == "__main__":
    main()
