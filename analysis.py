"""
EV CHARGING DURATION ANALYSIS
=============================

This script reproduces the analysis and graphs used in the project report.
Keep the dataset CSV in this folder, then run this file directly:

    python analysis.py

Required input:
    ev_charging_patterns.csv

Output:
    outputs/ (CSV result tables and PNG figures)
"""

# ============================================================
# 0. IMPORT LIBRARIES AND SET UP FILE PATHS
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

DATA_FILE = Path("ev_charging_patterns.csv")
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = OUTPUT_DIR / "figures"

DURATION = "Charging Duration (hours)"
CHARGER = "Charger Type"
ENERGY = "Energy Consumed (kWh)"

CHARGER_ORDER = ["Level 1", "Level 2", "DC Fast Charger"]

CORRELATION_VARIABLES = [
    "Energy Consumed (kWh)",
    "Charging Rate (kW)",
    "Charging Cost (USD)",
    "Battery Capacity (kWh)",
    "Temperature (°C)",
    "State of Charge (Start %)",
    "State of Charge (End %)",
    "Distance Driven (since last charge) (km)",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATASET AND PERFORM BASIC CLEANING
# ============================================================

print("\n" + "=" * 65)
print("1. DATASET LOADING AND BASIC INSPECTION")
print("=" * 65)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find '{DATA_FILE}'. Place ev_charging_patterns.csv "
        "in the same folder as analysis.py."
    )

df = pd.read_csv(DATA_FILE)

# Remove accidental spaces around column names and charger labels.
df.columns = df.columns.str.strip()
df[CHARGER] = df[CHARGER].astype("string").str.strip()
df[DURATION] = pd.to_numeric(df[DURATION], errors="coerce")

print(f"Number of records: {df.shape[0]}")
print(f"Number of variables: {df.shape[1]}")
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head().to_string(index=False))


# ============================================================
# 2. DATA QUALITY: MISSING VALUES AND DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 65)
print("2. DATA QUALITY CHECKS")
print("=" * 65)

missing_counts = df.isna().sum()
missing_summary = pd.DataFrame({
    "Variable": missing_counts.index,
    "Missing values": missing_counts.values,
    "Missing percent": (missing_counts.values / len(df) * 100).round(2),
})
missing_summary = missing_summary[missing_summary["Missing values"] > 0]
missing_summary.to_csv(OUTPUT_DIR / "missing_values_summary.csv", index=False)

duplicate_rows = int(df.duplicated().sum())
rows_with_missing = int(df.isna().any(axis=1).sum())

quality_summary = pd.DataFrame([
    ["Number of records", len(df)],
    ["Number of variables", df.shape[1]],
    ["Exact duplicate rows", duplicate_rows],
    ["Rows with at least one missing value", rows_with_missing],
    ["Missing duration values", int(df[DURATION].isna().sum())],
    ["Missing charger-type values", int(df[CHARGER].isna().sum())],
], columns=["Check", "Result"])
quality_summary.to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False)

print("\nMissing values by column:")
if missing_summary.empty:
    print("No missing values found.")
else:
    print(missing_summary.to_string(index=False))

print(f"\nExact duplicate rows: {duplicate_rows}")
print(f"Rows with at least one missing value: {rows_with_missing}")
print("Missing values are not imputed; pairwise complete rows are used when needed.")


# ============================================================
# 3. DATA PREPARATION FOR CHARGING-DURATION ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("3. PREPARE DATA FOR MAIN ANALYSIS")
print("=" * 65)

# Only rows missing the outcome or group label are excluded from the
# main charger-category comparison.
analysis_df = df.dropna(subset=[DURATION, CHARGER]).copy()

print(f"Rows in original dataset: {len(df)}")
print(f"Rows available for duration-by-charger analysis: {len(analysis_df)}")
print("\nNumber of sessions in each charger category:")
print(analysis_df[CHARGER].value_counts().to_string())


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS: DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 65)
print("4. EDA — DESCRIPTIVE STATISTICS BY CHARGER TYPE")
print("=" * 65)

duration_summary = (
    analysis_df.groupby(CHARGER)[DURATION]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reindex(CHARGER_ORDER)
    .rename(columns={
        "count": "n",
        "mean": "mean_hours",
        "median": "median_hours",
        "std": "sd_hours",
        "min": "min_hours",
        "max": "max_hours",
    })
)

duration_summary.round(4).to_csv(OUTPUT_DIR / "duration_summary_by_charger.csv")
print(duration_summary.round(3).to_string())


# ============================================================
# 5. EDA GRAPH 1: MISSING VALUES
# ============================================================

print("\n" + "=" * 65)
print("5. EDA GRAPH — MISSING VALUES")
print("=" * 65)

missing_to_plot = missing_counts[missing_counts > 0].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
if missing_to_plot.empty:
    ax.text(0.5, 0.5, "No missing values found", ha="center", va="center")
    ax.set_axis_off()
else:
    missing_to_plot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Missing records")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)

ax.set_title("Missing values by variable")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "missing_values.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {FIGURES_DIR / 'missing_values.png'}")


# ============================================================
# 6. EDA GRAPH 2: BOXPLOT OF DURATION BY CHARGER TYPE
# ============================================================

print("\n" + "=" * 65)
print("6. EDA GRAPH — CHARGING DURATION BOXPLOT")
print("=" * 65)

boxplot_data = [
    analysis_df.loc[analysis_df[CHARGER] == charger_name, DURATION]
    .dropna()
    .to_numpy()
    for charger_name in CHARGER_ORDER
]

fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot(boxplot_data, tick_labels=CHARGER_ORDER)
ax.set_title("Charging duration by charger category")
ax.set_ylabel("Charging duration (hours)")
ax.set_xlabel("Charger type")
ax.tick_params(axis="x", rotation=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "duration_boxplot.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {FIGURES_DIR / 'duration_boxplot.png'}")


# ============================================================
# 7. EDA GRAPH 3: ENERGY CONSUMED VS CHARGING DURATION
# ============================================================

print("\n" + "=" * 65)
print("7. EDA GRAPH — ENERGY VS DURATION SCATTERPLOT")
print("=" * 65)

scatter_df = df[[ENERGY, DURATION, CHARGER]].dropna()

fig, ax = plt.subplots(figsize=(8, 4))
for charger_name in CHARGER_ORDER:
    subset = scatter_df[scatter_df[CHARGER] == charger_name]
    ax.scatter(
        subset[ENERGY],
        subset[DURATION],
        label=charger_name,
        alpha=0.6,
        s=14,
    )

ax.set_title("Energy consumed vs. charging duration")
ax.set_xlabel("Energy consumed (kWh)")
ax.set_ylabel("Charging duration (hours)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "energy_duration_scatter.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Complete energy-duration pairs plotted: {len(scatter_df)}")
print(f"Saved: {FIGURES_DIR / 'energy_duration_scatter.png'}")


# ============================================================
# 8. DATA TRANSFORMATION: Z-SCORE NORMALISATION
# ============================================================

print("\n" + "=" * 65)
print("8. DATA TRANSFORMATION — Z-SCORE NORMALISATION")
print("=" * 65)

duration_values = analysis_df[DURATION]
duration_mean = duration_values.mean()
duration_sd = duration_values.std(ddof=1)  # sample standard deviation

# z = (x - mean) / sample standard deviation
analysis_df["Duration Z-Score"] = (
    (duration_values - duration_mean) / duration_sd
)

first_duration = float(duration_values.iloc[0])
first_z_score = float(analysis_df["Duration Z-Score"].iloc[0])

z_score_summary = pd.DataFrame([
    ["Duration mean (hours)", duration_mean],
    ["Duration sample standard deviation (hours)", duration_sd],
    ["First analysed duration (hours)", first_duration],
    ["First analysed duration z-score", first_z_score],
], columns=["Measure", "Value"])
z_score_summary.to_csv(OUTPUT_DIR / "z_score_summary.csv", index=False)

print("Formula: z = (x - mean) / sample standard deviation")
print(f"Mean duration: {duration_mean:.3f} hours")
print(f"Sample standard deviation: {duration_sd:.3f} hours")
print(f"First analysed duration: {first_duration:.3f} hours")
print(f"Example z-score: {first_z_score:.3f}")


# ============================================================
# 9. CDA: HYPOTHESIS TESTING USING ONE-WAY ANOVA
# ============================================================

print("\n" + "=" * 65)
print("9. CDA — ONE-WAY ANOVA")
print("=" * 65)

print("Research question: Does mean charging duration differ by charger type?")
print("H0: Mean duration is equal across Level 1, Level 2 and DC Fast Charger.")
print("H1: At least one charger category has a different mean duration.")
print("Significance level: alpha = 0.05")

anova_groups = [
    analysis_df.loc[analysis_df[CHARGER] == charger_name, DURATION]
    .dropna()
    .to_numpy()
    for charger_name in CHARGER_ORDER
]

if any(len(group) == 0 for group in anova_groups):
    raise ValueError("At least one charger group is empty. Check the category names.")

anova_result = stats.f_oneway(*anova_groups)
anova_decision = (
    "Reject H0" if anova_result.pvalue < 0.05 else "Fail to reject H0"
)

print(f"F-statistic: {anova_result.statistic:.6f}")
print(f"p-value: {anova_result.pvalue:.6f}")
print(f"Decision at alpha = 0.05: {anova_decision}")


# ============================================================
# 10. CDA: NON-PARAMETRIC CROSS-CHECK USING KRUSKAL-WALLIS
# ============================================================

print("\n" + "=" * 65)
print("10. CDA — KRUSKAL-WALLIS TEST")
print("=" * 65)

kruskal_result = stats.kruskal(*anova_groups)
kruskal_decision = (
    "Reject H0" if kruskal_result.pvalue < 0.05 else "Fail to reject H0"
)

print(f"H-statistic: {kruskal_result.statistic:.6f}")
print(f"p-value: {kruskal_result.pvalue:.6f}")
print(f"Decision at alpha = 0.05: {kruskal_decision}")

hypothesis_results = pd.DataFrame([
    ["One-way ANOVA", anova_result.statistic, anova_result.pvalue, anova_decision],
    ["Kruskal-Wallis", kruskal_result.statistic, kruskal_result.pvalue, kruskal_decision],
], columns=["Test", "Statistic", "p_value", "Decision_at_0.05"])
hypothesis_results.to_csv(OUTPUT_DIR / "hypothesis_test_results.csv", index=False)


# ============================================================
# 11. RELATIONSHIPS: PEARSON CORRELATION WITH DURATION
# ============================================================

print("\n" + "=" * 65)
print("11. RELATIONSHIPS — PEARSON CORRELATION")
print("=" * 65)

correlation_rows = []

for variable in CORRELATION_VARIABLES:
    if variable not in df.columns:
        print(f"Skipped missing column: {variable}")
        continue

    pair_df = df[[variable, DURATION]].dropna()
    if len(pair_df) >= 2:
        correlation_value = pair_df[variable].corr(pair_df[DURATION], method="pearson")
    else:
        correlation_value = np.nan

    correlation_rows.append([
        variable,
        correlation_value,
        len(pair_df),
    ])

correlation_summary = pd.DataFrame(
    correlation_rows,
    columns=["Variable", "Pearson_r_with_duration", "Complete_pairs"],
)
correlation_summary.to_csv(OUTPUT_DIR / "duration_correlations.csv", index=False)

print(correlation_summary.round(3).to_string(index=False))
print("\nCorrelation indicates linear association, not causation.")


# ============================================================
# 12. FINAL SUMMARY AND OUTPUT FILES
# ============================================================

print("\n" + "=" * 65)
print("12. FINAL SUMMARY")
print("=" * 65)

print(f"Total charging sessions: {len(df)}")
print(f"Sessions used in main comparison: {len(analysis_df)}")
print("\nMean duration by charger type (hours):")
print(duration_summary["mean_hours"].round(3).to_string())

print("\nHypothesis test summary:")
print(hypothesis_results.round(4).to_string(index=False))

print("\nConclusion:")
if anova_result.pvalue < 0.05:
    print("At alpha = 0.05, the ANOVA result is statistically significant.")
else:
    print(
        "At alpha = 0.05, we fail to reject H0. "
        "The dataset does not provide statistically significant evidence "
        "of a difference in mean charging duration across charger categories."
    )

print("\nGenerated files:")
print(f"- {OUTPUT_DIR / 'data_quality_summary.csv'}")
print(f"- {OUTPUT_DIR / 'missing_values_summary.csv'}")
print(f"- {OUTPUT_DIR / 'duration_summary_by_charger.csv'}")
print(f"- {OUTPUT_DIR / 'z_score_summary.csv'}")
print(f"- {OUTPUT_DIR / 'hypothesis_test_results.csv'}")
print(f"- {OUTPUT_DIR / 'duration_correlations.csv'}")
print(f"- {FIGURES_DIR / 'missing_values.png'}")
print(f"- {FIGURES_DIR / 'duration_boxplot.png'}")
print(f"- {FIGURES_DIR / 'energy_duration_scatter.png'}")
