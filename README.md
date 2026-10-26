# EV Charging Duration Analysis

This repository contains the Python analysis code for the project report **EV Charging Duration Analysis**.

## Research question
Does charging duration differ across Level 1, Level 2 and DC Fast Charger sessions?

## Dataset
Download `ev_charging_patterns.csv` from:
https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns

Place the CSV in this folder, or pass its location to the script. The dataset is not included in this repository; follow Kaggle's terms and the dataset licence when sharing data.

## Install requirements
Python 3.9 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python analysis.py --data ev_charging_patterns.csv
```

To choose another output folder:
```bash
python analysis.py --data "path/to/ev_charging_patterns.csv" --output outputs
```

## What the script does
1. Loads the CSV and inspects its dimensions.
2. Checks missing values and exact duplicate rows.
3. Uses records with non-missing charging duration and charger type for the main comparison.
4. Calculates count, mean, median, standard deviation, minimum and maximum duration by charger type.
5. Demonstrates z-score normalisation using the sample standard deviation.
6. Tests the hypotheses with one-way ANOVA and Kruskal–Wallis at alpha = 0.05.
7. Calculates Pearson correlations between charging duration and selected numeric variables using complete pairs.
8. Saves three report figures: missing-values chart, charger-duration boxplot, and energy-duration scatterplot.

The heatmap is intentionally omitted from the outputs because it was excluded from the submitted report.

## Hypotheses
- **H0:** Mean charging duration is equal across the three charger categories.
- **H1:** At least one category has a different mean charging duration.
- **Significance level:** alpha = 0.05.

A p-value below 0.05 leads to rejection of H0; otherwise, the result is reported as “fail to reject H0.” A non-significant result is not proof that the groups are identical.

## Data handling notes
- Missing values are not imputed.
- Correlations use complete records for each variable pair.
- Exact duplicate rows are counted and reported, not silently removed.
- State-of-charge values outside the usual 0–100% range are not automatically changed.
- No heatmap is generated.

## Outputs
The script creates an `outputs/` folder with CSV summaries:
- `data_quality_summary.csv`
- `missing_values_summary.csv`
- `duration_summary_by_charger.csv`
- `z_score_summary.csv`
- `hypothesis_test_results.csv`
- `duration_correlations.csv`

It also creates `outputs/figures/` with the three PNG charts.
