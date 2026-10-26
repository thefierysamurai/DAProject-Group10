# EV Charging Duration Analysis

A Data Analytics project investigating whether charging duration differs between **Level 1, Level 2, and DC Fast Charger** sessions.

## Dataset

- **Name:** Electric Vehicle Charging Patterns
- **Source:** [Kaggle dataset](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns)
- Download `ev_charging_patterns.csv` and place it in the same folder as `analysis.py`. The dataset is not included in the GitHub repository.

## What the project does

The Python script performs:
- Data quality checks for missing values and duplicate rows
- Descriptive statistics and exploratory data analysis
- A missing-values chart, charging-duration boxplot, and energy-versus-duration scatterplot
- Z-score normalisation and Pearson correlation
- One-way ANOVA and Kruskal–Wallis hypothesis tests

The heatmap is intentionally excluded to match the project report.

## Project files

```text
EV-Charging-Duration-Analysis/
├── analysis.py
├── README.md
├── requirements.txt
└── .gitignore
```

After running the script, results are saved in an `outputs/` folder as CSV summaries and PNG charts.

## Setup and run

1. Install Python 3.9 or newer.
2. Download the dataset CSV and place it beside `analysis.py`.
3. Open a terminal in the project folder and install the required libraries:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Run the analysis:

   ```bash
   python analysis.py
   ```

On some systems, use `python3` instead of `python`.

## Hypothesis tests

- **H₀:** Mean charging duration is equal across the three charger types.
- **H₁:** At least one charger type has a different mean duration.
- **Significance level:** 0.05.

ANOVA compares group means; Kruskal–Wallis provides a non-parametric comparison. A p-value below 0.05 is treated as statistically significant. Pearson correlation measures linear association, not causation.

## Notes

Missing values are reported rather than filled with invented values. Correlations use records with valid values for each variable pair. The results describe this dataset and should not automatically be generalised to all EV charging sessions.

## Project team

- August Kumar Sasmal — Roll No. Q051
- Brijesh Kumar Jha — Roll No. Q053
- Madhavan Balachandra Menon — Roll No. Q043
- Surya S Shenoy — Roll No. Q032


