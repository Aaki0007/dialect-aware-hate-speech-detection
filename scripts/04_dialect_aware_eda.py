# ============================================================
# Dialect-Aware EDA for DWMW17 + TwitterAAE Scores
# Purpose:
#   - Analyze label distribution by dialect group
#   - Analyze p_aa distribution
#   - Analyze annotation agreement by dialect group
#   - Save tables and plots for thesis/presentation
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_PATH = DATA_DIR / "dwmw17_with_twitteraae_scores.csv"
OUTPUT_DIR = RESULTS_DIR / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. Load enriched dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 3. Recalculate annotation agreement if needed
# ------------------------------------------------------------

vote_cols = ["hate_speech", "offensive_language", "neither"]

df["max_vote"] = df[vote_cols].max(axis=1)

df["second_max_vote"] = df[vote_cols].apply(
    lambda row: sorted(row)[-2],
    axis=1
)

df["vote_margin"] = df["max_vote"] - df["second_max_vote"]
df["agreement_ratio"] = df["max_vote"] / df["count"]

# ------------------------------------------------------------
# 4. Dialect group distribution
# ------------------------------------------------------------

print("\n================ DIALECT GROUP DISTRIBUTION ================")

dialect_distribution = df["dialect_group"].value_counts().reset_index()
dialect_distribution.columns = ["dialect_group", "count"]
dialect_distribution["percentage"] = (
    dialect_distribution["count"] / len(df) * 100
).round(2)

print(dialect_distribution)

dialect_distribution.to_csv(
    f"{OUTPUT_DIR}/dialect_group_distribution.csv",
    index=False
)

plt.figure(figsize=(8, 5))
plt.bar(dialect_distribution["dialect_group"], dialect_distribution["count"])
plt.title("Dialect Group Distribution")
plt.xlabel("Dialect Group")
plt.ylabel("Number of Tweets")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dialect_group_distribution.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 5. Class distribution by dialect group
# ------------------------------------------------------------

print("\n================ CLASS DISTRIBUTION BY DIALECT GROUP ================")

class_by_dialect_counts = pd.crosstab(
    df["dialect_group"],
    df["class_name"]
)

class_by_dialect_percent = pd.crosstab(
    df["dialect_group"],
    df["class_name"],
    normalize="index"
) * 100

class_by_dialect_percent = class_by_dialect_percent.round(2)

print("\nCounts:")
print(class_by_dialect_counts)

print("\nPercentages:")
print(class_by_dialect_percent)

class_by_dialect_counts.to_csv(
    f"{OUTPUT_DIR}/class_distribution_by_dialect_counts.csv"
)

class_by_dialect_percent.to_csv(
    f"{OUTPUT_DIR}/class_distribution_by_dialect_percent.csv"
)

class_by_dialect_percent.plot(kind="bar", figsize=(10, 6))
plt.title("Class Distribution by Dialect Group")
plt.xlabel("Dialect Group")
plt.ylabel("Percentage")
plt.xticks(rotation=30)
plt.legend(title="Class")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_distribution_by_dialect_group.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 6. AAE quartile group analysis
# ------------------------------------------------------------

print("\n================ CLASS DISTRIBUTION BY AAE QUARTILE GROUP ================")

class_by_quartile_counts = pd.crosstab(
    df["aae_quartile_group"],
    df["class_name"]
)

class_by_quartile_percent = pd.crosstab(
    df["aae_quartile_group"],
    df["class_name"],
    normalize="index"
) * 100

class_by_quartile_percent = class_by_quartile_percent.round(2)

print("\nCounts:")
print(class_by_quartile_counts)

print("\nPercentages:")
print(class_by_quartile_percent)

class_by_quartile_counts.to_csv(
    f"{OUTPUT_DIR}/class_distribution_by_aae_quartile_counts.csv"
)

class_by_quartile_percent.to_csv(
    f"{OUTPUT_DIR}/class_distribution_by_aae_quartile_percent.csv"
)

class_by_quartile_percent.plot(kind="bar", figsize=(10, 6))
plt.title("Class Distribution by AAE Quartile Group")
plt.xlabel("AAE Quartile Group")
plt.ylabel("Percentage")
plt.xticks(rotation=30)
plt.legend(title="Class")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_distribution_by_aae_quartile_group.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 7. Annotation agreement by dialect group
# ------------------------------------------------------------

print("\n================ ANNOTATION AGREEMENT BY DIALECT GROUP ================")

agreement_by_dialect = df.groupby("dialect_group")[
    ["agreement_ratio", "vote_margin", "max_vote"]
].mean().round(3)

print(agreement_by_dialect)

agreement_by_dialect.to_csv(
    f"{OUTPUT_DIR}/annotation_agreement_by_dialect_group.csv"
)

agreement_by_dialect["agreement_ratio"].plot(kind="bar", figsize=(8, 5))
plt.title("Average Annotation Agreement by Dialect Group")
plt.xlabel("Dialect Group")
plt.ylabel("Average Agreement Ratio")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/annotation_agreement_by_dialect_group.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 8. p_aa distribution
# ------------------------------------------------------------

print("\n================ P_AA SCORE SUMMARY ================")

paa_summary = df["p_aa"].describe().round(4)
print(paa_summary)

paa_summary.to_csv(
    f"{OUTPUT_DIR}/p_aa_score_summary.csv"
)

plt.figure(figsize=(9, 5))
df["p_aa"].dropna().hist(bins=50)
plt.title("Distribution of p_aa Scores")
plt.xlabel("p_aa Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/p_aa_score_distribution.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 9. Average p_aa by class
# ------------------------------------------------------------

print("\n================ P_AA BY CLASS ================")

paa_by_class = df.groupby("class_name")["p_aa"].agg(
    ["count", "mean", "median", "std", "min", "max"]
).round(3)

print(paa_by_class)

paa_by_class.to_csv(
    f"{OUTPUT_DIR}/p_aa_by_class.csv"
)

paa_by_class["mean"].plot(kind="bar", figsize=(8, 5))
plt.title("Average p_aa Score by Class")
plt.xlabel("Class")
plt.ylabel("Average p_aa")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/average_p_aa_by_class.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 10. High-AAE vs Low-AAE focused comparison
# ------------------------------------------------------------

focused_df = df[df["dialect_group"].isin(["high_aae", "low_aae"])].copy()

focused_comparison = pd.crosstab(
    focused_df["dialect_group"],
    focused_df["class_name"],
    normalize="index"
) * 100

focused_comparison = focused_comparison.round(2)

print("\n================ HIGH-AAE VS LOW-AAE COMPARISON ================")
print(focused_comparison)

focused_comparison.to_csv(
    f"{OUTPUT_DIR}/high_aae_vs_low_aae_class_distribution.csv"
)

focused_comparison.plot(kind="bar", figsize=(9, 5))
plt.title("High-AAE vs Low-AAE Label Distribution")
plt.xlabel("Dialect Group")
plt.ylabel("Percentage")
plt.xticks(rotation=30)
plt.legend(title="Class")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/high_aae_vs_low_aae_class_distribution.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 11. Save enriched file with agreement metrics
# ------------------------------------------------------------

final_output_path = DATA_DIR / "dwmw17_with_twitteraae_scores_and_agreement.csv"
df.to_csv(final_output_path, index=False)

print("\nSaved enriched dataset with agreement metrics to:")
print(final_output_path)

print("\nDialect-aware EDA completed successfully.")
