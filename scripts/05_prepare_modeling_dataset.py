# ============================================================
# Prepare Modeling Dataset for Thesis Experiments
# Purpose:
#   - Load final enriched DWMW17 + TwitterAAE dataset
#   - Create binary hate-speech labels
#   - Remove unknown dialect rows
#   - Create train / validation / test split
#   - Save modeling-ready CSV files
# ============================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

INPUT_PATH = DATA_DIR / "dwmw17_with_twitteraae_scores_and_agreement.csv"
OUTPUT_DIR = DATA_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

print("Loading final enriched dataset...")

df = pd.read_csv(INPUT_PATH)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# 3. Basic checks
# ------------------------------------------------------------

required_columns = [
    "tweet",
    "class",
    "class_name",
    "p_aa",
    "dialect_group",
    "aae_quartile_group",
    "agreement_ratio"
]

missing_required = [col for col in required_columns if col not in df.columns]

if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

print("\nRequired columns found.")

print("\nOriginal class distribution:")
print(df["class_name"].value_counts())

print("\nOriginal dialect group distribution:")
print(df["dialect_group"].value_counts(dropna=False))

# ------------------------------------------------------------
# 4. Remove unknown / unscored p_aa rows
# ------------------------------------------------------------

unknown_rows = df[df["p_aa"].isna()].copy()

if len(unknown_rows) > 0:
    unknown_output_path = f"{OUTPUT_DIR}/dwmw17_unknown_paa_rows.csv"
    unknown_rows.to_csv(unknown_output_path, index=False)
    print(f"\nSaved unknown p_aa rows to: {unknown_output_path}")

df_model = df[df["p_aa"].notna()].copy()

print("\nAfter removing unknown p_aa rows:")
print("Shape:", df_model.shape)

# ------------------------------------------------------------
# 5. Create modeling labels
# ------------------------------------------------------------

# Main thesis task:
# Hate speech detection
# 1 = hate_speech
# 0 = not hate speech

df_model["binary_hate_label"] = df_model["class_name"].apply(
    lambda x: 1 if x == "hate_speech" else 0
)

df_model["binary_hate_label_name"] = df_model["binary_hate_label"].map({
    1: "hate",
    0: "not_hate"
})

# Secondary optional task:
# Flagged content detection
# 1 = hate_speech or offensive_language
# 0 = neither

df_model["binary_flagged_label"] = df_model["class_name"].apply(
    lambda x: 0 if x == "neither" else 1
)

df_model["binary_flagged_label_name"] = df_model["binary_flagged_label"].map({
    1: "flagged",
    0: "not_flagged"
})

print("\nBinary hate label distribution:")
print(df_model["binary_hate_label_name"].value_counts())
print(df_model["binary_hate_label_name"].value_counts(normalize=True).round(4) * 100)

print("\nBinary flagged label distribution:")
print(df_model["binary_flagged_label_name"].value_counts())
print(df_model["binary_flagged_label_name"].value_counts(normalize=True).round(4) * 100)

# ------------------------------------------------------------
# 6. Create stratification column
# ------------------------------------------------------------

# We want the split to preserve both:
# - hate / not_hate label
# - dialect group
#
# Example combined groups:
# hate_high_aae
# not_hate_low_aae
# not_hate_middle

df_model["stratify_group"] = (
    df_model["binary_hate_label_name"].astype(str)
    + "_"
    + df_model["dialect_group"].astype(str)
)

print("\nStratification group distribution:")
print(df_model["stratify_group"].value_counts())

# If any stratification group is too small, assign it to rare.
# This prevents train_test_split from failing.

group_counts = df_model["stratify_group"].value_counts()
rare_groups = group_counts[group_counts < 5].index.tolist()

df_model["stratify_group_safe"] = df_model["stratify_group"].apply(
    lambda x: "rare" if x in rare_groups else x
)

print("\nSafe stratification group distribution:")
print(df_model["stratify_group_safe"].value_counts())

# ------------------------------------------------------------
# 7. Train / validation / test split
# ------------------------------------------------------------

# Final proportions:
# Train: 70%
# Validation: 15%
# Test: 15%

print("\nCreating train / validation / test split...")

train_df, temp_df = train_test_split(
    df_model,
    test_size=0.30,
    random_state=42,
    stratify=df_model["stratify_group_safe"]
)

# For validation/test split, create safe stratification again on temp set
temp_group_counts = temp_df["stratify_group_safe"].value_counts()
temp_rare_groups = temp_group_counts[temp_group_counts < 2].index.tolist()

temp_df["temp_stratify_group_safe"] = temp_df["stratify_group_safe"].apply(
    lambda x: "rare" if x in temp_rare_groups else x
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["temp_stratify_group_safe"]
)

# Add split labels
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

combined_split_df = pd.concat([train_df, val_df, test_df], axis=0)

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))
print("Total:", len(combined_split_df))

# ------------------------------------------------------------
# 8. Check distributions after split
# ------------------------------------------------------------

def print_split_distribution(split_name, split_df):
    print(f"\n================ {split_name.upper()} DISTRIBUTION ================")

    print("\nBinary hate label:")
    print(split_df["binary_hate_label_name"].value_counts())
    print(split_df["binary_hate_label_name"].value_counts(normalize=True).round(4) * 100)

    print("\nOriginal 3-class label:")
    print(split_df["class_name"].value_counts())
    print(split_df["class_name"].value_counts(normalize=True).round(4) * 100)

    print("\nDialect group:")
    print(split_df["dialect_group"].value_counts())
    print(split_df["dialect_group"].value_counts(normalize=True).round(4) * 100)

    print("\nAAE quartile group:")
    print(split_df["aae_quartile_group"].value_counts())
    print(split_df["aae_quartile_group"].value_counts(normalize=True).round(4) * 100)


print_split_distribution("train", train_df)
print_split_distribution("validation", val_df)
print_split_distribution("test", test_df)

# ------------------------------------------------------------
# 9. Select modeling columns
# ------------------------------------------------------------

modeling_columns = [
    "original_index",
    "tweet",
    "class",
    "class_name",
    "binary_hate_label",
    "binary_hate_label_name",
    "binary_flagged_label",
    "binary_flagged_label_name",
    "p_aa",
    "p_hispanic",
    "p_other",
    "p_white",
    "dialect_group",
    "aae_quartile_group",
    "agreement_ratio",
    "vote_margin",
    "max_vote",
    "split"
]

# Keep only columns that exist
modeling_columns = [col for col in modeling_columns if col in combined_split_df.columns]

train_out = train_df[modeling_columns].copy()
val_out = val_df[modeling_columns].copy()
test_out = test_df[modeling_columns].copy()
combined_out = combined_split_df[modeling_columns].copy()

# ------------------------------------------------------------
# 10. Save outputs
# ------------------------------------------------------------

train_path = f"{OUTPUT_DIR}/modeling_train.csv"
val_path = f"{OUTPUT_DIR}/modeling_validation.csv"
test_path = f"{OUTPUT_DIR}/modeling_test.csv"
combined_path = f"{OUTPUT_DIR}/modeling_dataset_with_splits.csv"

train_out.to_csv(train_path, index=False)
val_out.to_csv(val_path, index=False)
test_out.to_csv(test_path, index=False)
combined_out.to_csv(combined_path, index=False)

print("\nSaved modeling files:")
print(train_path)
print(val_path)
print(test_path)
print(combined_path)

# ------------------------------------------------------------
# 11. Save split summary
# ------------------------------------------------------------

summary = []

for split_name, split_df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df)
]:
    summary.append({
        "split": split_name,
        "rows": len(split_df),
        "hate_count": int((split_df["binary_hate_label"] == 1).sum()),
        "not_hate_count": int((split_df["binary_hate_label"] == 0).sum()),
        "high_aae_count": int((split_df["dialect_group"] == "high_aae").sum()),
        "low_aae_count": int((split_df["dialect_group"] == "low_aae").sum()),
        "middle_count": int((split_df["dialect_group"] == "middle").sum()),
        "mean_p_aa": round(split_df["p_aa"].mean(), 4),
        "mean_agreement_ratio": round(split_df["agreement_ratio"].mean(), 4)
    })

summary_df = pd.DataFrame(summary)

summary_path = f"{OUTPUT_DIR}/modeling_split_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\nSplit summary:")
print(summary_df)

print("\nSaved split summary to:")
print(summary_path)

print("\nModeling dataset preparation completed successfully.")