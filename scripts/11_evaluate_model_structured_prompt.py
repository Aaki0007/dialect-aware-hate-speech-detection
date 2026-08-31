"""
==============================================================
MASTER THESIS

Universal Evaluation Script

Author: Aakash Vashist

This script evaluates any experiment.

Current experiment:
Structured Decision Prompting

Future experiments:
- RoBERTa

==============================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "Structured Decision Prompting"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR = RESULTS_DIR / "structured_prompt"
INPUT_FILE = OUTPUT_DIR / "structured_prompt_predictions.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("Loading prediction file...")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded : {len(df)}")

# ============================================================
# REMOVE REFUSALS
# ============================================================

total_rows = len(df)

refusals = df[df["predicted_label"] == "REFUSAL"]

valid_df = df[df["predicted_label"] != "REFUSAL"].copy()

print()

print(f"Total rows       : {total_rows}")
print(f"Valid rows       : {len(valid_df)}")
print(f"Refusals         : {len(refusals)}")
print(f"Refusal Rate (%) : {100*len(refusals)/total_rows:.2f}")

# ============================================================
# LABEL CONVERSION
# ============================================================

label_map = {
    "hate": 1,
    "not_hate": 0
}

valid_df["y_true"] = (
    valid_df["binary_hate_label_name"]
    .map(label_map)
)

valid_df["y_pred"] = (
    valid_df["predicted_label"]
    .map(label_map)
)

# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    valid_df["y_true"],
    valid_df["y_pred"]
)

precision = precision_score(
    valid_df["y_true"],
    valid_df["y_pred"],
    zero_division=0
)

recall = recall_score(
    valid_df["y_true"],
    valid_df["y_pred"],
    zero_division=0
)

f1 = f1_score(
    valid_df["y_true"],
    valid_df["y_pred"],
    zero_division=0
)

print()
print("=" * 70)
print("OVERALL METRICS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    valid_df["y_true"],
    valid_df["y_pred"]
)

TN, FP, FN, TP = cm.ravel()

print()

print("Confusion Matrix")

print(cm)

print()

print(f"TP : {TP}")
print(f"TN : {TN}")
print(f"FP : {FP}")
print(f"FN : {FN}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    valid_df["y_true"],
    valid_df["y_pred"],
    target_names=[
        "not_hate",
        "hate"
    ],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

# ============================================================
# SAVE OVERALL METRICS
# ============================================================

metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ],
        "Value": [
            accuracy,
            precision,
            recall,
            f1
        ]
    }
)

metrics_df.to_csv(
    OUTPUT_DIR / f"{EXPERIMENT_NAME}_metrics.csv",
    index=False
)

report_df.to_csv(
    OUTPUT_DIR / f"{EXPERIMENT_NAME}_classification_report.csv"
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_Not_Hate",
        "Actual_Hate"
    ],
    columns=[
        "Pred_Not_Hate",
        "Pred_Hate"
    ]
)

cm_df.to_csv(
    OUTPUT_DIR / f"{EXPERIMENT_NAME}_confusion_matrix.csv"
)

# ============================================================
# FAIRNESS METRICS
# ============================================================

def compute_group_metrics(group):

    y_true = group["y_true"]
    y_pred = group["y_pred"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0,1]
    )

    TN, FP, FN, TP = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    fpr = FP / (FP + TN) if (FP+TN)>0 else 0

    fnr = FN / (FN + TP) if (FN+TP)>0 else 0

    tpr = TP / (TP + FN) if (TP+FN)>0 else 0

    tnr = TN / (TN + FP) if (TN+FP)>0 else 0

    return {

        "rows": len(group),

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "FPR": fpr,

        "FNR": fnr,

        "TPR": tpr,

        "TNR": tnr

    }

# ============================================================
# DIALECT ANALYSIS
# ============================================================

fairness_results = []

for dialect in [
    "high_aae",
    "middle",
    "low_aae"
]:

    subset = valid_df[
        valid_df["dialect_group"] == dialect
    ]

    result = compute_group_metrics(subset)

    result["Dialect"] = dialect

    fairness_results.append(result)

fairness_df = pd.DataFrame(fairness_results)

fairness_df.to_csv(
    OUTPUT_DIR / f"{EXPERIMENT_NAME}_fairness_metrics.csv",
    index=False
)

print()

print("=" * 70)
print("FAIRNESS METRICS")
print("=" * 70)

print(fairness_df)

# ============================================================
# SUMMARY FILE
# ============================================================

with open(
    OUTPUT_DIR / f"{EXPERIMENT_NAME}_summary.txt",
    "w"
) as f:

    f.write(f"Experiment: {EXPERIMENT_NAME}\n\n")

    f.write(f"Total Rows: {total_rows}\n")
    f.write(f"Valid Rows: {len(valid_df)}\n")
    f.write(f"Refusals : {len(refusals)}\n\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n\n")

    f.write("Confusion Matrix\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write(fairness_df.to_string())


    # ============================================================
# VISUALISATIONS
# ============================================================

plt.rcParams["figure.figsize"] = (8, 6)

# ------------------------------------------------------------
# 1. Confusion Matrix
# ------------------------------------------------------------

fig, ax = plt.subplots()

im = ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Not Hate", "Hate"])
ax.set_yticklabels(["Not Hate", "Hate"])

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title(f"{EXPERIMENT_NAME} Confusion Matrix")

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold"
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / f"{EXPERIMENT_NAME}_confusion_matrix.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 2. Prediction Distribution
# ------------------------------------------------------------

prediction_counts = (
    valid_df["predicted_label"]
    .value_counts()
)

plt.figure()

prediction_counts.plot(kind="bar")

plt.title(f"{EXPERIMENT_NAME} Prediction Distribution")

plt.xlabel("Prediction")

plt.ylabel("Number of Tweets")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / f"{EXPERIMENT_NAME}_prediction_distribution.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 3. Accuracy by Dialect
# ------------------------------------------------------------

plt.figure()

plt.bar(
    fairness_df["Dialect"],
    fairness_df["accuracy"]
)

plt.ylim(0, 1)

plt.title("Accuracy by Dialect Group")

plt.ylabel("Accuracy")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / f"{EXPERIMENT_NAME}_accuracy_by_dialect.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 4. False Positive Rate
# ------------------------------------------------------------

plt.figure()

plt.bar(
    fairness_df["Dialect"],
    fairness_df["FPR"]
)

plt.ylim(0, 1)

plt.title("False Positive Rate by Dialect")

plt.ylabel("False Positive Rate")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / f"{EXPERIMENT_NAME}_FPR_by_dialect.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 5. False Negative Rate
# ------------------------------------------------------------

plt.figure()

plt.bar(
    fairness_df["Dialect"],
    fairness_df["FNR"]
)

plt.ylim(0, 1)

plt.title("False Negative Rate by Dialect")

plt.ylabel("False Negative Rate")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / f"{EXPERIMENT_NAME}_FNR_by_dialect.png",
    dpi=300
)

plt.close()


# ============================================================
# FAIRNESS SUMMARY
# ============================================================

print()

print("=" * 70)
print("FAIRNESS SUMMARY")
print("=" * 70)

high = fairness_df.loc[
    fairness_df["Dialect"] == "high_aae",
    "FPR"
].iloc[0]

middle = fairness_df.loc[
    fairness_df["Dialect"] == "middle",
    "FPR"
].iloc[0]

low = fairness_df.loc[
    fairness_df["Dialect"] == "low_aae",
    "FPR"
].iloc[0]

print(f"High AAE FPR : {high:.4f}")
print(f"Middle FPR   : {middle:.4f}")
print(f"Low AAE FPR  : {low:.4f}")

print()

print(f"High - Low FPR Difference : {high-low:.4f}")

if low > 0:
    print(f"High / Low FPR Ratio      : {high/low:.4f}")

print()

# ============================================================
# OUTPUT FILES
# ============================================================

print("=" * 70)
print("FILES GENERATED")
print("=" * 70)

for file in sorted(OUTPUT_DIR.iterdir()):
    print(file.name)

print()

print("=" * 70)
print("Evaluation Completed Successfully")
print("=" * 70)