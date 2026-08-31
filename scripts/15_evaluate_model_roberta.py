"""
==============================================================
MASTER THESIS

Experiment 4
RoBERTa Baseline Evaluation

Author: Aakash Vashist

==============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "roberta"
INPUT_FILE = OUTPUT_DIR / "roberta_test_predictions.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# LOAD DATA
# ============================================================

print("="*70)
print("Loading prediction file...")
print("="*70)

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded : {len(df)}")

# ============================================================
# REMOVE REFUSALS IF PRESENT
# ============================================================

if "REFUSAL" in df["predicted_label"].astype(str).unique():

    valid_df = df[
        df["predicted_label"] != "REFUSAL"
    ].copy()

else:

    valid_df = df.copy()

refusal_count = len(df) - len(valid_df)

print()

print(f"Total rows       : {len(df)}")
print(f"Valid rows       : {len(valid_df)}")
print(f"Refusals         : {refusal_count}")

print(
    f"Refusal Rate (%) : "
    f"{100*refusal_count/len(df):.2f}"
)

# ============================================================
# LABEL ENCODING
# ============================================================

label_map = {

    "not_hate":0,

    "hate":1

}

y_true = (
    valid_df["binary_hate_label_name"]
    .map(label_map)
)

y_pred = (
    valid_df["predicted_label"]
    .map(label_map)
)

# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

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

cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

print()

print("="*70)
print("OVERALL METRICS")
print("="*70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print()

print("Confusion Matrix")

print(cm)

print()

print(f"TP : {tp}")
print(f"TN : {tn}")
print(f"FP : {fp}")
print(f"FN : {fn}")

# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1"

    ],

    "Value":[

        accuracy,

        precision,

        recall,

        f1

    ]

})

metrics_df.to_csv(

    OUTPUT_DIR /
    "RoBERTa_metrics.csv",

    index=False

)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_true,

    y_pred,

    output_dict=True,

    zero_division=0

)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(

    OUTPUT_DIR /
    "RoBERTa_classification_report.csv"

)

# ============================================================
# CONFUSION MATRIX CSV
# ============================================================

cm_df = pd.DataFrame(

    cm,

    index=["Actual Not Hate","Actual Hate"],

    columns=["Predicted Not Hate","Predicted Hate"]

)

cm_df.to_csv(

    OUTPUT_DIR /
    "RoBERTa_confusion_matrix.csv"

)

# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(6,5))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    [0,1],
    ["Not Hate","Hate"]
)

plt.yticks(
    [0,1],
    ["Not Hate","Hate"]
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i,j],
            ha="center",
            va="center",
            fontsize=12
        )

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("RoBERTa Confusion Matrix")

plt.tight_layout()

plt.savefig(

    OUTPUT_DIR /
    "RoBERTa_confusion_matrix.png",

    dpi=300

)

plt.close()

# ============================================================
# FAIRNESS METRICS
# ============================================================

fairness_results = []

for dialect in sorted(valid_df["dialect_group"].dropna().unique()):

    subset = valid_df[
        valid_df["dialect_group"] == dialect
    ].copy()

    y_true_group = subset["binary_hate_label_name"].map(label_map)
    y_pred_group = subset["predicted_label"].map(label_map)

    acc = accuracy_score(
        y_true_group,
        y_pred_group
    )

    prec = precision_score(
        y_true_group,
        y_pred_group,
        zero_division=0
    )

    rec = recall_score(
        y_true_group,
        y_pred_group,
        zero_division=0
    )

    f1_group = f1_score(
        y_true_group,
        y_pred_group,
        zero_division=0
    )

    cm_group = confusion_matrix(
        y_true_group,
        y_pred_group,
        labels=[0,1]
    )

    tn_g, fp_g, fn_g, tp_g = cm_group.ravel()

    fpr = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0
    fnr = fn_g / (fn_g + tp_g) if (fn_g + tp_g) > 0 else 0
    tpr = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
    tnr = tn_g / (tn_g + fp_g) if (tn_g + fp_g) > 0 else 0

    fairness_results.append({

        "rows": len(subset),

        "accuracy": acc,

        "precision": prec,

        "recall": rec,

        "f1": f1_group,

        "FPR": fpr,

        "FNR": fnr,

        "TPR": tpr,

        "TNR": tnr,

        "Dialect": dialect

    })

fairness_df = pd.DataFrame(fairness_results)

print()

print("="*70)
print("FAIRNESS METRICS")
print("="*70)

print(fairness_df)

fairness_df.to_csv(

    OUTPUT_DIR /
    "RoBERTa_fairness_metrics.csv",

    index=False

)

# ============================================================
# FAIRNESS SUMMARY
# ============================================================

high = fairness_df.loc[
    fairness_df["Dialect"]=="high_aae",
    "FPR"
].values[0]

middle = fairness_df.loc[
    fairness_df["Dialect"]=="middle",
    "FPR"
].values[0]

low = fairness_df.loc[
    fairness_df["Dialect"]=="low_aae",
    "FPR"
].values[0]

gap = high - low

ratio = high / low if low > 0 else np.nan

print()

print("="*70)
print("FAIRNESS SUMMARY")
print("="*70)

print(f"High AAE FPR : {high:.4f}")
print(f"Middle FPR   : {middle:.4f}")
print(f"Low AAE FPR  : {low:.4f}")

print()

print(f"High - Low FPR Difference : {gap:.4f}")
print(f"High / Low FPR Ratio      : {ratio:.4f}")

# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

plt.figure(figsize=(6,4))

predictions = (
    valid_df["predicted_label"]
    .value_counts()
)

plt.bar(
    predictions.index,
    predictions.values
)

plt.title("RoBERTa Prediction Distribution")

plt.tight_layout()

plt.savefig(

    OUTPUT_DIR /
    "RoBERTa_prediction_distribution.png",

    dpi=300

)

plt.close()

# ============================================================
# ACCURACY BY DIALECT
# ============================================================

plt.figure(figsize=(6,4))

plt.bar(
    fairness_df["Dialect"],
    fairness_df["accuracy"]
)

plt.ylabel("Accuracy")

plt.title("RoBERTa Accuracy by Dialect")

plt.tight_layout()

plt.savefig(

    OUTPUT_DIR /
    "RoBERTa_accuracy_by_dialect.png",

    dpi=300

)

plt.close()

# ============================================================
# FPR
# ============================================================

plt.figure(figsize=(6,4))

plt.bar(
    fairness_df["Dialect"],
    fairness_df["FPR"]
)

plt.ylabel("False Positive Rate")

plt.title("RoBERTa FPR by Dialect")

plt.tight_layout()

plt.savefig(

    OUTPUT_DIR /
    "RoBERTa_FPR_by_dialect.png",

    dpi=300

)

plt.close()

# ============================================================
# FNR
# ============================================================

plt.figure(figsize=(6,4))

plt.bar(
    fairness_df["Dialect"],
    fairness_df["FNR"]
)

plt.ylabel("False Negative Rate")

plt.title("RoBERTa FNR by Dialect")

plt.tight_layout()

plt.savefig(

    OUTPUT_DIR /
    "RoBERTa_FNR_by_dialect.png",

    dpi=300

)

plt.close()

# ============================================================
# SUMMARY FILE
# ============================================================

with open(
    OUTPUT_DIR /
    "RoBERTa_summary.txt",
    "w"
) as f:

    f.write("="*70 + "\n")
    f.write("RoBERTa Baseline Evaluation\n")
    f.write("="*70 + "\n\n")

    f.write(f"Rows Evaluated : {len(valid_df)}\n")
    f.write(f"Refusals       : {refusal_count}\n\n")

    f.write(f"Accuracy  : {accuracy:.4f}\n")
    f.write(f"Precision : {precision:.4f}\n")
    f.write(f"Recall    : {recall:.4f}\n")
    f.write(f"F1 Score  : {f1:.4f}\n\n")

    f.write("Fairness Metrics\n")
    f.write("-----------------------------\n")

    f.write(fairness_df.to_string(index=False))

    f.write("\n\n")

    f.write(f"High-Low FPR Gap : {gap:.4f}\n")
    f.write(f"High/Low FPR Ratio : {ratio:.4f}\n")

# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("="*70)
print("FILES GENERATED")
print("="*70)

for file in sorted(OUTPUT_DIR.iterdir()):

    if file.is_file():

        print(file.name)

print()

print("="*70)
print("Evaluation Completed Successfully")
print("="*70)