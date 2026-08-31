"""
==============================================================
MASTER THESIS

Final Comparison
All Models

Author: Aakash Vashist

==============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

ZERO_DIR = RESULTS_DIR / "zero_shot"
SIMPLE_DIR = RESULTS_DIR / "simple_prompt"
STRUCTURED_DIR = RESULTS_DIR / "structured_prompt"
ROBERTA_DIR = RESULTS_DIR / "roberta"
OUTPUT_DIR = RESULTS_DIR / "final_comparison"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# LOAD FILES
# ============================================================

print("="*70)
print("Loading experiment results...")
print("="*70)

zero_metrics = pd.read_csv(
    ZERO_DIR/"Zero-Shot_metrics.csv"
)

simple_metrics = pd.read_csv(
    SIMPLE_DIR/"Simple Prompt_metrics.csv"
)

structured_metrics = pd.read_csv(
    STRUCTURED_DIR/"Structured Decision Prompting_metrics.csv"
)

roberta_metrics = pd.read_csv(
    ROBERTA_DIR/"RoBERTa_metrics.csv"
)

zero_fair = pd.read_csv(
    ZERO_DIR/"Zero-Shot_fairness_metrics.csv"
)

simple_fair = pd.read_csv(
    SIMPLE_DIR/"Simple Prompt_fairness_metrics.csv"
)

structured_fair = pd.read_csv(
    STRUCTURED_DIR/"Structured Decision Prompting_fairness_metrics.csv"
)

roberta_fair = pd.read_csv(
    ROBERTA_DIR/"RoBERTa_fairness_metrics.csv"
)

print("Loaded successfully.")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_metric(df, metric):

    return float(
        df.loc[
            df["Metric"] == metric,
            "Value"
        ].iloc[0]
    )

def get_fpr(df, dialect):

    return float(
        df.loc[
            df["Dialect"] == dialect,
            "FPR"
        ].iloc[0]
    )

# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Experiment":[

        "Zero-Shot",

        "Simple Prompt",

        "Structured Prompt",

        "RoBERTa"

    ],

    "Accuracy":[

        get_metric(zero_metrics,"Accuracy"),

        get_metric(simple_metrics,"Accuracy"),

        get_metric(structured_metrics,"Accuracy"),

        get_metric(roberta_metrics,"Accuracy")

    ],

    "Precision":[

        get_metric(zero_metrics,"Precision"),

        get_metric(simple_metrics,"Precision"),

        get_metric(structured_metrics,"Precision"),

        get_metric(roberta_metrics,"Precision")

    ],

    "Recall":[

        get_metric(zero_metrics,"Recall"),

        get_metric(simple_metrics,"Recall"),

        get_metric(structured_metrics,"Recall"),

        get_metric(roberta_metrics,"Recall")

    ],

    "F1":[

        get_metric(zero_metrics,"F1"),

        get_metric(simple_metrics,"F1"),

        get_metric(structured_metrics,"F1"),

        get_metric(roberta_metrics,"F1")

    ],

    "High AAE FPR":[

        get_fpr(zero_fair,"high_aae"),

        get_fpr(simple_fair,"high_aae"),

        get_fpr(structured_fair,"high_aae"),

        get_fpr(roberta_fair,"high_aae")

    ],

    "Middle FPR":[

        get_fpr(zero_fair,"middle"),

        get_fpr(simple_fair,"middle"),

        get_fpr(structured_fair,"middle"),

        get_fpr(roberta_fair,"middle")

    ],

    "Low AAE FPR":[

        get_fpr(zero_fair,"low_aae"),

        get_fpr(simple_fair,"low_aae"),

        get_fpr(structured_fair,"low_aae"),

        get_fpr(roberta_fair,"low_aae")

    ]

})

comparison["High-Low Gap"] = (
    comparison["High AAE FPR"]
    -
    comparison["Low AAE FPR"]
)

comparison.to_csv(

    OUTPUT_DIR /
    "all_models_comparison.csv",

    index=False

)

print()

print("="*70)
print("FINAL COMPARISON")
print("="*70)

print(
    comparison.round(4)
)

# ============================================================
# ACCURACY COMPARISON
# ============================================================

metrics = [
    ("Accuracy", "comparison_accuracy.png"),
    ("Precision", "comparison_precision.png"),
    ("Recall", "comparison_recall.png"),
    ("F1", "comparison_f1.png")
]

for metric, filename in metrics:

    plt.figure(figsize=(7,5))

    plt.bar(
        comparison["Experiment"],
        comparison[metric]
    )

    plt.ylabel(metric)

    plt.title(f"{metric} Comparison")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300
    )

    plt.close()

# ============================================================
# HIGH-AAE FALSE POSITIVE RATE
# ============================================================

plt.figure(figsize=(7,5))

plt.bar(
    comparison["Experiment"],
    comparison["High AAE FPR"]
)

plt.ylabel("False Positive Rate")

plt.title("High-AAE False Positive Rate")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "comparison_high_aae_fpr.png",
    dpi=300
)

plt.close()

# ============================================================
# HIGH-LOW GAP
# ============================================================

plt.figure(figsize=(7,5))

plt.bar(
    comparison["Experiment"],
    comparison["High-Low Gap"]
)

plt.ylabel("FPR Gap")

plt.title("Dialect Fairness Gap")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "comparison_high_low_gap.png",
    dpi=300
)

plt.close()

# ============================================================
# FAIRNESS TABLE
# ============================================================

fairness_summary = comparison[[
    "Experiment",
    "High AAE FPR",
    "Middle FPR",
    "Low AAE FPR",
    "High-Low Gap"
]]

fairness_summary.to_csv(
    OUTPUT_DIR /
    "fairness_comparison.csv",
    index=False
)

# ============================================================
# BEST MODELS
# ============================================================

best_accuracy = comparison.loc[
    comparison["Accuracy"].idxmax()
]

best_precision = comparison.loc[
    comparison["Precision"].idxmax()
]

best_recall = comparison.loc[
    comparison["Recall"].idxmax()
]

best_f1 = comparison.loc[
    comparison["F1"].idxmax()
]

lowest_fpr = comparison.loc[
    comparison["High AAE FPR"].idxmin()
]

smallest_gap = comparison.loc[
    comparison["High-Low Gap"].abs().idxmin()
]

# ============================================================
# FINAL REPORT
# ============================================================

REPORT_FILE = OUTPUT_DIR / "Final_Comparison_Report.txt"

with open(REPORT_FILE, "w") as f:

    f.write("="*70 + "\n")
    f.write("MASTER THESIS\n")
    f.write("FINAL MODEL COMPARISON\n")
    f.write("="*70 + "\n\n")

    f.write("Overall Results\n")
    f.write("-"*70 + "\n\n")

    f.write(
        comparison.round(4).to_string(index=False)
    )

    f.write("\n\n")

    f.write("Best Models\n")
    f.write("-"*70 + "\n\n")

    f.write(
        f"Highest Accuracy : {best_accuracy['Experiment']} "
        f"({best_accuracy['Accuracy']:.4f})\n"
    )

    f.write(
        f"Highest Precision : {best_precision['Experiment']} "
        f"({best_precision['Precision']:.4f})\n"
    )

    f.write(
        f"Highest Recall : {best_recall['Experiment']} "
        f"({best_recall['Recall']:.4f})\n"
    )

    f.write(
        f"Highest F1 : {best_f1['Experiment']} "
        f"({best_f1['F1']:.4f})\n"
    )

    f.write(
        f"Lowest High-AAE FPR : {lowest_fpr['Experiment']} "
        f"({lowest_fpr['High AAE FPR']:.4f})\n"
    )

    f.write(
        f"Smallest Fairness Gap : {smallest_gap['Experiment']} "
        f"({smallest_gap['High-Low Gap']:.4f})\n"
    )

    f.write("\n")

    f.write("="*70 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*70 + "\n")

# ============================================================
# TERMINAL SUMMARY
# ============================================================

print()

print("="*70)
print("FINAL COMPARISON SUMMARY")
print("="*70)

print()

print(comparison.round(4))

print()

print("="*70)
print("BEST PERFORMERS")
print("="*70)

print()

print(
    f"Highest Accuracy : "
    f"{best_accuracy['Experiment']} "
    f"({best_accuracy['Accuracy']:.4f})"
)

print(
    f"Highest Precision : "
    f"{best_precision['Experiment']} "
    f"({best_precision['Precision']:.4f})"
)

print(
    f"Highest Recall : "
    f"{best_recall['Experiment']} "
    f"({best_recall['Recall']:.4f})"
)

print(
    f"Highest F1 Score : "
    f"{best_f1['Experiment']} "
    f"({best_f1['F1']:.4f})"
)

print(
    f"Lowest High-AAE FPR : "
    f"{lowest_fpr['Experiment']} "
    f"({lowest_fpr['High AAE FPR']:.4f})"
)

print(
    f"Smallest Fairness Gap : "
    f"{smallest_gap['Experiment']} "
    f"({smallest_gap['High-Low Gap']:.4f})"
)

print()

print("="*70)
print("FILES GENERATED")
print("="*70)

for file in sorted(OUTPUT_DIR.iterdir()):

    if file.is_file():

        print(file.name)

print()

print("="*70)
print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
print("="*70)