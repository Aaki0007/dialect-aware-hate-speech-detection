"""
==============================================================
MASTER THESIS

Compare all LLaMA Experiments

Author: Aakash Vashist

==============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ============================================================
# DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

ZERO_DIR = RESULTS_DIR / "zero_shot"
SIMPLE_DIR = RESULTS_DIR / "simple_prompt"
STRUCTURED_DIR = RESULTS_DIR / "structured_prompt"
OUTPUT_DIR = RESULTS_DIR / "llama_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

EXPERIMENTS = {
    "Zero-Shot": {
        "metrics": ZERO_DIR / "Zero-Shot_metrics.csv",
        "fairness": ZERO_DIR / "Zero-Shot_fairness_metrics.csv",
        "predictions": ZERO_DIR / "zero_shot_predictions.csv",
    },
    "Simple Prompt": {
        "metrics": SIMPLE_DIR / "Simple Prompt_metrics.csv",
        "fairness": SIMPLE_DIR / "Simple Prompt_fairness_metrics.csv",
        "predictions": SIMPLE_DIR / "simple_prompt_predictions.csv",
    },
    "Structured Decision Prompting": {
        "metrics": STRUCTURED_DIR / "Structured Decision Prompting_metrics.csv",
        "fairness": STRUCTURED_DIR / "Structured Decision Prompting_fairness_metrics.csv",
        "predictions": STRUCTURED_DIR / "structured_prompt_predictions.csv",
    },
}

print("="*70)
print("Loading Experiment Files...")
print("="*70)

# ============================================================
# METRIC READER
# ============================================================

def load_metrics(metric_file):

    df = pd.read_csv(metric_file)

    metrics = {}

    for _, row in df.iterrows():

        metrics[
            str(row["Metric"]).strip()
        ] = float(row["Value"])

    return metrics

# ============================================================
# FAIRNESS READER
# ============================================================

def load_fairness(fairness_file):

    df = pd.read_csv(fairness_file)

    fairness = {}

    for _, row in df.iterrows():

        dialect = row["Dialect"]

        fairness[dialect] = {

            "FPR":row["FPR"],

            "FNR":row["FNR"],

            "Accuracy":row["accuracy"],

            "Precision":row["precision"],

            "Recall":row["recall"],

            "F1":row["f1"]

        }

    return fairness

# ============================================================
# PREDICTION READER
# ============================================================

def load_prediction_stats(prediction_file):

    df = pd.read_csv(prediction_file)

    total = len(df)

    refusals = (
        df["predicted_label"]
        .eq("REFUSAL")
        .sum()
    )

    refusal_rate = 100*refusals/total

    prediction_counts = (
        df["predicted_label"]
        .value_counts()
        .to_dict()
    )

    return {

        "Total":total,

        "Refusals":refusals,

        "Refusal Rate":refusal_rate,

        "Prediction Counts":prediction_counts

    }

# ============================================================
# LOAD EVERYTHING
# ============================================================

all_metrics = {}

all_fairness = {}

all_prediction_stats = {}

for experiment, files in EXPERIMENTS.items():

    print(f"Loading {experiment}...")

    all_metrics[experiment] = load_metrics(
        files["metrics"]
    )

    all_fairness[experiment] = load_fairness(
        files["fairness"]
    )

    all_prediction_stats[experiment] = load_prediction_stats(
        files["predictions"]
    )

print()

print("All files loaded successfully.")

# ============================================================
# BUILD METRIC TABLE
# ============================================================

metric_rows = []

for experiment in EXPERIMENTS.keys():

    m = all_metrics[experiment]

    p = all_prediction_stats[experiment]

    metric_rows.append({

        "Experiment":experiment,

        "Accuracy":m["Accuracy"],

        "Precision":m["Precision"],

        "Recall":m["Recall"],

        "F1":m["F1"],

        "Total Predictions":p["Total"],

        "Refusals":p["Refusals"],

        "Refusal Rate":p["Refusal Rate"]

    })

comparison = pd.DataFrame(metric_rows)

print()

print("="*70)

print("OVERALL METRICS")

print("="*70)

print(comparison)

comparison.to_csv(

    OUTPUT_DIR/

    "llama_metrics_comparison.csv",

    index=False

)

# ============================================================
# BUILD FAIRNESS TABLE
# ============================================================

fairness_rows=[]

for experiment in EXPERIMENTS.keys():

    f=all_fairness[experiment]

    fairness_rows.append({

        "Experiment":experiment,

        "High AAE FPR":f["high_aae"]["FPR"],

        "Middle FPR":f["middle"]["FPR"],

        "Low AAE FPR":f["low_aae"]["FPR"],

        "High AAE FNR":f["high_aae"]["FNR"],

        "Middle FNR":f["middle"]["FNR"],

        "Low AAE FNR":f["low_aae"]["FNR"],

        "High-Low FPR Gap":

        f["high_aae"]["FPR"]-

        f["low_aae"]["FPR"]

    })

fairness=pd.DataFrame(fairness_rows)

print()

print("="*70)

print("FAIRNESS METRICS")

print("="*70)

print(fairness)

fairness.to_csv(

    OUTPUT_DIR/

    "llama_fairness_comparison.csv",

    index=False

)

print()

print("Part 1 Completed Successfully.")

# ============================================================
# IMPROVEMENT ANALYSIS
# ============================================================

print()
print("=" * 70)
print("CALCULATING IMPROVEMENTS")
print("=" * 70)

baseline = comparison.loc[
    comparison["Experiment"] == "Zero-Shot"
].iloc[0]

improvement_rows = []

for _, row in comparison.iterrows():

    improvement_rows.append({

        "Experiment": row["Experiment"],

        "Accuracy Improvement":
            row["Accuracy"] - baseline["Accuracy"],

        "Precision Improvement":
            row["Precision"] - baseline["Precision"],

        "Recall Change":
            row["Recall"] - baseline["Recall"],

        "F1 Improvement":
            row["F1"] - baseline["F1"],

        "Refusal Rate Change":
            baseline["Refusal Rate"] - row["Refusal Rate"]

    })

improvements = pd.DataFrame(improvement_rows)

print(improvements)

improvements.to_csv(
    OUTPUT_DIR / "llama_improvements.csv",
    index=False
)

# ============================================================
# CHART HELPER
# ============================================================

def bar_chart(df,
              column,
              title,
              ylabel,
              filename):

    plt.figure(figsize=(8,5))

    plt.bar(
        df["Experiment"],
        df[column]
    )

    plt.title(title)

    plt.ylabel(ylabel)

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300
    )

    plt.close()

# ============================================================
# OVERALL METRIC CHARTS
# ============================================================

print()
print("=" * 70)
print("Generating Overall Metric Charts...")
print("=" * 70)

bar_chart(
    comparison,
    "Accuracy",
    "Accuracy Comparison",
    "Accuracy",
    "Accuracy_Comparison.png"
)

bar_chart(
    comparison,
    "Precision",
    "Precision Comparison",
    "Precision",
    "Precision_Comparison.png"
)

bar_chart(
    comparison,
    "Recall",
    "Recall Comparison",
    "Recall",
    "Recall_Comparison.png"
)

bar_chart(
    comparison,
    "F1",
    "F1 Score Comparison",
    "F1 Score",
    "F1_Comparison.png"
)

bar_chart(
    comparison,
    "Refusal Rate",
    "Refusal Rate Comparison",
    "Refusal Rate (%)",
    "Refusal_Rate_Comparison.png"
)

# ============================================================
# FAIRNESS CHARTS
# ============================================================

print()
print("=" * 70)
print("Generating Fairness Charts...")
print("=" * 70)

bar_chart(
    fairness,
    "High AAE FPR",
    "High AAE False Positive Rate",
    "False Positive Rate",
    "High_AAE_FPR.png"
)

bar_chart(
    fairness,
    "Middle FPR",
    "Middle Dialect False Positive Rate",
    "False Positive Rate",
    "Middle_FPR.png"
)

bar_chart(
    fairness,
    "Low AAE FPR",
    "Low AAE False Positive Rate",
    "False Positive Rate",
    "Low_AAE_FPR.png"
)

bar_chart(
    fairness,
    "High-Low FPR Gap",
    "High vs Low AAE FPR Gap",
    "Difference",
    "FPR_Gap.png"
)

bar_chart(
    fairness,
    "High AAE FNR",
    "High AAE False Negative Rate",
    "False Negative Rate",
    "High_AAE_FNR.png"
)

bar_chart(
    fairness,
    "Middle FNR",
    "Middle Dialect False Negative Rate",
    "False Negative Rate",
    "Middle_FNR.png"
)

bar_chart(
    fairness,
    "Low AAE FNR",
    "Low AAE False Negative Rate",
    "False Negative Rate",
    "Low_AAE_FNR.png"
)

# ============================================================
# IMPROVEMENT CHARTS
# ============================================================

print()
print("=" * 70)
print("Generating Improvement Charts...")
print("=" * 70)

bar_chart(
    improvements,
    "Accuracy Improvement",
    "Accuracy Improvement vs Zero-Shot",
    "Improvement",
    "Accuracy_Improvement.png"
)

bar_chart(
    improvements,
    "Precision Improvement",
    "Precision Improvement vs Zero-Shot",
    "Improvement",
    "Precision_Improvement.png"
)

bar_chart(
    improvements,
    "Recall Change",
    "Recall Difference vs Zero-Shot",
    "Difference",
    "Recall_Change.png"
)

bar_chart(
    improvements,
    "F1 Improvement",
    "F1 Improvement vs Zero-Shot",
    "Improvement",
    "F1_Improvement.png"
)

bar_chart(
    improvements,
    "Refusal Rate Change",
    "Reduction in Refusal Rate",
    "Percentage Points",
    "Refusal_Reduction.png"
)

print()

print("=" * 70)
print("PART 2 COMPLETED")
print("=" * 70)

# ============================================================
# BEST EXPERIMENTS
# ============================================================

print()
print("=" * 70)
print("FINDING BEST EXPERIMENTS")
print("=" * 70)

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

lowest_refusal = comparison.loc[
    comparison["Refusal Rate"].idxmin()
]

lowest_high_fpr = fairness.loc[
    fairness["High AAE FPR"].idxmin()
]

lowest_gap = fairness.loc[
    fairness["High-Low FPR Gap"].idxmin()
]

# ============================================================
# THESIS SUMMARY REPORT
# ============================================================

summary_file = OUTPUT_DIR / "comparison_summary.txt"

with open(summary_file, "w") as f:

    f.write("=" * 70 + "\n")
    f.write("MASTER THESIS\n")
    f.write("LLaMA EXPERIMENT COMPARISON REPORT\n")
    f.write("=" * 70 + "\n\n")

    f.write("Experiments Compared\n")
    f.write("------------------------------\n")
    for exp in comparison["Experiment"]:
        f.write(f"- {exp}\n")

    f.write("\n")

    f.write("OVERALL PERFORMANCE\n")
    f.write("------------------------------\n")
    f.write(comparison.to_string(index=False))
    f.write("\n\n")

    f.write("FAIRNESS METRICS\n")
    f.write("------------------------------\n")
    f.write(fairness.to_string(index=False))
    f.write("\n\n")

    f.write("IMPROVEMENTS VS ZERO-SHOT\n")
    f.write("------------------------------\n")
    f.write(improvements.to_string(index=False))
    f.write("\n\n")

    f.write("BEST RESULTS\n")
    f.write("------------------------------\n")

    f.write(
        f"Highest Accuracy : "
        f"{best_accuracy['Experiment']} "
        f"({best_accuracy['Accuracy']:.4f})\n"
    )

    f.write(
        f"Highest Precision : "
        f"{best_precision['Experiment']} "
        f"({best_precision['Precision']:.4f})\n"
    )

    f.write(
        f"Highest Recall : "
        f"{best_recall['Experiment']} "
        f"({best_recall['Recall']:.4f})\n"
    )

    f.write(
        f"Highest F1 Score : "
        f"{best_f1['Experiment']} "
        f"({best_f1['F1']:.4f})\n"
    )

    f.write(
        f"Lowest Refusal Rate : "
        f"{lowest_refusal['Experiment']} "
        f"({lowest_refusal['Refusal Rate']:.2f}%)\n"
    )

    f.write(
        f"Lowest High-AAE FPR : "
        f"{lowest_high_fpr['Experiment']} "
        f"({lowest_high_fpr['High AAE FPR']:.4f})\n"
    )

    f.write(
        f"Smallest High-Low FPR Gap : "
        f"{lowest_gap['Experiment']} "
        f"({lowest_gap['High-Low FPR Gap']:.4f})\n"
    )

    f.write("\n")

    f.write("=" * 70 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 70 + "\n")

# ============================================================
# TERMINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print()

print(
    f"Best Accuracy        : "
    f"{best_accuracy['Experiment']} "
    f"({best_accuracy['Accuracy']:.4f})"
)

print(
    f"Best Precision       : "
    f"{best_precision['Experiment']} "
    f"({best_precision['Precision']:.4f})"
)

print(
    f"Best Recall          : "
    f"{best_recall['Experiment']} "
    f"({best_recall['Recall']:.4f})"
)

print(
    f"Best F1              : "
    f"{best_f1['Experiment']} "
    f"({best_f1['F1']:.4f})"
)

print(
    f"Lowest Refusal Rate  : "
    f"{lowest_refusal['Experiment']} "
    f"({lowest_refusal['Refusal Rate']:.2f}%)"
)

print(
    f"Lowest High-AAE FPR  : "
    f"{lowest_high_fpr['Experiment']} "
    f"({lowest_high_fpr['High AAE FPR']:.4f})"
)

print(
    f"Smallest FPR Gap     : "
    f"{lowest_gap['Experiment']} "
    f"({lowest_gap['High-Low FPR Gap']:.4f})"
)

# ============================================================
# GENERATED FILES
# ============================================================

print()
print("=" * 70)
print("FILES GENERATED")
print("=" * 70)

for file in sorted(OUTPUT_DIR.iterdir()):

    print(file.name)

print()

print("=" * 70)
print("LLAMA COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 70)

print()

print("Output Folder:")
print(OUTPUT_DIR)