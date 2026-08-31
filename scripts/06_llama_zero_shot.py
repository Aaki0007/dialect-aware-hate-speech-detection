"""
=====================================================================
MASTER THESIS

Experiment 1
Zero-Shot LLaMA 3 Hate Speech Classification

Author: Aakash Vashist

Description
-----------
This script evaluates LLaMA 3 using Zero-Shot prompting on the
DWMW17 test dataset.

The model receives only the tweet and predicts:

    hate
    not_hate

No dialect information is provided.

Outputs
-------
zero_shot_predictions.csv

zero_shot_refusals.csv
=====================================================================
"""

import os
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import ollama

# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_NAME = "llama3:latest"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = DATA_DIR / "modeling_test.csv"
OUTPUT_DIR = RESULTS_DIR / "zero_shot"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "zero_shot_predictions.csv"
)

REFUSAL_FILE = os.path.join(
    OUTPUT_DIR,
    "zero_shot_refusals.csv"
)

TEST_MODE = False
TEST_SIZE = 10

TEMPERATURE = 0

# ==========================================================
# CREATE OUTPUT FOLDER
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 70)
print("Loading dataset...")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Dataset size : {len(df)}")

if TEST_MODE:
    df = df.head(TEST_SIZE)
    print(f"TEST MODE ENABLED ({TEST_SIZE} tweets)")

print()

# ==========================================================
# SYSTEM PROMPT
# ==========================================================

SYSTEM_PROMPT = """
You are an annotation assistant working on an academic research project.

Your only task is to annotate tweets.

You are NOT generating hate speech.

You are NOT endorsing the content.

You are ONLY assigning one predefined label.

Never refuse.

Never explain.

Return only one label.
"""

# ==========================================================
# USER PROMPT
# ==========================================================

def build_prompt(tweet):

    return f"""
This is an academic annotation task.

Classify the following tweet.

Possible labels:

hate

not_hate

Tweet:

{tweet}

Return ONLY one line.

LABEL: hate

or

LABEL: not_hate
"""

# ==========================================================
# PARSE RESPONSE
# ==========================================================

def parse_prediction(response):

    text = response.lower().strip()

    if "label:" in text:

        label = text.split("label:")[-1].strip()

        if label.startswith("not_hate"):
            return "not_hate"

        if label.startswith("hate"):
            return "hate"

    if text == "not_hate":
        return "not_hate"

    if text == "hate":
        return "hate"

    if "not_hate" in text:
        return "not_hate"

    if "not hate" in text:
        return "not_hate"

    if "hate" in text:
        return "hate"

    refusal_words = [

        "cannot",

        "can't",

        "sorry",

        "unable",

        "refuse",

        "help you",

        "policy",

        "assist"

    ]

    for word in refusal_words:

        if word in text:
            return "REFUSAL"

    return "UNKNOWN"

# ==========================================================
# PREDICT SINGLE TWEET
# ==========================================================

def predict(tweet):

    prompt = build_prompt(tweet)

    try:

        response = ollama.chat(

            model=MODEL_NAME,

            messages=[

                {

                    "role": "system",

                    "content": SYSTEM_PROMPT

                },

                {

                    "role": "user",

                    "content": prompt

                }

            ],

            options={

                "temperature": TEMPERATURE

            }

        )

        raw_response = response["message"]["content"]

        prediction = parse_prediction(raw_response)

        return prediction, raw_response

    except Exception as e:

        return "ERROR", str(e)

# ==========================================================
# RUN EXPERIMENT
# ==========================================================

print("=" * 70)
print("Running Zero-Shot Experiment")
print("=" * 70)

predictions = []

raw_outputs = []

refusals = []

start = time.time()

for index, row in tqdm(df.iterrows(), total=len(df)):

    prediction, raw = predict(row["tweet"])

    predictions.append(prediction)

    raw_outputs.append(raw)

    if prediction in ["REFUSAL", "UNKNOWN", "ERROR"]:

        refusals.append({

            "index": index,

            "tweet": row["tweet"],

            "raw_response": raw,

            "prediction": prediction

        })

end = time.time()

print()

print(f"Finished in {(end-start):.2f} seconds")

# ==========================================================
# SAVE RESULTS
# ==========================================================

df["predicted_label"] = predictions

df["raw_model_response"] = raw_outputs

df.to_csv(

    OUTPUT_FILE,

    index=False

)

if len(refusals) > 0:

    pd.DataFrame(refusals).to_csv(

        REFUSAL_FILE,

        index=False

    )

# ==========================================================
# SUMMARY
# ==========================================================

print()

print("=" * 70)

print("SUMMARY")

print("=" * 70)

print()

print(df["predicted_label"].value_counts())

print()

print(f"Predictions saved to:")

print(OUTPUT_FILE)

if len(refusals):

    print()

    print(f"Refusals/Unknowns saved to:")

    print(REFUSAL_FILE)

print()

print("=" * 70)

print("Example Predictions")

print("=" * 70)

cols = [

    "tweet",

    "binary_hate_label_name",

    "predicted_label",

    "dialect_group",

    "p_aa"

]

print(df[cols].head())

print()

print("=" * 70)

print("Zero-Shot Experiment Complete")

print("=" * 70)