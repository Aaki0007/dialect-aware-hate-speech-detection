"""
==============================================================
MASTER THESIS

Experiment 3
Structured Decision Prompting using Llama 3

Author: Aakash Vashist

==============================================================
"""

import time
import requests
import pandas as pd

from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "llama3"

OLLAMA_URL = "http://localhost:11434/api/generate"

TEST_MODE = False

TEST_SIZE = 10

DEBUG = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = DATA_DIR / "modeling_test.csv"
OUTPUT_DIR = RESULTS_DIR / "structured_prompt"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_FILE = OUTPUT_DIR / "structured_prompt_predictions.csv"

REFUSAL_FILE = OUTPUT_DIR / "structured_prompt_refusals.csv"

# ============================================================
# STRUCTURED PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are assisting a university research project on hate speech detection.

Before making your decision, carefully consider the following steps.

Step 1
Determine whether the tweet contains African-American English (AAE),
slang, informal language, profanity, or conversational expressions.

Step 2
Remember:

AAE is a legitimate dialect of English.

Dialectal vocabulary, reclaimed language, profanity,
slang, or offensive wording alone is NOT hate speech.

Step 3

Determine whether the tweet attacks or expresses hatred toward a
protected group such as race, ethnicity, religion,
nationality, gender, sexual orientation,
or disability.

Decision Rule

If Step 3 is TRUE:

LABEL: hate

Otherwise:

LABEL: not_hate

Reply using ONLY the LABEL.

Do not explain your answer.
"""

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("Loading dataset...")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Dataset size : {len(df)}")

if TEST_MODE:

    df = df.head(TEST_SIZE)

    print(f"TEST MODE ENABLED ({TEST_SIZE} tweets)")

# ============================================================
# LABEL PARSER
# ============================================================

def parse_prediction(text):

    if text is None:
        return "REFUSAL"

    response = text.lower().strip()

    if "label:" in response:

        label = response.split("label:")[-1].strip()

        if label.startswith("not_hate"):
            return "not_hate"

        if label.startswith("hate"):
            return "hate"

    if response == "not_hate":
        return "not_hate"

    if response == "hate":
        return "hate"

    if "not_hate" in response:
        return "not_hate"

    if "not hate" in response:
        return "not_hate"

    if response.startswith("hate"):
        return "hate"

    return "REFUSAL"

# ============================================================
# OLLAMA
# ============================================================

def run_llama(prompt):

    payload = {

        "model": MODEL_NAME,

        "prompt": prompt,

        "stream": False,

        "options": {

            "temperature": 0,

            "top_p": 1,

            "num_predict": 8

        }

    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        if DEBUG:

            print("HTTP:", response.status_code)

            print(response.text[:400])

        if response.status_code != 200:

            return None

        return response.json()["response"]

    except Exception as e:

        if DEBUG:

            print(e)

        return None

# ============================================================
# RETRY PROMPT
# ============================================================

def retry_prompt(tweet):

    return f"""
Classify this tweet.

Remember:

AAE is NOT hate speech.

Only classify as hate if it attacks a protected group.

Return ONLY

LABEL: hate

or

LABEL: not_hate

Tweet:

{tweet}
"""

# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(tweet):

    return f"""
{SYSTEM_PROMPT}

Tweet:

{tweet}
"""

# ============================================================
# RUN INFERENCE
# ============================================================

print()
print("=" * 70)
print("Running Structured Prompt Experiment")
print("=" * 70)

predictions = []
refusals = []

start_time = time.time()

for idx, row in tqdm(df.iterrows(), total=len(df)):

    tweet = str(row["tweet"])

    prompt = build_prompt(tweet)

    raw_response = run_llama(prompt)

    prediction = parse_prediction(raw_response)

    # --------------------------------------------------------
    # Retry once if model refuses
    # --------------------------------------------------------

    if prediction == "REFUSAL":

        raw_response = run_llama(
            retry_prompt(tweet)
        )

        prediction = parse_prediction(raw_response)

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    result = row.to_dict()

    result["predicted_label"] = prediction

    result["raw_response"] = raw_response

    predictions.append(result)

    if prediction == "REFUSAL":

        refusals.append(result)

# ============================================================
# SAVE RESULTS
# ============================================================

predictions_df = pd.DataFrame(predictions)

predictions_df.to_csv(
    PREDICTION_FILE,
    index=False
)

if len(refusals) > 0:

    refusals_df = pd.DataFrame(refusals)

    refusals_df.to_csv(
        REFUSAL_FILE,
        index=False
    )

# ============================================================
# SUMMARY
# ============================================================

elapsed = time.time() - start_time

print()

print(f"Finished in {elapsed:.2f} seconds")

print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)

summary = (
    predictions_df["predicted_label"]
    .value_counts(dropna=False)
)

print(summary)

print()

print(f"Total predictions : {len(predictions_df)}")

print(f"Refusals          : {len(refusals)}")

print(
    f"Refusal Rate (%)  : "
    f"{100 * len(refusals) / len(predictions_df):.2f}"
)

print()

print("Predictions saved to:")

print(PREDICTION_FILE)

if len(refusals) > 0:

    print()

    print("Refusals saved to:")

    print(REFUSAL_FILE)

# ============================================================
# QUICK METRICS
# ============================================================

if "predicted_label" in predictions_df.columns:

    print()

    print("=" * 70)
    print("Prediction Distribution")
    print("=" * 70)

    distribution = (
        predictions_df["predicted_label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(distribution)

# ============================================================
# SAMPLE OUTPUT
# ============================================================

print()

print("=" * 70)
print("Example Predictions")
print("=" * 70)

columns = [

    "tweet",

    "binary_hate_label_name",

    "predicted_label",

    "dialect_group",

    "p_aa"

]

available_columns = [

    c for c in columns

    if c in predictions_df.columns

]

print(

    predictions_df[available_columns]

    .head()

)

# ============================================================
# SAVE SUMMARY FILE
# ============================================================

summary_file = OUTPUT_DIR / "structured_prompt_summary.txt"

with open(summary_file, "w") as f:

    f.write("=" * 60 + "\n")

    f.write("Structured Prompt Experiment Summary\n")

    f.write("=" * 60 + "\n\n")

    f.write(f"Model               : {MODEL_NAME}\n")

    f.write(f"Dataset Size        : {len(df)}\n")

    f.write(f"Total Predictions   : {len(predictions_df)}\n")

    f.write(f"Refusals            : {len(refusals)}\n")

    f.write(
        f"Refusal Rate (%)    : "
        f"{100 * len(refusals) / len(predictions_df):.2f}\n\n"
    )

    f.write("Prediction Distribution\n")

    f.write("-----------------------\n")

    f.write(summary.to_string())

    f.write("\n")

# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 70)
print("Structured Prompt Experiment Complete")
print("=" * 70)

print()

print("Files Generated:")

print()

print(PREDICTION_FILE)

if len(refusals) > 0:

    print(REFUSAL_FILE)

print(summary_file)