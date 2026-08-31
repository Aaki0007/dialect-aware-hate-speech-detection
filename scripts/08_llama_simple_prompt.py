"""
==============================================================
MASTER THESIS

Experiment 2
Simple Dialect-Aware Prompting using Llama 3

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = DATA_DIR / "modeling_test.csv"
OUTPUT_DIR = RESULTS_DIR / "simple_prompt"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_FILE = OUTPUT_DIR / "simple_prompt_predictions.csv"

REFUSAL_FILE = OUTPUT_DIR / "simple_prompt_refusals.csv"

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are assisting a university research project.

Task:
Classify a tweet into exactly one label.

Labels:
- hate
- not_hate

African-American English (AAE) is a legitimate dialect of English.
Dialectal vocabulary, slang, profanity, or reclaimed language alone is NOT evidence of hate speech.

Classify a tweet as hate ONLY if it attacks or expresses hatred toward a protected group such as race, ethnicity, religion, nationality, gender, sexual orientation, or disability.

Otherwise classify it as not_hate.

Return ONLY one line in this exact format:

LABEL: hate

or

LABEL: not_hate
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

    # Structured output
    if "label:" in response:

        label = response.split("label:")[-1].strip()

        if label.startswith("not_hate"):
            return "not_hate"

        if label.startswith("hate"):
            return "hate"

    # Exact matches
    if response == "not_hate":
        return "not_hate"

    if response == "hate":
        return "hate"

    # Common variants
    if "not_hate" in response:
        return "not_hate"

    if "not hate" in response:
        return "not_hate"

    if response.startswith("hate"):
        return "hate"

    return "REFUSAL"

# ============================================================
# OLLAMA CALL
# ============================================================

def run_llama(prompt):

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "num_predict": 5
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            return None

        return response.json()["response"]

    except Exception as e:

        print("ERROR:", e)

        return None

# ============================================================
# SECOND ATTEMPT IF REFUSED
# ============================================================

def retry_prompt(tweet):

    return f"""
You are a hate speech classifier.

Reply using ONLY ONE WORD.

hate

or

not_hate

Tweet:

{tweet}
"""

# ============================================================
# MAIN PROMPT
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
print("Running Simple Dialect-Aware Prompt Experiment")
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
    # Retry once if the model refuses or produces an
    # unexpected response
    # --------------------------------------------------------

    if prediction == "REFUSAL":

        raw_response = run_llama(
            retry_prompt(tweet)
        )

        prediction = parse_prediction(raw_response)

    # --------------------------------------------------------
    # Store prediction
    # --------------------------------------------------------

    result = row.to_dict()

    result["predicted_label"] = prediction

    result["raw_response"] = raw_response

    predictions.append(result)

    if prediction == "REFUSAL":

        refusals.append(result)

# ============================================================
# SAVE OUTPUTS
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
    f"{100*len(refusals)/len(predictions_df):.2f}"
)

print()

print("Predictions saved to:")

print(PREDICTION_FILE)

if len(refusals):

    print()

    print("Refusals saved to:")

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

available_cols = [
    c for c in cols
    if c in predictions_df.columns
]

print(
    predictions_df[available_cols]
    .head()
)

print()

print("=" * 70)
print("Simple Prompt Experiment Complete")
print("=" * 70)