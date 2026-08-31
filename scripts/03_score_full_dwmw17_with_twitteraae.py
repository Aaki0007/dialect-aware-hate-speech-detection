# ============================================================
# Full TwitterAAE Scoring for DWMW17
# Purpose:
#   - Load the full DWMW17 dataset
#   - Load TwitterAAE demographic/dialect model
#   - Add p_aa, p_hispanic, p_other, p_white scores
#   - Create dialect groups
#   - Save enriched dataset
# ============================================================

import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

DATA_PATH = DATA_DIR / "DWMV17_labeled_data.csv"
TWITTERAAE_DIR = DATA_DIR / "twitteraae"
TWITTERAAE_CODE_PATH = TWITTERAAE_DIR / "code"
TWITTERAAE_MODEL_PATH = TWITTERAAE_DIR / "model" / "model_count_table.txt"
TWITTERAAE_VOCAB_PATH = TWITTERAAE_DIR / "model" / "model_vocab.txt"

# Generated enriched data are kept under data/ and are not intended for Git.
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(TWITTERAAE_CODE_PATH))

import predict


# ------------------------------------------------------------
# 2. Check files
# ------------------------------------------------------------

print("Checking files:")
print("Dataset exists:", os.path.exists(DATA_PATH))
print("Model exists:", os.path.exists(TWITTERAAE_MODEL_PATH))
print("Vocab exists:", os.path.exists(TWITTERAAE_VOCAB_PATH))


# ------------------------------------------------------------
# 3. Load TwitterAAE model
# ------------------------------------------------------------

print("\nLoading TwitterAAE model...")

predict.vocabfile = str(TWITTERAAE_VOCAB_PATH)
predict.modelfile = str(TWITTERAAE_MODEL_PATH)
predict.load_model()

print("Model loaded successfully.")
print("Model vocabulary size:", len(predict.w2num))
print("Sample vocabulary tokens:", list(predict.w2num.keys())[:20])


# ------------------------------------------------------------
# 4. Load DWMW17 dataset
# ------------------------------------------------------------

print("\nLoading DWMW17 dataset...")

df = pd.read_csv(DATA_PATH)

if "Unnamed: 0" in df.columns:
    df = df.rename(columns={"Unnamed: 0": "original_index"})

class_map = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither"
}

df["class_name"] = df["class"].map(class_map)

print("DWMW17 loaded successfully.")
print("Shape:", df.shape)


# ------------------------------------------------------------
# 5. Simple tokenizer
# ------------------------------------------------------------

def simple_tweet_tokenizer(text):
    """
    Simple tokenizer for TwitterAAE scoring.

    It:
    - lowercases text
    - removes URLs
    - removes mentions
    - removes RT marker
    - keeps alphabetic words and apostrophes
    """

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\brt\b", " ", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)

    tokens = text.split()

    return tokens


# ------------------------------------------------------------
# 6. TwitterAAE scoring function
# ------------------------------------------------------------

def score_tweet(tweet):
    """
    Returns:
    p_aa, p_hispanic, p_other, p_white

    If the model cannot score a tweet, return NaN values.
    """

    try:
        tokens = simple_tweet_tokenizer(tweet)
        scores = predict.predict(tokens)

        if scores is None:
            return [np.nan, np.nan, np.nan, np.nan]

        return scores.tolist()

    except Exception:
        return [np.nan, np.nan, np.nan, np.nan]


# ------------------------------------------------------------
# 7. Score full dataset
# ------------------------------------------------------------

print("\nScoring full DWMW17 dataset...")
print("This may take a few minutes.")

score_columns = ["p_aa", "p_hispanic", "p_other", "p_white"]

scores = []

for i, tweet in enumerate(df["tweet"]):
    scores.append(score_tweet(tweet))

    if (i + 1) % 1000 == 0:
        print(f"Scored {i + 1} / {len(df)} tweets")

scores_df = pd.DataFrame(scores, columns=score_columns)

df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)


# ------------------------------------------------------------
# 8. Scoring summary
# ------------------------------------------------------------

num_total = len(df)
num_scored = df["p_aa"].notna().sum()
num_nan = df["p_aa"].isna().sum()
success_rate = round((num_scored / num_total) * 100, 2)

print("\nScoring summary:")
print("Total tweets:", num_total)
print("Successfully scored:", num_scored)
print("Returned NaN:", num_nan)
print("Success rate:", success_rate, "%")


# ------------------------------------------------------------
# 9. Create dialect groups
# ------------------------------------------------------------

df["dialect_group"] = "middle"

df.loc[df["p_aa"] >= 0.80, "dialect_group"] = "high_aae"
df.loc[df["p_aa"] <= 0.20, "dialect_group"] = "low_aae"

df.loc[df["p_aa"].isna(), "dialect_group"] = "unknown"


# Quartile groups for robustness checking
q25 = df["p_aa"].quantile(0.25)
q75 = df["p_aa"].quantile(0.75)

df["aae_quartile_group"] = "middle"
df.loc[df["p_aa"] <= q25, "aae_quartile_group"] = "bottom_25_aae"
df.loc[df["p_aa"] >= q75, "aae_quartile_group"] = "top_25_aae"
df.loc[df["p_aa"].isna(), "aae_quartile_group"] = "unknown"

print("\nDialect group distribution:")
print(df["dialect_group"].value_counts(dropna=False))

print("\nAAE quartile group distribution:")
print(df["aae_quartile_group"].value_counts(dropna=False))

print("\nAAE score summary:")
print(df["p_aa"].describe())


# ------------------------------------------------------------
# 10. Save outputs
# ------------------------------------------------------------

full_output_path = os.path.join(
    OUTPUT_DIR,
    "dwmw17_with_twitteraae_scores.csv"
)

summary_output_path = os.path.join(
    OUTPUT_DIR,
    "twitteraae_scoring_summary.csv"
)

df.to_csv(full_output_path, index=False)

summary_df = pd.DataFrame({
    "metric": [
        "total_tweets",
        "successfully_scored",
        "returned_nan",
        "success_rate_percent",
        "p_aa_q25",
        "p_aa_q75"
    ],
    "value": [
        num_total,
        num_scored,
        num_nan,
        success_rate,
        q25,
        q75
    ]
})

summary_df.to_csv(summary_output_path, index=False)

print("\nSaved full enriched dataset to:")
print(full_output_path)

print("\nSaved scoring summary to:")
print(summary_output_path)

print("\nFull TwitterAAE scoring completed.")
