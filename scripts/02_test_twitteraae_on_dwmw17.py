# ============================================================
# Test TwitterAAE Scoring on DWMW17 Tweets
# Purpose:
#   - Load DWMW17 tweets
#   - Load TwitterAAE model
#   - Score a small sample of tweets
#   - Add p_aa, p_hispanic, p_other, p_white columns
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
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_PATH = DATA_DIR / "DWMV17_labeled_data.csv"
TWITTERAAE_DIR = DATA_DIR / "twitteraae"
TWITTERAAE_CODE_PATH = TWITTERAAE_DIR / "code"
TWITTERAAE_MODEL_PATH = TWITTERAAE_DIR / "model" / "model_count_table.txt"
TWITTERAAE_VOCAB_PATH = TWITTERAAE_DIR / "model" / "model_vocab.txt"

OUTPUT_DIR = RESULTS_DIR / "twitteraae_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add TwitterAAE code folder so Python can import predict.py
sys.path.append(str(TWITTERAAE_CODE_PATH))

import predict.py
sys.path.append(TWITTERAAE_CODE_PATH)

import predict


# ------------------------------------------------------------
# 2. Check files
# ------------------------------------------------------------

print("DATA_PATH:", DATA_PATH)
print("TWITTERAAE_CODE_PATH:", TWITTERAAE_CODE_PATH)
print("TWITTERAAE_MODEL_PATH:", TWITTERAAE_MODEL_PATH)
print("TWITTERAAE_VOCAB_PATH:", TWITTERAAE_VOCAB_PATH)

print("\nChecking files:")
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

# Check whether vocabulary now contains real words
sample_vocab = list(predict.w2num.keys())[:20]
print("Sample vocabulary tokens:", sample_vocab)


# ------------------------------------------------------------
# 4. Load DWMW17 dataset
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

if "Unnamed: 0" in df.columns:
    df = df.rename(columns={"Unnamed: 0": "original_index"})

class_map = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither"
}

df["class_name"] = df["class"].map(class_map)

print("\nDWMW17 loaded successfully.")
print("Shape:", df.shape)


# ------------------------------------------------------------
# 5. Simple tokenizer
# ------------------------------------------------------------

def simple_tweet_tokenizer(text):
    """
    Simple tokenizer for testing TwitterAAE scoring.

    It:
    - lowercases text
    - removes URLs
    - removes @mentions
    - keeps alphabetic words and apostrophes
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove RT marker
    text = re.sub(r"\brt\b", " ", text)

    # Keep only letters and apostrophes
    text = re.sub(r"[^a-zA-Z']", " ", text)

    # Split into tokens
    tokens = text.split()

    return tokens


# ------------------------------------------------------------
# 6. TwitterAAE scoring function
# ------------------------------------------------------------

def score_tweet(tweet):
    """
    Returns:
    p_aa, p_hispanic, p_other, p_white

    If too few words are in the TwitterAAE vocabulary,
    predict.predict() returns None, so we store NaN.
    """

    try:
        tokens = simple_tweet_tokenizer(tweet)

        scores = predict.predict(tokens)

        if scores is None:
            return [np.nan, np.nan, np.nan, np.nan]

        return scores.tolist()

    except Exception as e:
        print("Error scoring tweet:", e)
        return [np.nan, np.nan, np.nan, np.nan]


# ------------------------------------------------------------
# 7. Test on first 100 tweets
# ------------------------------------------------------------

sample = df.head(100).copy()

score_columns = ["p_aa", "p_hispanic", "p_other", "p_white"]

scores = sample["tweet"].apply(score_tweet)
scores_df = pd.DataFrame(scores.tolist(), columns=score_columns)

sample = pd.concat([sample.reset_index(drop=True), scores_df], axis=1)


# ------------------------------------------------------------
# 8. Print scoring summary
# ------------------------------------------------------------

num_scored = sample["p_aa"].notna().sum()
num_total = len(sample)

print("\nScoring summary:")
print("Tweets tested:", num_total)
print("Successfully scored:", num_scored)
print("Returned NaN:", num_total - num_scored)
print("Success rate:", round((num_scored / num_total) * 100, 2), "%")


print("\nFirst successfully scored tweets:")
print(
    sample[sample["p_aa"].notna()][
        [
            "tweet",
            "class",
            "class_name",
            "p_aa",
            "p_hispanic",
            "p_other",
            "p_white"
        ]
    ].head(10)
)


# ------------------------------------------------------------
# 9. Save test output
# ------------------------------------------------------------

output_path = os.path.join(
    OUTPUT_DIR,
    "twitteraae_test_100_dwmw17_tweets.csv"
)

sample.to_csv(output_path, index=False)

print("\nSaved test output to:")
print(output_path)

print("\nTest completed.")
