import os
import re
import string
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_PATH = DATA_DIR / "DWMV17_labeled_data.csv"
OUTPUT_DIR = RESULTS_DIR / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())



print("\n================ BASIC DATASET INFO ================")
print(df.info())

print("\n================ MISSING VALUES ================")
missing_values = df.isnull().sum()
print(missing_values)

print("\n================ DUPLICATE ROWS ================")
print("Duplicate rows:", df.duplicated().sum())

print("\n================ DUPLICATE TWEETS ================")
print("Duplicate tweets:", df["tweet"].duplicated().sum())



if "Unnamed: 0" in df.columns:
    df = df.rename(columns={"Unnamed: 0": "original_index"})



class_map = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither"
}

df["class_name"] = df["class"].map(class_map)

print("\n================ CLASS DISTRIBUTION ================")
class_counts = df["class_name"].value_counts()
class_percentages = df["class_name"].value_counts(normalize=True) * 100

class_distribution = pd.DataFrame({
    "count": class_counts,
    "percentage": class_percentages.round(2)
})

print(class_distribution)

class_distribution.to_csv(
    os.path.join(OUTPUT_DIR, "class_distribution.csv")
)



plt.figure(figsize=(8, 5))
class_counts.plot(kind="bar")
plt.title("Class Distribution in DWMW17 Dataset")
plt.xlabel("Class")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=300)
plt.show()




def count_hashtags(text):
    return len(re.findall(r"#\w+", str(text)))

def count_mentions(text):
    return len(re.findall(r"@\w+", str(text)))

def count_urls(text):
    return len(re.findall(r"http\S+|www\S+", str(text)))

def count_exclamation_marks(text):
    return str(text).count("!")

def count_question_marks(text):
    return str(text).count("?")

def count_uppercase_words(text):
    words = str(text).split()
    return sum(1 for word in words if word.isupper() and len(word) > 1)

def count_punctuation(text):
    return sum(1 for char in str(text) if char in string.punctuation)

def count_digits(text):
    return sum(1 for char in str(text) if char.isdigit())


df["char_length"] = df["tweet"].astype(str).apply(len)
df["word_count"] = df["tweet"].astype(str).apply(lambda x: len(x.split()))
df["hashtag_count"] = df["tweet"].apply(count_hashtags)
df["mention_count"] = df["tweet"].apply(count_mentions)
df["url_count"] = df["tweet"].apply(count_urls)
df["exclamation_count"] = df["tweet"].apply(count_exclamation_marks)
df["question_count"] = df["tweet"].apply(count_question_marks)
df["uppercase_word_count"] = df["tweet"].apply(count_uppercase_words)
df["punctuation_count"] = df["tweet"].apply(count_punctuation)
df["digit_count"] = df["tweet"].apply(count_digits)


print("\n================ TEXT FEATURE SUMMARY ================")
text_features = [
    "char_length",
    "word_count",
    "hashtag_count",
    "mention_count",
    "url_count",
    "exclamation_count",
    "question_count",
    "uppercase_word_count",
    "punctuation_count",
    "digit_count"
]

print(df[text_features].describe())

df[text_features].describe().to_csv(
    os.path.join(OUTPUT_DIR, "text_feature_summary.csv")
)


print("\n================ TEXT FEATURES BY CLASS ================")

text_features_by_class = df.groupby("class_name")[text_features].mean().round(2)
print(text_features_by_class)

text_features_by_class.to_csv(
    os.path.join(OUTPUT_DIR, "text_features_by_class.csv")
)



plt.figure(figsize=(9, 5))

for class_name in df["class_name"].unique():
    subset = df[df["class_name"] == class_name]
    plt.hist(
        subset["word_count"],
        bins=40,
        alpha=0.5,
        label=class_name
    )

plt.title("Tweet Word Count Distribution by Class")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "word_count_distribution_by_class.png"), dpi=300)
plt.show()


plt.figure(figsize=(9, 5))

for class_name in df["class_name"].unique():
    subset = df[df["class_name"] == class_name]
    plt.hist(
        subset["char_length"],
        bins=40,
        alpha=0.5,
        label=class_name
    )

plt.title("Tweet Character Length Distribution by Class")
plt.xlabel("Character Length")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "char_length_distribution_by_class.png"), dpi=300)
plt.show()


vote_columns = ["hate_speech", "offensive_language", "neither"]

df["max_vote"] = df[vote_columns].max(axis=1)

df["second_max_vote"] = df[vote_columns].apply(
    lambda row: sorted(row)[-2],
    axis=1
)

df["vote_margin"] = df["max_vote"] - df["second_max_vote"]
df["agreement_ratio"] = df["max_vote"] / df["count"]

print("\n================ ANNOTATION AGREEMENT SUMMARY ================")
agreement_summary = df[["max_vote", "vote_margin", "agreement_ratio"]].describe()
print(agreement_summary)

agreement_summary.to_csv(
    os.path.join(OUTPUT_DIR, "annotation_agreement_summary.csv")
)

print("\n================ AGREEMENT BY CLASS ================")
agreement_by_class = df.groupby("class_name")[["max_vote", "vote_margin", "agreement_ratio"]].mean().round(3)
print(agreement_by_class)

agreement_by_class.to_csv(
    os.path.join(OUTPUT_DIR, "annotation_agreement_by_class.csv")
)


plt.figure(figsize=(8, 5))
df.boxplot(column="agreement_ratio", by="class_name")
plt.title("Annotation Agreement Ratio by Class")
plt.suptitle("")
plt.xlabel("Class")
plt.ylabel("Agreement Ratio")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "agreement_ratio_by_class.png"), dpi=300)
plt.show()


plt.figure(figsize=(8, 5))
df["agreement_ratio"].hist(bins=30)
plt.title("Distribution of Annotation Agreement Ratio")
plt.xlabel("Agreement Ratio")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "agreement_ratio_distribution.png"), dpi=300)
plt.show()


low_agreement_df = df.sort_values("agreement_ratio").head(50)

low_agreement_df[
    [
        "tweet",
        "class_name",
        "hate_speech",
        "offensive_language",
        "neither",
        "count",
        "agreement_ratio",
        "vote_margin"
    ]
].to_csv(
    os.path.join(OUTPUT_DIR, "low_agreement_examples.csv"),
    index=False
)

print("\nSaved 50 low-agreement examples.")



def clean_text_for_eda(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["tweet_clean_eda"] = df["tweet"].apply(clean_text_for_eda)



stopwords = set([
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "to", "of", "in", "on", "for", "with", "this", "that", "it", "be",
    "as", "at", "by", "from", "you", "i", "me", "my", "we", "our",
    "he", "she", "they", "them", "his", "her", "their", "your",
    "<url>", "<user>"
])

def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
    return tokens


top_token_results = {}

for class_name in df["class_name"].unique():
    subset = df[df["class_name"] == class_name]
    all_tokens = []

    for tweet in subset["tweet"]:
        all_tokens.extend(tokenize(tweet))

    token_counts = pd.Series(all_tokens).value_counts().head(30)
    top_token_results[class_name] = token_counts

    token_counts.to_csv(
        os.path.join(OUTPUT_DIR, f"top_tokens_{class_name}.csv")
    )

    plt.figure(figsize=(10, 6))
    token_counts.sort_values().plot(kind="barh")
    plt.title(f"Top 30 Tokens: {class_name}")
    plt.xlabel("Frequency")
    plt.ylabel("Token")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"top_tokens_{class_name}.png"), dpi=300)
    plt.show()



df["binary_flagged_label"] = df["class"].apply(
    lambda x: 1 if x in [0, 1] else 0
)


df["binary_hate_label"] = df["class"].apply(
    lambda x: 1 if x == 0 else 0
)

print("\n================ BINARY LABEL DISTRIBUTION ================")
print("Flagged label:")
print(df["binary_flagged_label"].value_counts())

print("\nHate-only label:")
print(df["binary_hate_label"].value_counts())



processed_path = os.path.join(OUTPUT_DIR, "dwml17_eda_processed.csv")
df.to_csv(processed_path, index=False)

print("\nProcessed EDA dataset saved to:")
print(processed_path)


summary_report = {
    "total_rows": len(df),
    "total_columns": len(df.columns),
    "missing_values_total": int(df.isnull().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "duplicate_tweets": int(df["tweet"].duplicated().sum()),
    "average_word_count": round(df["word_count"].mean(), 2),
    "average_char_length": round(df["char_length"].mean(), 2),
    "average_annotation_agreement": round(df["agreement_ratio"].mean(), 3)
}

summary_df = pd.DataFrame([summary_report])
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "eda_summary_report.csv"),
    index=False
)

print("\n================ FINAL EDA SUMMARY ================")
print(summary_df)