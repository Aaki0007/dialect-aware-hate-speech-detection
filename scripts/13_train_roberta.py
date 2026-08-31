
# ============================================================
# INSTALL DEPENDENCIES
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pathlib import Path

from datasets import Dataset

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

print("Torch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ROBERTA_DIR = PROJECT_ROOT / "results" / "roberta"
CHECKPOINT_DIR = ROBERTA_DIR / "checkpoints"
MODEL_OUTPUT = ROBERTA_DIR / "best_roberta"

ROBERTA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "modeling_train.csv"
VALIDATION_FILE = DATA_DIR / "modeling_validation.csv"

# ============================================================
# LOAD DATASETS
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALIDATION_FILE)

print("Training:", train_df.shape)
print("Validation:", valid_df.shape)

print()
print(train_df["binary_hate_label_name"].value_counts())

print()
print(valid_df["binary_hate_label_name"].value_counts())

# ============================================================
# LABEL ENCODING
# ============================================================

label_map = {
    "not_hate": 0,
    "hate": 1
}

train_df["label"] = train_df["binary_hate_label_name"].map(label_map)
valid_df["label"] = valid_df["binary_hate_label_name"].map(label_map)

train_dataset = Dataset.from_pandas(
    train_df[["tweet", "label"]]
)

validation_dataset = Dataset.from_pandas(
    valid_df[["tweet", "label"]]
)

print(train_dataset)
print(validation_dataset)

# ============================================================
# TOKENIZER
# ============================================================

MODEL_NAME = "roberta-base"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

MAX_LENGTH = 128

def tokenize(batch):

    return tokenizer(
        batch["tweet"],
        truncation=True,
        max_length=MAX_LENGTH
    )

train_dataset = train_dataset.map(tokenize, batched=True)
validation_dataset = validation_dataset.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

print("Tokenization complete.")

# ============================================================
# LOAD MODEL
# ============================================================

model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

print(model.config)

# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=str(CHECKPOINT_DIR),

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=16,

    per_device_eval_batch_size=16,

    num_train_epochs=3,

    weight_decay=0.01,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    save_total_limit=1,

    report_to="none",

    seed=42
)

print("TrainingArguments created successfully.")

# ============================================================
# CREATE TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("="*70)
print("Trainer created successfully.")
print("="*70)

# ============================================================
# TRAIN MODEL
# ============================================================

print("=" * 70)
print("Training RoBERTa...")
print("=" * 70)

train_result = trainer.train()

print()

print("=" * 70)
print("Training Completed")
print("=" * 70)

# ============================================================
# FINAL VALIDATION
# ============================================================

print("=" * 70)
print("Evaluating Best Model...")
print("=" * 70)

validation_metrics = trainer.evaluate()

print()

for key, value in validation_metrics.items():
    print(f"{key:25s}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

# ============================================================
# SAVE MODEL
# ============================================================

trainer.save_model(str(MODEL_OUTPUT))
tokenizer.save_pretrained(str(MODEL_OUTPUT))

print("="*70)
print("Best model saved successfully.")
print("="*70)

print("Saved to:")
print(MODEL_OUTPUT)

# ============================================================
# SAVE VALIDATION METRICS
# ============================================================

validation_df = pd.DataFrame([validation_metrics])

validation_df.to_csv(
    ROBERTA_DIR / "roberta_validation_metrics.csv",
    index=False
)

validation_df

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history = pd.DataFrame(trainer.state.log_history)

history.to_csv(
    ROBERTA_DIR / "roberta_training_history.csv",
    index=False
)

history.head()

# ============================================================
# TRAINING LOSS
# ============================================================

loss_df = history.dropna(subset=["loss"])

plt.figure(figsize=(8,5))

plt.plot(
    loss_df["epoch"],
    loss_df["loss"],
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("RoBERTa Training Loss")
plt.grid(True)

plt.show()

# ============================================================
# SAVE LOSS CURVE
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(
    loss_df["epoch"],
    loss_df["loss"],
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("RoBERTa Training Loss")
plt.grid(True)

plt.savefig(
    ROBERTA_DIR / "roberta_loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Loss curve saved.")

