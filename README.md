# Mitigating Racial Bias in Hate Speech Detection through Dialect-Aware Prompting

**Reproducibility repository for the MSc thesis _“Mitigating Racial Bias in Hate Speech Detection through Dialect-Aware Prompting”_**

This repository contains the code, exact prompt templates, environment information, aggregate evaluation outputs, and selected exploratory-analysis artifacts used in an MSc research project on dialect-associated disparities in hate-speech classification.

The study evaluates whether prompting a fixed local **LLaMA 3 8B** model with dialect-aware instructions changes classification behaviour across groups with different levels of African-American English (AAE) association, and compares those prompting conditions with a fine-tuned **RoBERTa** baseline.

> **Important terminology note:** the TwitterAAE score used in this project is a **linguistic/dialect-association proxy**. It is not a verified measure of a user's race, ethnicity, or identity. Throughout this repository, terms such as **High-AAE**, **Low-AAE**, and **AAE-associated content** refer to linguistic groups derived from the TwitterAAE model, not demographic identity labels.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Questions](#research-questions)
- [Experimental Design](#experimental-design)
- [Repository Structure](#repository-structure)
- [Data Sources](#data-sources)
- [Data Processing and Group Construction](#data-processing-and-group-construction)
- [Train, Validation, and Test Splits](#train-validation-and-test-splits)
- [Experimental Conditions](#experimental-conditions)
- [Evaluation Metrics](#evaluation-metrics)
- [Main Results](#main-results)
- [Interpretation of the Results](#interpretation-of-the-results)
- [Installation and Environments](#installation-and-environments)
- [Reproducing the Pipeline](#reproducing-the-pipeline)
- [Script Inventory](#script-inventory)
- [Prompts](#prompts)
- [Results Directory](#results-directory)
- [Reproducibility Notes](#reproducibility-notes)
- [Data and Privacy Policy](#data-and-privacy-policy)
- [Limitations](#limitations)
- [Ethical Considerations](#ethical-considerations)
- [License](#license)
- [References](#references)

---

## Project Overview

Automated hate-speech classifiers can confuse dialectal vocabulary, slang, profanity, reclaimed expressions, or informal speech with hate speech. This is especially important when models are evaluated on AAE-associated social-media language, where lexical and stylistic features may differ from the varieties of English that dominate many NLP training corpora.

This project studies that problem using a common evaluation framework built from two existing resources:

1. the **Davidson, Warmsley, Macy, and Weber 2017 (DWMW17)** hate-speech/offensive-language dataset; and
2. the **TwitterAAE** demographic language model introduced by Blodgett, Green, and O'Connor (2016).

The DWMW17 tweets provide the text and human-annotated harmful-language labels. TwitterAAE provides probabilistic language-association scores, including the African-American English component \(p_{AA}\). The two resources are combined so that classification errors can be evaluated across linguistic groups with different levels of AAE association.

The final experiments compare four model conditions:

| Condition | Model | Intervention |
|---|---|---|
| Zero-Shot | LLaMA 3 8B | Generic hate-speech classification prompt |
| Simple Dialect-Aware | LLaMA 3 8B | Explicit reminder that AAE/slang/profanity alone is not hate speech |
| Structured Decision Prompting (SDP) | LLaMA 3 8B | Structured decision procedure considering dialectal features and protected-group targeting |
| RoBERTa | `roberta-base` | Supervised fine-tuning on the study's training split |

The LLaMA model weights are held fixed across the three prompting conditions. The intervention is the prompt, not parameter updating.

---

## Research Questions

The thesis addresses the following research questions.

### RQ1

> To what extent does zero-shot LLaMA 3 8B exhibit differential false-positive rates across AAE-associated dialect groups in hate-speech classification?

### RQ2

> To what extent do simple dialect-aware prompting and Structured Decision Prompting alter classification performance and false-positive-rate disparity relative to the zero-shot condition?

### RQ3

> How do the LLaMA 3 8B prompting conditions compare with a fine-tuned RoBERTa baseline in terms of classification performance, false-positive disparity, false-negative behaviour, and overall fairness trade-offs?

---

## Experimental Design

The study is a controlled empirical comparison rather than a new model architecture, dataset, or fairness metric.

The central design principles are:

- use the same final binary task across all model conditions;
- use the same dialect-group definitions across all conditions;
- use the same final test set for LLaMA and RoBERTa comparison;
- keep LLaMA model parameters fixed while changing only the prompt intervention;
- fine-tune RoBERTa only on the training split and use the validation split for checkpoint selection;
- evaluate both overall classification performance and dialect-specific error behaviour;
- report refusals/coverage separately for generative-model conditions;
- distinguish false-positive disparity from false-negative behaviour.

The study therefore does **not** treat a single fairness metric as sufficient evidence of fairness.

---

## Repository Structure

```text
dialect-aware-hate-speech-detection/
├── README.md
├── requirements.txt
├── environment.md
├── .gitignore
├── LICENSE
├── scripts/
│   ├── 01_DWMW17_eda.py
│   ├── 02_test_twitteraae_on_dwmw17.py
│   ├── 03_score_full_dwmw17_with_twitteraae.py
│   ├── 04_dialect_aware_eda.py
│   ├── 05_prepare_modeling_dataset.py
│   ├── 06_llama_zero_shot.py
│   ├── 07_evaluate_model_zero_shot.py
│   ├── 08_llama_simple_prompt.py
│   ├── 09_evaluate_model_simple_prompt.py
│   ├── 10_llama_structured_prompt.py
│   ├── 11_evaluate_model_structured_prompt.py
│   ├── 12_compare_llama_experiments.py
│   ├── 13_train_roberta.py
│   ├── 14_run_roberta_inference.py
│   ├── 15_evaluate_model_roberta.py
│   └── 16_compare_all_models.py
├── prompts/
│   ├── zero_shot.txt
│   ├── simple_dialect_aware.txt
│   └── structured_decision.txt
├── data/
│   └── README.md
└── results/
    ├── eda/
    ├── zero_shot/
    ├── simple_prompt/
    ├── structured_prompt/
    ├── roberta/
    ├── llama_comparison/
    └── final_comparison/
```

The public repository intentionally does **not** include the tweet-level datasets, generated tweet-level prediction files, TwitterAAE model assets, or RoBERTa checkpoints. See [`data/README.md`](data/README.md) for dataset acquisition and local setup instructions.

---

## Data Sources

### 1. DWMW17 Hate-Speech and Offensive-Language Dataset

The labeled corpus is the dataset introduced in:

> Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). _Automated Hate Speech Detection and the Problem of Offensive Language._ Proceedings of the International AAAI Conference on Web and Social Media, 11(1), 512–515.

The copy used for this project was downloaded from the following Kaggle access page:

**Kaggle:**  
https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

Original paper:  
https://doi.org/10.1609/icwsm.v11i1.14955

The study refers to this corpus as **DWMW17**.

The version used contains **24,783 tweets** with the following original three-class distribution:

| Original class | Count | Percentage |
|---|---:|---:|
| Offensive language | 19,190 | 77.43% |
| Neither | 4,163 | 16.80% |
| Hate speech | 1,430 | 5.77% |
| **Total** | **24,783** | **100.00%** |

The class distribution is strongly imbalanced, with hate speech representing only about 5.77% of the observations.

### 2. TwitterAAE

Dialect association is estimated with the TwitterAAE resource introduced in:

> Blodgett, S. L., Green, L., & O'Connor, B. (2016). _Demographic Dialectal Variation in Social Media: A Case Study of African-American English._ Proceedings of EMNLP 2016, 1119–1130.

Project page used to obtain the model/resources:

**TwitterAAE:**  
https://slanglab.cs.umass.edu/TwitterAAE/

Original paper:  
https://aclanthology.org/D16-1120/

The TwitterAAE project describes its demographic language model as producing demographic dialect proportions for text, including an AAE-associated proportion. In this project, the variable of primary interest is:

```text
p_aa
```

A higher `p_aa` means that the linguistic features of the text are more strongly associated with the AAE component of the TwitterAAE model.

### What `p_aa` does not mean

`p_aa` is **not** treated as:

- a race probability;
- a verified demographic attribute;
- a self-identified racial category;
- proof that a particular author speaks AAE;
- a direct measure of individual identity.

It is used only as a linguistic association score for subgroup analysis.

### Why the data are not included here

The `data/` directory is intentionally empty apart from its documentation. The repository does not redistribute tweet-level text or third-party model assets. Users should obtain the original resources from their source pages and follow their applicable terms, research-use conditions, and citation requirements.

The TwitterAAE project page explicitly states that its released dataset/model resources are made available for research purposes and asks users to cite the associated work.

See [`data/README.md`](data/README.md) for the detailed local data setup.

---

## Data Processing and Group Construction

### TwitterAAE scoring

TwitterAAE scoring produced usable dialect-association values for **24,777 of 24,783** DWMW17 observations.

| Scoring outcome | Count |
|---|---:|
| Usable TwitterAAE score | 24,777 |
| Unusable / missing score | 6 |
| Total | 24,783 |

The six observations without usable `p_aa` values are excluded from the final dialect-based modelling dataset.

For the successfully scored observations, the `p_aa` distribution used in the study is approximately:

| Statistic | `p_aa` |
|---|---:|
| Mean | 0.4138 |
| Standard deviation | 0.2157 |
| Minimum | 0.0006 |
| 25th percentile | 0.2309 |
| Median | 0.4035 |
| 75th percentile | 0.5844 |
| Maximum | 0.9898 |

### Final dialect groups

The primary experiments use fixed thresholds rather than quartiles:

```text
Low-AAE:   p_aa <= 0.20
High-AAE:  p_aa >= 0.80
Middle:    0.20 < p_aa < 0.80
```

This produces:

| Dialect group | Count | Percentage |
|---|---:|---:|
| Low-AAE | 5,101 | 20.58% |
| Middle | 18,774 | 75.75% |
| High-AAE | 902 | 3.64% |
| Missing `p_aa` | 6 | 0.02% |

Quartile-based groups were explored during EDA but are not the primary grouping scheme used for the final experiments.

### Original labels by dialect group

The original DWMW17 class distribution differs substantially across the final dialect groups:

| Dialect group | Hate | Offensive | Neither |
|---|---:|---:|---:|
| High-AAE | 4.55% | 94.12% | 1.33% |
| Middle | 4.43% | 86.37% | 9.20% |
| Low-AAE | 10.94% | 41.60% | 47.46% |

These are properties of this particular dataset and grouping procedure. They should not be generalized to AAE speakers or demographic populations.

### Binary target construction

The original three-class task is converted into a binary hate-speech task:

```text
hate     = original hate-speech class
not_hate = original offensive-language class + original neither class
```

After removal of the six rows without usable TwitterAAE scores:

| Binary label | Count |
|---|---:|
| `hate` | 1,430 |
| `not_hate` | 23,347 |
| **Total** | **24,777** |

This binary task remains highly imbalanced.

---

## Train, Validation, and Test Splits

The final modelling dataset is split using stratification over the combination of:

```text
binary label + dialect group
```

A fixed random seed of **42** is used for split generation.

The final split sizes are:

| Split | Rows |
|---|---:|
| Training | 17,343 |
| Validation | 3,717 |
| Test | 3,717 |
| **Total** | **24,777** |

### Binary-label composition

| Split | Hate | Not hate |
|---|---:|---:|
| Train | 1,001 | 16,342 |
| Validation | 215 | 3,502 |
| Test | 214 | 3,503 |

### Dialect-group composition

| Split | Low-AAE | Middle | High-AAE |
|---|---:|---:|---:|
| Train | 3,570 | 13,141 | 632 |
| Validation | 765 | 2,817 | 135 |
| Test | 766 | 2,816 | 135 |

### Test-set subgroup support

The final test set is especially important when interpreting subgroup false-negative behaviour:

| Dialect group | Not hate | Hate | Total |
|---|---:|---:|---:|
| High-AAE | 129 | **6** | 135 |
| Middle | 2,692 | 124 | 2,816 |
| Low-AAE | 682 | 84 | 766 |

Only **six positive hate examples** occur in the High-AAE test subgroup.

This means:

- High-AAE FPR is calculated from 129 negative examples;
- High-AAE recall/FNR is calculated from only 6 positive examples;
- a single positive-example error changes High-AAE recall by roughly 16.7 percentage points;
- High-AAE positive-class metrics therefore require substantial caution.

---

## Experimental Conditions

### Condition 1 — Zero-Shot LLaMA

The zero-shot condition provides task instructions and the allowed output labels without explicitly discussing dialect.

Implementation characteristics:

| Setting | Value |
|---|---|
| Runtime | Local Ollama |
| Model identifier in script | `llama3:latest` |
| Temperature | 0 |
| Retry | None |
| Output labels | `hate`, `not_hate` |

The complete prompt is stored in:

```text
prompts/zero_shot.txt
```

### Condition 2 — Simple Dialect-Aware Prompting

The simple dialect-aware condition explicitly states that:

- AAE is a legitimate dialect of English;
- dialectal vocabulary, slang, profanity, or reclaimed language alone is not evidence of hate speech;
- `hate` should be assigned only when the text attacks or expresses hatred toward a protected group under the operational definition used by the prompt.

Implementation characteristics:

| Setting | Value |
|---|---|
| Runtime | Local Ollama `/api/generate` |
| Model identifier in script | `llama3` |
| Temperature | 0 |
| `top_p` | 1 |
| `num_predict` | 5 |
| Retry | One shortened retry if the primary output is unparseable/refusal-like |

Exact prompt:

```text
prompts/simple_dialect_aware.txt
```

### Condition 3 — Structured Decision Prompting (SDP)

The proposal initially conceptualized the third condition as Chain-of-Thought dialect prompting. In the implemented experiment, the intervention was operationalized as a **structured decision prompt** that explicitly directs the model to consider:

1. whether the tweet contains AAE-associated, slang, informal, profanity, or conversational features;
2. that those linguistic features alone are not hate speech;
3. whether the tweet attacks or expresses hatred toward a protected group;
4. the final binary label.

Internal reasoning traces were not evaluated. For that reason, this repository and the final thesis refer to the condition as **Structured Decision Prompting (SDP)** rather than Chain-of-Thought.

Implementation characteristics:

| Setting | Value |
|---|---|
| Runtime | Local Ollama `/api/generate` |
| Model identifier in script | `llama3` |
| Temperature | 0 |
| `top_p` | 1 |
| `num_predict` | 8 |
| Retry | One shortened retry if required |

Exact prompt:

```text
prompts/structured_decision.txt
```

### Condition 4 — Fine-Tuned RoBERTa

The supervised baseline uses `roberta-base` with a two-label classification head.

Final configuration:

| Setting | Value |
|---|---|
| Base checkpoint | `roberta-base` |
| Output labels | 2 |
| Maximum sequence length | 128 |
| Padding | Dynamic |
| Learning rate | `2e-5` |
| Training batch size | 16 |
| Evaluation batch size | 16 |
| Epochs | 3 |
| Weight decay | 0.01 |
| Random seed | 42 |
| Evaluation frequency | End of each epoch |
| Save frequency | End of each epoch |
| Best-model criterion | Validation positive-class F1 |
| Load best model at end | Yes |
| Save limit | 1 checkpoint |
| Explicit class weighting | None |
| Final inference batch size | 32 |

The training script does not explicitly set a separate optimizer or warm-up schedule as manually selected experimental settings. The repository therefore does not claim a custom optimizer or warm-up configuration.

No retraining is performed after inspecting final test results.

---

## Evaluation Metrics

The evaluation reports both overall classification metrics and subgroup error rates.

### Overall metrics

- **Accuracy**
- **Positive-class precision**
- **Positive-class recall**
- **Positive-class F1**

The reported F1 values in the main comparison table are **positive-class F1 for the `hate` class**, not macro-F1.

### Fairness/error metrics

For each dialect group:

- **False Positive Rate (FPR)**  
  `FP / (FP + TN)`

- **False Negative Rate (FNR)**  
  `FN / (FN + TP)`

The primary disparity summary is the signed High-AAE versus Low-AAE FPR difference:

```text
FPR gap = FPR(High-AAE) - FPR(Low-AAE)
```

Interpretation:

- positive gap → higher FPR for High-AAE content;
- zero → equal FPR between the two endpoint groups;
- negative gap → lower FPR for High-AAE than Low-AAE content.

A negative signed gap is a **direction reversal**, not proof that fairness has been achieved.

### Refusals and coverage

For LLaMA conditions, malformed outputs, refusals, or unparseable responses are tracked separately.

Primary classification metrics are computed on valid model responses rather than treating refusals as an automatic class label.

Because the number of valid responses differs between LLaMA conditions, coverage/refusal rate must be considered alongside classification metrics.

RoBERTa produces a class prediction for every test observation.

---

## Main Results

### Overall performance and FPR disparity

| Condition | Accuracy | Precision | Recall | Positive F1 | High-AAE FPR | Middle FPR | Low-AAE FPR | High-Low FPR Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-Shot | 0.2558 | 0.0639 | **0.9340** | 0.1196 | 0.9291 | 0.8519 | 0.4820 | +0.4471 |
| Simple Dialect-Aware | 0.5875 | 0.1090 | 0.8775 | 0.1939 | 0.5372 | 0.4614 | 0.2895 | **+0.2477** |
| Structured Decision Prompting | 0.6570 | 0.1246 | 0.8224 | 0.2163 | 0.5969 | 0.3674 | 0.2507 | +0.3462 |
| RoBERTa | **0.9448** | **0.5283** | 0.3925 | **0.4504** | **0.0078** | **0.0115** | **0.0630** | **-0.0553** |

### LLaMA response coverage

| Condition | Test rows | Valid responses | Refusals / invalid | Refusal rate |
|---|---:|---:|---:|---:|
| Zero-Shot | 3,717 | 3,639 | 78 | 2.10% |
| Simple Dialect-Aware | 3,717 | 3,607 | 110 | 2.96% |
| Structured Decision Prompting | 3,717 | 3,717 | 0 | 0.00% |
| RoBERTa | 3,717 | 3,717 | 0 | 0.00% |

### Zero-Shot confusion matrix

On valid Zero-Shot responses:

```text
TN = 747
FP = 2695
FN = 13
TP = 184
```

The model achieves very high hate-class recall but produces a very large number of false positives.

### Simple Dialect-Aware confusion matrix

On valid Simple Prompt responses:

```text
TN = 1940
FP = 1463
FN = 25
TP = 179
```

### Structured Decision Prompting confusion matrix

```text
TN = 2266
FP = 1237
FN = 38
TP = 176
```

SDP provides the strongest aggregate LLaMA classification performance and eliminates remaining refusal cases in the final run, but it does not produce the smallest LLaMA FPR gap.

### RoBERTa confusion matrix

```text
TN = 3428
FP = 75
FN = 130
TP = 84
```

RoBERTa substantially reduces false positives but has much lower hate-class recall than the LLaMA conditions.

### RoBERTa subgroup results

| Group | n | Accuracy | Precision | Recall | Positive F1 | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| High-AAE | 135 | 0.9481 | 0.0000 | 0.0000 | 0.0000 | 0.0078 | **1.0000** |
| Middle | 2,816 | 0.9577 | 0.5373 | 0.2903 | 0.3770 | 0.0115 | 0.7097 |
| Low-AAE | 766 | 0.8969 | 0.5275 | 0.5714 | 0.5486 | 0.0630 | 0.4286 |

For the High-AAE subgroup:

```text
TP = 0
FN = 6
FP = 1
TN = 128
```

The high subgroup accuracy is therefore largely driven by the fact that 129 of 135 High-AAE test examples are negative.

---

## Interpretation of the Results

The results show a substantial trade-off between sensitivity to hate speech and false-positive control.

### Zero-Shot LLaMA

The Zero-Shot model has the highest recall of all evaluated conditions, but this comes with extremely high false-positive rates. The High-AAE FPR is approximately 0.929, compared with approximately 0.482 for Low-AAE content.

This produces the largest signed High-Low FPR gap:

```text
+0.4471
```

The result indicates strong differential over-flagging across the study's dialect-associated groups under the Zero-Shot condition.

### Simple Dialect-Aware Prompting

Simple dialect-aware instructions substantially reduce false positives across all three dialect groups compared with Zero-Shot.

Among the three LLaMA conditions, Simple Dialect-Aware Prompting produces the smallest High-Low FPR gap:

```text
+0.2477
```

However, it also has slightly more refusals/unparseable outputs than the Zero-Shot condition in the final run.

### Structured Decision Prompting

SDP produces the best aggregate LLaMA performance:

- highest LLaMA accuracy;
- highest LLaMA precision;
- highest LLaMA positive-class F1;
- zero remaining refusals.

However, its High-Low FPR gap is:

```text
+0.3462
```

which is larger than the Simple Prompt gap.

Therefore, the study does **not** find a monotonic relationship in which increasingly structured prompting automatically reduces dialect-associated FPR disparity.

### RoBERTa

RoBERTa produces:

- the highest accuracy;
- the highest precision;
- the highest positive-class F1;
- the lowest High-AAE FPR;
- the smallest absolute High-Low FPR gap.

However, it also produces a major false-negative concern in the High-AAE subgroup: all six High-AAE positive hate examples are missed.

This means that the low High-AAE FPR should not be interpreted in isolation as equivalent fairness or as elimination of disparity. The direction of the error trade-off changed: RoBERTa strongly suppresses false positives while potentially under-detecting hate in the small High-AAE positive subset.

Because the High-AAE subgroup contains only six positive test examples, the FNR result must be interpreted cautiously.

### Overall takeaway

The experiments demonstrate why aggregate performance and one-dimensional fairness measures should be evaluated together.

In this study:

- the model with the highest recall is not the model with the lowest FPR;
- the LLaMA prompt with the best overall performance is not the LLaMA prompt with the smallest FPR disparity;
- the model with the smallest FPR disparity exhibits a serious High-AAE false-negative issue;
- prompt complexity does not produce a monotonic reduction in FPR disparity.

The repository therefore reports FPR, FNR, performance metrics, and response coverage together.

---

## Installation and Environments

The original workflow used two environments:

1. a **local macOS environment** for data processing, EDA, TwitterAAE integration, and LLaMA/Ollama inference;
2. a **Google Colab/Linux environment** for RoBERTa training and inference.

The repository keeps a single [`requirements.txt`](requirements.txt) with platform markers so that the environment-specific package versions can be documented in one place.

Detailed environment information is available in:

```text
environment.md
```

### Local LLaMA / EDA environment

Recorded versions:

```text
Python:             3.13.9
Ollama server/app:  0.31.1
Ollama Python:      0.6.2
NumPy:              1.26.4
pandas:             2.3.3
Matplotlib:         3.10.7
scikit-learn:       1.8.0
tqdm:               4.67.3
requests:           2.32.5
```

### RoBERTa / Colab environment

Recorded reproduction-environment versions:

```text
Python:             3.13.15
PyTorch:            2.11.0+cu128
Transformers:       5.15.1
Datasets:           4.0.0
Accelerate:         1.14.0
NumPy:              2.1.3
pandas:             2.2.3
scikit-learn:       1.6.1
Matplotlib:         3.10.0
GPU shown:          NVIDIA T4
```

See [`environment.md`](environment.md) for the reproducibility caveat concerning the Colab runtime capture.

### Install Python dependencies

From the repository root:

```bash
python -m pip install -r requirements.txt
```

Because `requirements.txt` mirrors the original environment split using platform markers, users reproducing the experiments on a different operating-system arrangement may need to create equivalent environments manually.

---

## Reproducing the Pipeline

### Step 1 — Clone the repository

```bash
git clone <repository-url>
cd dialect-aware-hate-speech-detection
```

Replace `<repository-url>` with the final public repository URL.

### Step 2 — Create an isolated Python environment

Example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, virtual-environment activation uses a different command.

### Step 3 — Obtain the DWMW17 dataset

Download the dataset from:

https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

Do not commit the downloaded tweet-level CSV to GitHub.

Follow [`data/README.md`](data/README.md) for the expected local file placement and naming.

> Historical filename note: the local preprocessing workflow may use the filename `DWMV17_labeled_data.csv`. The research resource itself is referred to as **DWMW17** throughout the project.

### Step 4 — Obtain TwitterAAE

Download the required TwitterAAE code/model resources from:

https://slanglab.cs.umass.edu/TwitterAAE/

The project page links to the TwitterAAE model implementation and learned model parameters.

Place the required resources locally as described in [`data/README.md`](data/README.md). Third-party TwitterAAE assets are excluded from Git through `.gitignore`.

### Step 5 — Run initial EDA

```bash
python scripts/01_DWMW17_eda.py
```

This examines the original label distribution and other descriptive characteristics of the DWMW17 data.

### Step 6 — Test TwitterAAE integration

```bash
python scripts/02_test_twitteraae_on_dwmw17.py
```

This provides a small-scale check that the TwitterAAE code, vocabulary, model parameters, and DWMW17 input can be loaded correctly before full scoring.

### Step 7 — Score the full DWMW17 corpus with TwitterAAE

```bash
python scripts/03_score_full_dwmw17_with_twitteraae.py
```

This produces the TwitterAAE probability fields used by the downstream analysis, including:

```text
p_aa
p_hispanic
p_other
p_white
```

The resulting enriched tweet-level file should remain local and should not be committed to the public repository.

### Step 8 — Run dialect-aware EDA

```bash
python scripts/04_dialect_aware_eda.py
```

This examines the `p_aa` distribution, dialect groups, class distributions across groups, annotation-agreement summaries, and related descriptive outputs.

Selected aggregate EDA outputs can be stored in:

```text
results/eda/
```

Tweet-level examples and intermediate data should remain private/local.

### Step 9 — Build the final modelling dataset

```bash
python scripts/05_prepare_modeling_dataset.py
```

The preprocessing stage creates the fixed train, validation, and test splits.

Expected generated files include:

```text
modeling_train.csv
modeling_validation.csv
modeling_test.csv
modeling_dataset_with_splits.csv
modeling_split_summary.csv
```

These tweet-level modelling files are intentionally excluded from the public repository.

### Step 10 — Prepare Ollama for LLaMA experiments

Install and start Ollama separately from the Python environment.

The scripts use local model identifiers:

```text
llama3:latest
llama3
```

The exact prompt strings are archived under `prompts/`.

**Important:** Ollama tags can change over time. Re-pulling a mutable tag in the future may not guarantee byte-for-byte identical model weights. If an exact model digest or Modelfile from the original run is available, archive that information separately for stronger model-level reproducibility.

Temperature is set to `0`, but deterministic decoding settings alone do not guarantee identical outputs across all inference-engine versions, hardware configurations, or model builds.

### Step 11 — Run Zero-Shot LLaMA

```bash
python scripts/06_llama_zero_shot.py
python scripts/07_evaluate_model_zero_shot.py
```

The inference script may produce row-level prediction/refusal files containing tweet text. These are intentionally ignored by Git.

The evaluation script produces aggregate metrics, fairness tables, confusion matrices, and figures suitable for the public `results/zero_shot/` directory.

### Step 12 — Run Simple Dialect-Aware LLaMA

```bash
python scripts/08_llama_simple_prompt.py
python scripts/09_evaluate_model_simple_prompt.py
```

### Step 13 — Run Structured Decision Prompting

```bash
python scripts/10_llama_structured_prompt.py
python scripts/11_evaluate_model_structured_prompt.py
```

### Step 14 — Compare the three LLaMA conditions

```bash
python scripts/12_compare_llama_experiments.py
```

Aggregate comparison outputs can be placed under:

```text
results/llama_comparison/
```

### Step 15 — Train RoBERTa

```bash
python scripts/13_train_roberta.py
```

The training stage uses:

```text
modeling_train.csv
modeling_validation.csv
```

and selects the best checkpoint using validation positive-class F1.

The model/checkpoint directory is intentionally excluded from Git because trained checkpoints are large and are not necessary for preserving the aggregate thesis results.

### Step 16 — Run final RoBERTa test inference

```bash
python scripts/14_run_roberta_inference.py
```

The final test input is:

```text
modeling_test.csv
```

The generated row-level prediction file should remain local.

### Step 17 — Evaluate RoBERTa

```bash
python scripts/15_evaluate_model_roberta.py
```

Aggregate results can be stored under:

```text
results/roberta/
```

### Step 18 — Compare all four model conditions

```bash
python scripts/16_compare_all_models.py
```

The final comparison produces aggregate model-performance and fairness outputs such as:

```text
all_models_comparison.csv
fairness_comparison.csv
```

along with comparison figures.

These final aggregate artifacts belong under:

```text
results/final_comparison/
```

---

## Script Inventory

| Script | Purpose |
|---|---|
| `01_DWMW17_eda.py` | Exploratory analysis of the original DWMW17 dataset |
| `02_test_twitteraae_on_dwmw17.py` | Small-sample integration check for TwitterAAE scoring |
| `03_score_full_dwmw17_with_twitteraae.py` | Scores the full corpus with TwitterAAE and creates dialect-association fields |
| `04_dialect_aware_eda.py` | EDA of `p_aa`, dialect groups, class patterns, and annotation-agreement variables |
| `05_prepare_modeling_dataset.py` | Constructs binary labels and fixed stratified train/validation/test splits |
| `06_llama_zero_shot.py` | Runs the Zero-Shot LLaMA condition |
| `07_evaluate_model_zero_shot.py` | Computes Zero-Shot overall and dialect-specific metrics |
| `08_llama_simple_prompt.py` | Runs Simple Dialect-Aware Prompting |
| `09_evaluate_model_simple_prompt.py` | Evaluates Simple Prompt outputs |
| `10_llama_structured_prompt.py` | Runs Structured Decision Prompting |
| `11_evaluate_model_structured_prompt.py` | Evaluates SDP outputs |
| `12_compare_llama_experiments.py` | Compares the three LLaMA prompting conditions |
| `13_train_roberta.py` | Fine-tunes `roberta-base` using train/validation data |
| `14_run_roberta_inference.py` | Runs final RoBERTa inference on the common test set |
| `15_evaluate_model_roberta.py` | Computes RoBERTa overall and subgroup metrics |
| `16_compare_all_models.py` | Produces final LLaMA-versus-RoBERTa comparison outputs |

The scripts are numbered according to the intended pipeline order.

---

## Prompts

The `prompts/` directory preserves the exact experimental prompt content separately from the executable Python code:

```text
prompts/
├── zero_shot.txt
├── simple_dialect_aware.txt
└── structured_decision.txt
```

This is important because prompt wording is part of the experimental condition.

The Simple and SDP prompt files also document the shortened retry instructions used when the primary model output could not be parsed into one of the two required labels.

No externally calculated `p_aa` score or dialect-group label is passed to LLaMA as a numeric feature. The dialect-aware intervention is expressed through the natural-language instruction.

---

## Results Directory

The public `results/` directory is intended for **aggregate** outputs only.

Recommended structure:

```text
results/
├── eda/
├── zero_shot/
├── simple_prompt/
├── structured_prompt/
├── roberta/
├── llama_comparison/
└── final_comparison/
```

Examples of appropriate public result files include:

```text
*_metrics.csv
*_fairness_metrics.csv
*_classification_report.csv
*_confusion_matrix.csv
*_summary.txt
*_accuracy_by_dialect.png
*_FPR_by_dialect.png
*_FNR_by_dialect.png
*_confusion_matrix.png
*_prediction_distribution.png
```

Examples of files that should remain local include:

```text
*predictions*.csv
*refusals*.csv
```

when those files contain original tweet text or raw model responses.

The repository `.gitignore` is configured to reduce the risk of accidentally committing tweet-level data, raw prediction files, model checkpoints, caches, or local environment files.

---

## Reproducibility Notes

Reproducibility for this project has several layers.

### Fixed data construction

The final experiments use:

- a fixed set of 24,777 usable scored observations;
- fixed binary-label construction;
- fixed dialect thresholds;
- a fixed train/validation/test split;
- combined label-and-dialect stratification;
- random seed 42 for split generation.

### LLaMA reproducibility

The final LLaMA experiments use local Ollama rather than a remote hosted inference API.

Recorded settings include:

- exact prompt text;
- local model identifier;
- temperature;
- `top_p` where explicitly set;
- output-token limit where explicitly set;
- retry behaviour;
- parser behaviour;
- refusal handling.

However, LLM inference can still vary across:

- model revisions under the same tag;
- Ollama versions;
- quantization/model builds;
- operating systems;
- hardware;
- inference-engine implementation changes.

Therefore, the repository supports strong **procedural reproducibility**, while exact response-level replication additionally depends on access to the same underlying model build.

### RoBERTa reproducibility

RoBERTa uses fixed seeds for:

- Python;
- NumPy;
- PyTorch;
- CUDA where applicable.

The number of training epochs is fixed at three, and best-checkpoint selection depends only on validation positive-class F1.

No class weighting is used.

No test-set threshold optimization is used.

GPU training may still contain nondeterministic operations, so fixed seeds substantially improve reproducibility but do not mathematically guarantee identical floating-point results on every hardware/software stack.

### Test-set development caveat

The final comparison uses a common test set across all four conditions.

The RoBERTa model is trained and checkpoint-selected without retraining on the final test set. However, LLaMA prompt behaviour was inspected iteratively during prompt development. The final test set should therefore be described as the **common final evaluation set**, rather than making a stronger claim that it was untouched throughout all prompt-development activity.

### Descriptive rather than inferential claims

The reported subgroup comparisons are descriptive.

The repository does not claim statistical significance for differences between model conditions unless a separate inferential analysis is explicitly added.

This is particularly important for the High-AAE positive subgroup because the final test set contains only six positive examples in that group.

---

## Data and Privacy Policy

This repository intentionally avoids redistributing tweet-level content.

The following are excluded from the public repository by default:

- original DWMW17 tweet-level CSV files;
- enriched DWMW17 + TwitterAAE tweet-level files;
- train/validation/test tweet-level split files;
- row-level LLaMA prediction files;
- row-level refusal files;
- row-level RoBERTa prediction files;
- third-party TwitterAAE model/code assets where redistribution has not been separately established;
- RoBERTa model checkpoints.

Researchers who reproduce the study should obtain source data independently and comply with the source providers' current terms and applicable research-ethics requirements.

Aggregate statistics, confusion matrices, fairness metrics, and plots are included because they do not require publishing the underlying tweet text.

---

## Limitations

Several limitations should be considered when interpreting or reproducing this work.

### 1. TwitterAAE is a proxy

The dialect score is inferred from linguistic patterns learned from geolocated social-media data. It does not establish a writer's race or self-identified dialect.

### 2. High-AAE positive support is very small

Only six High-AAE positive hate examples occur in the final test set. High-AAE FNR, recall, and positive-class F1 are therefore unstable and should not be generalized broadly.

### 3. Dataset composition is not population prevalence

The DWMW17 dataset was constructed for harmful-language research and has a particular sampling and annotation process. Its class and dialect distributions should not be interpreted as prevalence estimates for Twitter/X users, AAE speakers, or any demographic population.

### 4. Binary-label simplification

The study collapses the original `offensive` and `neither` classes into `not_hate`. This is appropriate for the research question but removes distinctions present in the original annotation task.

### 5. Prompt conditions do not isolate every possible factor

Although the LLaMA weights remain fixed, the prompt conditions differ in wording, structure, retry behaviour, and output constraints. The experiment measures the behaviour of these implemented conditions, not an abstract universal effect of "dialect awareness."

### 6. Refusal subsets differ

Zero-Shot and Simple Prompt metrics are computed over different valid-response subsets because refusal/unparseable counts differ. Coverage must therefore be considered when comparing their aggregate metrics.

### 7. Model tags may be mutable

An Ollama tag such as `llama3` may resolve to a different model build in the future. Model identifier preservation is useful but weaker than preserving an immutable model digest.

### 8. Software/hardware nondeterminism

RoBERTa training and LLM inference may exhibit small differences across hardware and software environments even when seeds and decoding settings are fixed.

### 9. No fairness claim from one metric

A low FPR gap does not imply complete fairness. FNR, subgroup support, overall performance, and error direction must also be examined.

---

## Ethical Considerations

Hate-speech datasets contain potentially harmful, offensive, discriminatory, and identity-targeting language.

Users of this repository should:

- avoid unnecessary redistribution of tweet text;
- avoid presenting example tweets when aggregate analysis is sufficient;
- respect the terms and citation requirements of the original data providers;
- avoid treating inferred dialect association as verified racial identity;
- avoid generalizing dataset-level patterns to demographic populations;
- interpret subgroup results in light of sample size and dataset construction;
- consider both over-moderation and under-detection harms.

The purpose of the dialect-group analysis is to evaluate model error behaviour across **linguistic association groups**, not to infer personal identity.

---

## License

The repository's `LICENSE` applies to the original code and repository materials authored for this project, subject to the terms stated in that file.

It does **not** transfer or replace the licenses, terms, copyrights, model licenses, dataset conditions, or usage restrictions of third-party resources, including:

- DWMW17 / the Davidson et al. dataset;
- TwitterAAE;
- LLaMA model weights;
- Ollama;
- RoBERTa / Hugging Face model assets;
- PyTorch, Transformers, and other external software dependencies.

Users are responsible for reviewing and complying with the original third-party terms.

---

## Author

**Aakash Vashist**  
M.Sc. Big Data & Artificial Intelligence  
SRH University of Applied Sciences, Leipzig, Germany  

Master’s thesis:  
*Mitigating Racial Bias in Hate Speech Detection through Dialect-Aware Prompting*

