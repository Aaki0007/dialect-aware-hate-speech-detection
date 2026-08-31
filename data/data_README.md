# Data

This directory is intentionally kept free of tweet-level data in the public repository.

The experiments in this project use two external research resources:

1. the **Hate Speech and Offensive Language dataset** introduced by Davidson, Warmsley, Macy, and Weber (2017), referred to in the thesis as **DWMW17**; and
2. the **TwitterAAE demographic language model** introduced by Blodgett, Green, and O'Connor (2016).

The original data and third-party model files are **not redistributed here**. This README explains where they were obtained, how they should be placed locally, how the project transforms them, and the checks that can be used to verify a reproduction.

> **Content warning:** the DWMW17 dataset contains racist, sexist, homophobic, profane, and otherwise offensive language. It also contains social-media text that may include usernames or other user-generated content. Handle the data accordingly.

---

## 1. Required local directory structure

After downloading the external resources, the local `data/` directory should look like this:

```text
data/
├── README.md
├── DWMV17_labeled_data.csv
└── twitteraae/
    ├── code/
    │   └── ...
    └── model/
        ├── model_count_table.txt
        ├── model_vocab.txt
        └── ...
```

The CSV and `twitteraae/` directory are ignored by Git and should remain local.

### Important filename note

The thesis refers to the Davidson et al. dataset as **DWMW17**, based on the authors Davidson, Warmsley, Macy, and Weber.

The local filename expected by the project scripts is:

```text
DWMV17_labeled_data.csv
```

This filename is retained for compatibility with the implemented pipeline even though the thesis uses the abbreviation **DWMW17**.

---

# 2. DWMW17 hate-speech dataset

## Source used in this project

The dataset was downloaded from the following Kaggle page:

https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

The underlying dataset originates from:

**Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017).  
Automated Hate Speech Detection and the Problem of Offensive Language.  
Proceedings of the 11th International AAAI Conference on Web and Social Media (ICWSM 2017), 512-515.**

Original project repository:

https://github.com/t-davidson/hate-speech-and-offensive-language

Paper:

https://arxiv.org/abs/1703.04009

The Kaggle page was used as the download location for this thesis; Davidson et al. (2017) should be cited as the dataset's original research source.

---

## Dataset size

The version used in this project contains:

| Original class | Count | Percentage |
|---|---:|---:|
| Hate speech | 1,430 | 5.77% |
| Offensive language | 19,190 | 77.43% |
| Neither | 4,163 | 16.80% |
| **Total** | **24,783** | **100.00%** |

The strong imbalance is important when interpreting model accuracy and subgroup error rates.

---

## Original class encoding

The original `class` variable is interpreted as:

| Value | Meaning |
|---:|---|
| `0` | hate speech |
| `1` | offensive language |
| `2` | neither |

The source data also contain crowd-annotation vote counts. Depending on the downloaded CSV version, the principal fields include:

| Field | Description |
|---|---|
| `count` | Number of annotators who coded the tweet |
| `hate_speech` | Number of annotators assigning hate speech |
| `offensive_language` | Number of annotators assigning offensive language |
| `neither` | Number of annotators assigning neither |
| `class` | Majority class label |
| `tweet` | Original tweet text |

Some CSV versions also contain an unnamed index column. The preprocessing code handles this as an original row identifier.

---

# 3. Binary hate-speech target used in this project

The experiments do **not** use the original three-class task directly.

The final binary target is:

```text
hate      = original class 0
not_hate  = original classes 1 and 2
```

In other words:

```text
hate speech                         -> hate
offensive language                  -> not_hate
neither                             -> not_hate
```

This distinction is central to the study: offensive language, profanity, slang, or dialectal vocabulary alone is not treated as hate speech.

After TwitterAAE scoring and exclusion of six rows without usable dialect scores, the final modeling dataset contains:

| Binary class | Count |
|---|---:|
| `hate` | 1,430 |
| `not_hate` | 23,347 |
| **Total** | **24,777** |

---

# 4. TwitterAAE

## Official source used in this project

TwitterAAE was obtained from the official SLANG Lab page at the University of Massachusetts Amherst:

https://slanglab.cs.umass.edu/TwitterAAE/

The page provides both the TwitterAAE corpus and the demographic language model implementation.

The model is associated with:

**Blodgett, S. L., Green, L., & O'Connor, B. (2016).  
Demographic Dialectal Variation in Social Media: A Case Study of African-American English.  
Proceedings of EMNLP 2016.**

The official TwitterAAE page links to the model implementation here:

https://github.com/slanglab/twitteraae

For this project, the important component is the **TwitterAAE model implementation and learned parameters** used to calculate demographic language proportions. The complete multi-gigabyte TwitterAAE tweet corpus is not required to reproduce the scoring step used in this thesis.

The official TwitterAAE site states that the dataset/model materials are provided for research purposes and asks researchers to cite the relevant Blodgett et al. work. Users of this repository should follow the current terms stated by the original source.

---

# 5. What the TwitterAAE model produces

For each usable tweet, the implementation produces a probability vector containing demographic language components. The project retains:

```text
p_aa
p_hispanic
p_other
p_white
```

The main variable used for subgroup analysis is:

```text
p_aa
```

## Interpretation of `p_aa`

`p_aa` is used as a **linguistic association score**.

A higher value means that the linguistic patterns in the text are more strongly associated with the African-American language component learned by the TwitterAAE model.

It must **not** be interpreted as:

- the probability that the author is Black or African American;
- self-identified race;
- verified racial identity;
- a biological or demographic classification of an individual.

Accordingly, this repository uses terms such as **AAE-associated content**, **AAE-associated linguistic group**, and **dialect-associated disparity**.

---

# 6. TwitterAAE files expected by the scripts

The scoring scripts expect the TwitterAAE implementation to be available locally under:

```text
data/twitteraae/
```

with code and model assets separated approximately as:

```text
data/twitteraae/
├── code/
│   ├── predict.py
│   └── ...
└── model/
    ├── model_count_table.txt
    ├── model_vocab.txt
    └── ...
```

The two model files explicitly used by the scoring pipeline are:

```text
model_count_table.txt
model_vocab.txt
```

Do not commit these third-party assets to this repository unless their original distribution terms explicitly permit doing so.

---

# 7. TwitterAAE scoring outcome

The DWMW17 dataset contains 24,783 rows.

TwitterAAE produced usable scores for:

```text
24,777 rows
```

and did not produce a usable score for:

```text
6 rows
```

Those six observations are excluded from the final dialect-based modeling dataset.

A successful reproduction should therefore obtain a final usable modeling population of:

```text
24,777
```

before train/validation/test splitting.

---

# 8. Dialect-group construction

The final experiments use fixed thresholds based on `p_aa`.

```text
Low-AAE   : p_aa <= 0.20
Middle    : 0.20 < p_aa < 0.80
High-AAE  : p_aa >= 0.80
```

The expected group counts across all 24,777 usable observations are:

| Dialect-associated group | Count |
|---|---:|
| Low-AAE | 5,101 |
| Middle | 18,774 |
| High-AAE | 902 |
| **Total** | **24,777** |

These groups are analytical categories derived from the TwitterAAE linguistic score. They are not verified racial groups.

Quartile-based groupings were explored during EDA, but the final experiments use the fixed `0.20` and `0.80` thresholds above.

---

# 9. Train, validation, and test splits

The final modeling data are divided into:

```text
70% training
15% validation
15% test
```

with random seed:

```text
42
```

Splitting is stratified using the combination of:

```text
binary hate-speech label + dialect group
```

This helps preserve both class and dialect-group composition across the partitions.

Expected split sizes:

| Split | Rows | Hate | Not hate |
|---|---:|---:|---:|
| Training | 17,343 | 1,001 | 16,342 |
| Validation | 3,717 | 215 | 3,502 |
| Test | 3,717 | 214 | 3,503 |

Expected dialect composition:

| Split | Low-AAE | Middle | High-AAE |
|---|---:|---:|---:|
| Training | 3,570 | 13,141 | 632 |
| Validation | 765 | 2,817 | 135 |
| Test | 766 | 2,816 | 135 |

The common final evaluation set therefore contains **3,717 observations**.

---

# 10. Generated local files

The preprocessing pipeline creates derived CSV files locally. These files contain tweet-level data and are intentionally excluded from the public Git repository.

Important generated files include:

```text
modeling_train.csv
modeling_validation.csv
modeling_test.csv
modeling_dataset_with_splits.csv
modeling_split_summary.csv
```

Intermediate stages may also create files containing TwitterAAE scores and EDA-derived fields, for example:

```text
dwmw17_with_twitteraae_scores.csv
dwmw17_with_twitteraae_scores_and_agreement.csv
twitteraae_scoring_summary.csv
```

The final modeling files retain fields needed for model training, inference, and subgroup evaluation, including:

| Field | Purpose |
|---|---|
| `original_index` | Original dataset row identifier |
| `tweet` | Tweet text used as model input |
| `class` | Original DWMW17 class |
| `class_name` | Human-readable original class |
| `binary_hate_label` | Numeric binary target |
| `binary_hate_label_name` | `hate` or `not_hate` |
| `p_aa` | TwitterAAE AAE-associated language score |
| `p_hispanic` | TwitterAAE Hispanic-associated component |
| `p_other` | TwitterAAE other component |
| `p_white` | TwitterAAE white-associated component |
| `dialect_group` | `low_aae`, `middle`, or `high_aae` |
| `aae_quartile_group` | EDA/robustness grouping |
| `agreement_ratio` | Annotation-agreement measure |
| `vote_margin` | Difference between leading annotation vote counts |
| `max_vote` | Largest annotation vote count |
| `split` | train, validation, or test |

Not every intermediate file necessarily contains every field.

---

# 11. Reproducing the data pipeline

From the repository root, the data-related scripts are intended to be executed in numerical order:

```text
01_DWMW17_eda.py
02_test_twitteraae_on_dwmw17.py
03_score_full_dwmw17_with_twitteraae.py
04_dialect_aware_eda.py
05_prepare_modeling_dataset.py
```

Their roles are:

| Script | Purpose |
|---|---|
| `01_DWMW17_eda.py` | Inspect the original dataset, label balance, annotation agreement, and descriptive text characteristics |
| `02_test_twitteraae_on_dwmw17.py` | Verify that the locally installed TwitterAAE model can score sample DWMW17 tweets |
| `03_score_full_dwmw17_with_twitteraae.py` | Score the complete dataset and attach TwitterAAE probabilities |
| `04_dialect_aware_eda.py` | Examine the score distribution and class composition across dialect-associated groups |
| `05_prepare_modeling_dataset.py` | Construct binary labels and generate the stratified train/validation/test partitions |

Before continuing to model inference, verify that the generated split counts match the values documented above.

---

# 12. Why the data are not stored in this repository

Tweet-level files are deliberately excluded for several reasons:

1. **Source ownership and distribution terms**  
   The original data and TwitterAAE assets are third-party research resources and remain subject to their own terms.

2. **Potentially sensitive social-media text**  
   The datasets contain user-generated text and potentially identifying usernames or references.

3. **Harmful-language content**  
   The dataset contains hate speech and other highly offensive material.

4. **Reproducibility does not require republication**  
   The source locations, expected local layout, preprocessing scripts, fixed thresholds, split procedure, and aggregate verification counts are documented here so that researchers can reconstruct the analysis from the original resources.

The repository `.gitignore` therefore excludes tweet-level CSV files and the local `data/twitteraae/` directory.

---

# 13. Data limitations

Several limitations should be kept in mind when reproducing or extending this work.

### Historical and platform-specific data

DWMW17 reflects a particular Twitter sampling strategy, annotation process, and historical period. Results should not automatically be generalized to current social-media content or other platforms.

### Annotation subjectivity

Hate speech and offensive language are difficult categories to annotate consistently. The original dataset therefore includes annotation-vote information, and disagreement is itself informative.

### Severe class imbalance

The positive hate class is a small minority of the data. Accuracy alone is therefore insufficient for evaluating classifier quality.

### TwitterAAE is a proxy for language association

TwitterAAE estimates demographic/dialectal language associations from text. It does not provide verified demographic identity.

### Small High-AAE positive-class support

The final test set contains only six `hate` observations in the High-AAE group. Metrics such as High-AAE recall and false-negative rate therefore have very small positive-class support and should be interpreted cautiously.

---

# 14. Citation

If you use the DWMW17 dataset, cite:

```text
Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017).
Automated Hate Speech Detection and the Problem of Offensive Language.
Proceedings of the 11th International AAAI Conference on Web and Social Media
(ICWSM 2017), 512-515.
```

If you use the TwitterAAE demographic language model, cite:

```text
Blodgett, S. L., Green, L., & O'Connor, B. (2016).
Demographic Dialectal Variation in Social Media:
A Case Study of African-American English.
Proceedings of the 2016 Conference on Empirical Methods
in Natural Language Processing (EMNLP).
```

Researchers should also consult the original source pages for the latest access and usage conditions:

- DWMW17 download used in this project:  
  https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

- Davidson et al. original repository:  
  https://github.com/t-davidson/hate-speech-and-offensive-language

- TwitterAAE official page:  
  https://slanglab.cs.umass.edu/TwitterAAE/

- TwitterAAE model repository:  
  https://github.com/slanglab/twitteraae

---

# 15. Quick reproducibility checklist

Before running the modeling experiments, verify:

```text
[ ] DWMV17_labeled_data.csv exists under data/
[ ] The raw dataset contains 24,783 rows
[ ] data/twitteraae/code/ is present
[ ] data/twitteraae/model/model_count_table.txt is present
[ ] data/twitteraae/model/model_vocab.txt is present
[ ] TwitterAAE test scoring succeeds
[ ] Full scoring leaves 24,777 usable observations
[ ] Low-AAE count is 5,101
[ ] Middle count is 18,774
[ ] High-AAE count is 902
[ ] modeling_train.csv contains 17,343 rows
[ ] modeling_validation.csv contains 3,717 rows
[ ] modeling_test.csv contains 3,717 rows
```

If these checks hold, the dataset preparation stage is aligned with the final experimental setup used in the thesis.
