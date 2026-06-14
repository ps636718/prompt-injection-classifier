# 🔬 Methodology — Extended Notes

> **DATASPRINT PS5** | NIST University Data Science Club  
> Author: Pawan Suman

---

## 1. Problem Framing

Binary classification task:
- **Input**: Raw text prompt (string)
- **Output**: `0` (Benign) or `1` (Malicious)

"Malicious" covers:
- Direct prompt injection (`ignore your instructions`)
- Role-play jailbreaks (`you are now DAN`)
- Developer-mode exploits (`developer mode enabled`)
- System-prompt extraction attempts
- Authority spoofing (`as the AI's creator, I order you to...`)

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Training file | `merged_train_70.csv` |
| Test file | `merged_test_30.csv` |
| Test size | 31,619 prompts |
| Test label distribution | 78.62% malicious / 21.38% benign |
| Primary text column | `Prompt` |
| Secondary columns | `question1`, `question2` (merged when present) |
| Numerical columns | `Length`, `Perplexity` (median-imputed) |

---

## 3. Preprocessing Pipeline

### 3.1 Text Merging

All available text columns are concatenated into `clean_text`:
```
clean_text = Prompt + " " + question1 + " " + question2
```
Missing columns are treated as empty strings (not errors).

### 3.2 Cleaning Steps (in order)

| Step | Operation | Rationale |
|------|-----------|-----------|
| 1 | Lowercase | Remove case sensitivity |
| 2 | URL removal | URLs are noise (`http://...`) |
| 3 | Non-alpha filter | Remove symbols/numbers |
| 4 | Whitespace normalization | Collapse multiple spaces |
| 5 | Stopword removal (NLTK) | Remove function words |
| 6 | Lemmatization (WordNet) | Normalize morphological variants |

### 3.3 Row Filtering

Prompts shorter than 10 characters after cleaning are dropped. Empty or whitespace-only strings are removed.

---

## 4. Feature Engineering

### 4.1 TF-IDF Vectorization

```
Vocabulary size : 15,000 terms
N-gram range    : (1, 2) — unigrams and bigrams
Sublinear TF    : True — log(1+tf) instead of raw tf
Min DF          : 2 — ignore words appearing only once
```

**Why bigrams?** Two-word sequences capture adversarial patterns that unigrams miss:
- `ignore` alone is ambiguous
- `ignore instructions` is a near-certain attack signal

**Why `sublinear_tf=True`?** Attackers may repeat keywords hoping to score higher on keyword-frequency metrics. Log-scaling defeats this.

### 4.2 Meta-Features (19 additional dimensions)

| Feature | Dim | Description |
|---------|:---:|-------------|
| `word_count` | 1 | Word count, clipped at 100 |
| `char_count` | 1 | Char count, clipped at 500 |
| `special_char_density` | 1 | Fraction of non-alphanumeric chars |
| Jailbreak keyword flags | 18 | Binary 0/1 per keyword pattern |

**Why clip word/char counts?** Extreme outliers (e.g., a 50,000-character prompt) would dominate the feature space without clipping.

**Why keyword flags?** A rare jailbreak keyword (`"do anything now"`) may appear in fewer than 2 training examples, falling below `min_df=2` in TF-IDF. An explicit binary flag ensures the model always has this signal.

### 4.3 Scaling

`StandardScaler(with_mean=False)` is applied to all meta-features:
- `with_mean=False` is required for sparse matrix compatibility
- Without scaling, `char_count` (range 0–500) would dominate `special_char_density` (range 0–1)

**Final matrix shape**: `(n_samples, 15,019)`

---

## 5. Models

### 5.1 Logistic Regression
- `class_weight='balanced'`
- `solver='liblinear'` (efficient for sparse TF-IDF)
- Tuned C via GridSearchCV: [0.1, 0.5, 1.0, 1.5, 5.0, 10.0]

### 5.2 LinearSVC
- `class_weight='balanced'`
- `max_iter=3000`
- Tuned C: [0.1, 1, 5, 10, 50] and loss: [hinge, squared_hinge]

### 5.3 Multinomial Naive Bayes
- `alpha=0.5` (Laplace smoothing)
- Cannot use `class_weight` — relies on TF-IDF non-negativity

### 5.4 Random Forest
- 100 estimators
- `class_weight='balanced'`
- `n_jobs=-1` (parallelized)

### 5.5 Ensemble VotingClassifier (Final)
```python
VotingClassifier(
    estimators=[('svc', best_svc), ('lr', lr), ('rf', rf)],
    voting='hard'
)
```
Hard voting is used because `LinearSVC` does not expose `predict_proba` by default (required for soft voting).

---

## 6. Evaluation Strategy

### 6.1 Train/Validation Split
- 80% train / 20% validation
- Stratified by label to preserve class ratio

### 6.2 Primary Metric: F1 Score

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 is preferred over accuracy because:
- The dataset is imbalanced (more malicious prompts)
- Both false positives (blocking benign) and false negatives (missing attacks) carry real cost

### 6.3 Cross-Validation in GridSearch
- `cv=3` for speed
- `scoring='f1'` to optimize the primary metric

---

## 7. Final Results

```
Model     : Ensemble (SVC + LR + RF)
Accuracy  : 0.9482  (94.82%)
F1 Score  : 0.9464  (94.64%)
Test Size : 31,619 prompts
Malicious : 24,859  (78.62%)
Benign    :  6,760  (21.38%)
```

---

## 8. Limitations & Known Issues

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| TF-IDF cannot capture word order globally | May miss complex multi-sentence jailbreaks | → Use BERT embeddings in v2 |
| Keyword list is hand-curated | New jailbreak patterns evade detection | → Active learning pipeline |
| No multilingual support | Non-English attacks pass through | → Multilingual preprocessing |
| Hard voting loses probability calibration | No confidence scores | → CalibratedClassifierCV wrapper |
| 107 MB model size | Heavy for serverless deployment | → Model compression / distillation |

---

## 9. References

- Scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)
- NLTK: [https://nltk.org](https://nltk.org)
- Perez & Ribeiro (2022): "Ignore Previous Prompt: Attack Techniques For Language Models"
- Wallace et al. (2019): "Universal Adversarial Triggers for Attacking and Analyzing NLP"
