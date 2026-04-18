# 📊 Statistical Terms Glossary — COVID-19 Survival Prediction Project

> **Purpose:** This document explains every statistical and machine-learning term used in the project pipeline (`pipeline.py`), the Streamlit dashboard (`app.py`), and the model outputs (`output.txt`, `run_log.txt`, `clinical_interpretation.txt`).  
> Each term includes: a plain-language definition, where it appears in the code, and the actual value from the project's output.

---

## Table of Contents

1. [Dataset & Sampling Terms](#1-dataset--sampling-terms)
2. [Feature Engineering Terms](#2-feature-engineering-terms)
3. [Preprocessing & Scaling Terms](#3-preprocessing--scaling-terms)
4. [Class Imbalance Terms](#4-class-imbalance-terms)
5. [Model Terms](#5-model-terms)
6. [Classification Threshold Terms](#6-classification-threshold-terms)
7. [Confusion Matrix Terms](#7-confusion-matrix-terms)
8. [Core Classification Metrics](#8-core-classification-metrics)
9. [Aggregate / Averaging Metrics](#9-aggregate--averaging-metrics)
10. [Ranking / Discrimination Metrics](#10-ranking--discrimination-metrics)
11. [Calibration Terms](#11-calibration-terms)
12. [Feature Importance & Explainability Terms](#12-feature-importance--explainability-terms)
13. [Logistic Regression Specific Terms](#13-logistic-regression-specific-terms)
14. [Clinical / Domain-Specific Terms](#14-clinical--domain-specific-terms)
15. [Descriptive Statistics Terms](#15-descriptive-statistics-terms)
16. [High-Level Model Comparison Summary](#16-high-level-model-comparison-summary)

---

## 1. Dataset & Sampling Terms

### 1.1 Train-Test Split

| Item | Detail |
|------|--------|
| **Definition** | Divides the dataset into a **training set** (used to learn patterns) and a **test set** (used to measure real-world performance on unseen data). Standard practice to detect overfitting. |
| **Code location** | `pipeline.py` — `step5_preprocess()`, line: `train_test_split(X, y, test_size=0.20, stratify=y, ...)` |
| **Parameters used** | `test_size=0.20` → 80% train, 20% test; `stratify=y` → preserves class proportions |
| **Actual output** | `Train: 7,284 patients — Test: 1,822 patients` (from `run_log.txt` line 65) |

---

### 1.2 Stratified Split

| Item | Detail |
|------|--------|
| **Definition** | A variant of train-test split that ensures each split **contains the same proportion of each class** as the full dataset. Critical when classes are imbalanced so the test set isn't accidentally all-survivors. |
| **Code location** | `pipeline.py` line 272: `stratify=y` argument in `train_test_split()` |
| **Why it matters** | Without stratification on a 24.7:1 imbalanced dataset, the 20% test set could end up with very few (or zero) "Died" patients, making evaluation meaningless. |
| **Actual output** | Test set correctly contained **71 Died + 1,751 Survived** — proportional to the original 3.9% death rate. |

---

### 1.3 Random State / Random Seed

| Item | Detail |
|------|--------|
| **Definition** | A fixed integer seed passed to all random processes so results are **exactly reproducible** across runs. |
| **Code location** | `pipeline.py` line 40: `RANDOM_STATE = 42`; used in `train_test_split`, `SMOTE`, all three models |
| **Actual output** | All results in `output.txt` will be identical on every re-run as long as data doesn't change. |

---

## 2. Feature Engineering Terms

### 2.1 Binary Feature (One-hot encoded flag)

| Item | Detail |
|------|--------|
| **Definition** | A feature that takes only the values **0 or 1**, encoding the absence or presence of a condition. |
| **Code location** | `pipeline.py` `feat_comorbidities()` — e.g., `pre_covid["has_diabetes"] = desc.str.contains("diabetes").astype(int)` |
| **Features created** | `has_hypertension`, `has_diabetes`, `has_obesity`, `has_asthma`, `has_COPD`, `has_heart_disease`, `was_hospitalized`, `is_icu`, `had_care_plan`, `low_spo2`, `high_crp`, `high_d_dimer`, `low_lymphocytes` |

---

### 2.2 Derived Binary Threshold Flags

| Item | Detail |
|------|--------|
| **Definition** | Binary (0/1) features derived from a continuous lab value by applying a **clinical cut-point**. Converts a raw measurement into a clinically meaningful signal. |
| **Code location** | `pipeline.py` lines 180–183 in `feat_labs()` |
| **Thresholds used** | `low_spo2 = (spo2 < 94)`, `high_crp = (crp > 10)`, `high_d_dimer = (d_dimer > 500)`, `low_lymphocytes = (lymphocytes < 1.0)` |
| **Clinical basis** | SpO2 < 94% = hypoxia definition (WHO); CRP > 10 mg/L = active inflammation; D-Dimer > 500 ng/mL = thrombotic risk; Lymphocytes < 1.0×10³/µL = lymphopenia |

---

### 2.3 Comorbidity Count

| Item | Detail |
|------|--------|
| **Definition** | A **count feature** summing all active binary comorbidity flags per patient. Captures additive disease burden in a single numerical feature. |
| **Code location** | `pipeline.py` line 138: `agg["comorbidity_count"] = agg[cols].sum(axis=1)` |
| **Descriptive stats** | Mean = 1.04, Max = 5 comorbidities per patient (from `run_log.txt` descriptive statistics table) |

---

### 2.4 Median Imputation

| Item | Detail |
|------|--------|
| **Definition** | Filling in **missing values** with the **median** of the observed values for that lab. The median is used instead of the mean because it is robust to extreme outliers (common in lab data). |
| **Code location** | `pipeline.py` lines 175–178 in `feat_labs()`: `median_val = lab_df[lab].median(); lab_df[lab] = lab_df[lab].fillna(median_val)` |
| **Actual values imputed** | `d_dimer: 7,353 missing → median = 0.300`; `ferritin: 7,353 missing → median = 423.300`; `crp: 7,353 missing → median = 10.200`; `wbc: 6,206 missing → median = 3.800` |
| **Zero imputation** | `il6`, `spo2`, `lymphocytes` had **no data at all** → imputed with 0 (flagged as WARNING in `run_log.txt`) |

---

### 2.5 Feature Matrix

| Item | Detail |
|------|--------|
| **Definition** | The rectangular table of shape **(patients × features)** that is fed into the machine learning models. Each row is one patient; each column is one feature. Also called the design matrix or **X**. |
| **Code location** | `pipeline.py` `step4_build_feature_matrix()` and `step5_preprocess()` |
| **Actual shape** | `(9,106 patients × 23 features)` (from `run_log.txt` line 63) |
| **Feature list (23)** | age, gender, has_hypertension, has_diabetes, has_obesity, has_asthma, has_COPD, has_heart_disease, comorbidity_count, d_dimer, il6, ferritin, crp, spo2, wbc, lymphocytes, low_spo2, high_crp, high_d_dimer, low_lymphocytes, was_hospitalized, is_icu, had_care_plan |

---

## 3. Preprocessing & Scaling Terms

### 3.1 Standard Scaler (Z-score Normalization)

| Item | Detail |
|------|--------|
| **Definition** | Transforms each continuous feature to have **mean = 0 and standard deviation = 1** using the formula: `z = (x − μ) / σ`. Prevents features with large numeric ranges (e.g., ferritin up to 1,197) from dominating features with small ranges (e.g., age 0–110). |
| **Formula** | `z = (value − training_mean) / training_std` |
| **Code location** | `pipeline.py` lines 277–283: `scaler = StandardScaler(); X_train_scaled[:, cont_idx] = scaler.fit_transform(...)` |
| **Applied to** | `age`, `d_dimer`, `il6`, `ferritin`, `crp`, `spo2`, `wbc`, `lymphocytes`, `comorbidity_count` (the 9 continuous features) |
| **Important** | The scaler is **fit only on training data** (`fit_transform`) and then **applied (not re-fit)** to test data (`transform`) — this prevents data leakage. |
| **Saved artifact** | `survival_scaler.pkl` (so the same scaling can be applied to new patients at inference time in `app.py`) |

---

### 3.2 Fit vs. Transform (Data Leakage Prevention)

| Item | Detail |
|------|--------|
| **Definition** | **Fitting** = computing the statistics (mean, std) from data. **Transforming** = applying those pre-computed statistics to scale data. Fitting on training data only and only transforming test data prevents the model from "peeking" at test-set information during training. |
| **Code location** | `pipeline.py` line 282: `scaler.fit_transform(X_train...)` vs. line 283: `scaler.transform(X_test...)` |

---

## 4. Class Imbalance Terms

### 4.1 Class Imbalance

| Item | Detail |
|------|--------|
| **Definition** | When one class in the target variable has **far more examples** than the other. In clinical datasets, this is typical because most patients survive. A naive model can achieve high accuracy by predicting the majority class for everyone. |
| **In this project** | **96.1% Survived (8,752) vs. 3.9% Died (354)** — a 24.7:1 ratio (from `run_log.txt` line 20) |
| **Problem** | A dummy model that always predicts "Survived" would score 96.1% accuracy — misleadingly high. The metrics that actually matter are **Recall(Died)**, **PR-AUC**, and **F1(Died)**. |

---

### 4.2 SMOTE (Synthetic Minority Over-sampling Technique)

| Item | Detail |
|------|--------|
| **Definition** | Creates **synthetic** (artificial, not duplicated) minority-class samples by interpolating between existing minority examples in feature space. Balances the training class distribution so models don't overfit to the majority class. |
| **Code location** | `pipeline.py` lines 287–290: `sm = SMOTE(random_state=RANDOM_STATE); X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)` |
| **Applied on** | Training set **only** — never on test data (which must reflect real-world distribution) |
| **Actual result** | `Before SMOTE: 7,001 Survived + 283 Died` → `After SMOTE: 7,001 Survived + 7,001 Died` (from `run_log.txt` line 66) |
| **Library** | `imblearn.over_sampling.SMOTE` |

---

### 4.3 Class Weight Balancing (`class_weight="balanced"`)

| Item | Detail |
|------|--------|
| **Definition** | An alternative/complement to SMOTE. Tells the model to assign **higher penalty for misclassifying minority class** samples during training. The weight for each class is inversely proportional to its frequency: `w = n_samples / (n_classes × class_count)`. |
| **Code location** | `pipeline.py` lines 303, 307: `class_weight="balanced"` in Logistic Regression and Random Forest |
| **Effect** | Errors on "Died" patients are penalized more than errors on "Survived" patients, pushing the model to avoid missing deaths. |

---

### 4.4 scale_pos_weight (XGBoost imbalance handling)

| Item | Detail |
|------|--------|
| **Definition** | XGBoost's equivalent of class weighting: the ratio of negative (survived) to positive (died) examples. Sets how much extra weight to give the minority class during gradient descent. |
| **Formula** | `scale_pos_weight = count(survived) / count(died)` |
| **Code location** | `pipeline.py` lines 312–315: `neg = (y_train == 1).sum(); pos = (y_train == 0).sum(); scale_pos_weight = neg / pos` |
| **Actual value** | `7,001 / 283 ≈ 24.7` |

---

## 5. Model Terms

### 5.1 Logistic Regression

| Item | Detail |
|------|--------|
| **Definition** | A linear classifier that models the **probability** of a binary outcome (survived/died) using the logistic (sigmoid) function: `P(y=1) = 1 / (1 + e^(−Xβ))`. The model learns coefficients **β** for each feature. Despite the name, it is a *classification* algorithm. |
| **Code location** | `pipeline.py` line 303: `LogisticRegression(class_weight="balanced", max_iter=1000, ...)` |
| **Key hyperparameters** | `class_weight="balanced"`, `max_iter=1000` (prevents non-convergence) |
| **Test AUC-ROC** | **0.9975** — highest recall for "Died" class (0.9859) |

---

### 5.2 Random Forest

| Item | Detail |
|------|--------|
| **Definition** | An **ensemble** of decision trees trained on random subsets of features and samples (bagging). Final prediction is the **majority vote** (classification) or average (regression) across all trees. It is robust to overfitting and handles non-linear relationships. |
| **Code location** | `pipeline.py` lines 307–309: `RandomForestClassifier(n_estimators=200, class_weight="balanced", ...)` |
| **Key hyperparameters** | `n_estimators=200` (number of trees), `class_weight="balanced"`, `n_jobs=-1` (parallel training) |
| **Test AUC-ROC** | **0.9919** |
| **Feature importance** | Provides `feature_importances_` attribute (Gini impurity-based) — plotted in `plot4_rf_feature_importance.png` |

---

### 5.3 XGBoost (Extreme Gradient Boosting)

| Item | Detail |
|------|--------|
| **Definition** | A **gradient boosting** algorithm that builds trees **sequentially**, where each tree corrects the errors of the previous one. Uses gradient descent to minimize a loss function. Typically the state-of-the-art for tabular data. |
| **Code location** | `pipeline.py` lines 314–320: `XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, ...)` |
| **Key hyperparameters** | `n_estimators=200`, `learning_rate=0.05` (small steps to prevent overfit), `max_depth=5`, `scale_pos_weight=24.7`, `eval_metric="logloss"` |
| **Test AUC-ROC** | **0.9980** — selected as **best model** |
| **Why selected** | Highest AUC-ROC among all three models |

---

### 5.4 Predict Probability (`predict_proba`)

| Item | Detail |
|------|--------|
| **Definition** | Instead of producing a hard class label (0 or 1), `predict_proba` outputs a **continuous probability score** between 0 and 1 for each class. This is used with a custom threshold (see Section 6) and for computing AUC-ROC/PR-AUC. |
| **Code location** | `pipeline.py` line 327: `y_prob = model.predict_proba(X_test_scaled)[:, 1]` — takes column index 1 = probability of class "Survived" |
| **In app.py** | Line 297: `survival_prob = model.predict_proba(X_input)[0, 1]` |

---

## 6. Classification Threshold Terms

### 6.1 Decision Threshold

| Item | Detail |
|------|--------|
| **Definition** | The **cut-point on the probability score** above which a patient is classified as "Survived" and below which as "Died". The default threshold for binary classifiers is **0.5**, but it can be lowered to increase sensitivity (recall) for the minority class at the cost of more false alarms. |
| **Code location** | `pipeline.py` line 41: `THRESHOLD = 0.35`; applied line 328: `y_pred = (y_prob >= THRESHOLD).astype(int)` |
| **Value used** | **0.35** — deliberately lower than 0.5 to catch more true deaths (improve recall for "Died") |
| **Clinical rationale** | In clinical settings, **missing a death (false negative) is more dangerous** than a false alarm (false positive). Lowering the threshold to 0.35 accepts slightly more false positives to avoid missing real deaths. |
| **In app.py** | The same threshold is applied: `"SURVIVED" if survival_prob >= 0.35 else "HIGH RISK — DIED"` (line 299) |

---

### 6.2 Survival Probability Score

| Item | Detail |
|------|--------|
| **Definition** | The model's raw output: a continuous number from 0 to 1 representing the estimated probability that a patient **survives** within 60 days of COVID-19 diagnosis. |
| **Score < 0.35** | → HIGH RISK: escalate to senior review / ICU |
| **Score 0.35–0.65** | → MODERATE RISK: close monitoring |
| **Score > 0.65** | → LOWER RISK: standard ward care |

---

## 7. Confusion Matrix Terms

The confusion matrix is the fundamental accountability grid for classification models. Rows represent **actual** class; columns represent **predicted** class.

```
                    Predicted: Died   Predicted: Survived
Actual: Died        TN (True Neg)     FP (False Pos)
Actual: Survived    FN (False Neg)    TP (True Pos)
```

> ⚠️ **Note on label encoding:** In this project, `survived = 1` is the "positive" class in sklearn's convention. "Died" = 0 = negative class. This is why the matrix labels may seem reversed from intuition.

---

### 7.1 True Positive (TP)

| Item | Detail |
|------|--------|
| **Definition** | Patient **actually survived** AND model predicted **survived**. A correct positive prediction. |
| **Output (Logistic Regression)** | **TP = 1,741** (out of 1,751 survivors in test set) |
| **Output (Random Forest)** | **TP = 1,751** — perfect recall on survivors |
| **Output (XGBoost)** | **TP = 1,751** — perfect recall on survivors |

---

### 7.2 True Negative (TN)

| Item | Detail |
|------|--------|
| **Definition** | Patient **actually died** AND model predicted **died**. A correct negative prediction. |
| **Output (Logistic Regression)** | **TN = 70** (out of 71 who died in test set) |
| **Output (Random Forest)** | **TN = 68** |
| **Output (XGBoost)** | **TN = 68** |

---

### 7.3 False Positive (FP)

| Item | Detail |
|------|--------|
| **Definition** | Patient **actually died** BUT model predicted **survived**. The model **missed a death** — a **Type I Error**. In clinical context: patient sent home when they should be escalated. **Most dangerous clinical error.** |
| **Output (Logistic Regression)** | **FP = 1** — missed only 1 death out of 71 |
| **Output (Random Forest)** | **FP = 3** — missed 3 deaths |
| **Output (XGBoost)** | **FP = 3** — missed 3 deaths |

> Note: sklearn uses the convention that "positive = survived (1)", so what the output file calls "FP" here is actually a predicted survival for a patient who died. This maps to FP in the survived-is-positive convention.

---

### 7.4 False Negative (FN)

| Item | Detail |
|------|--------|
| **Definition** | Patient **actually survived** BUT model predicted **died**. A **Type II Error** / false alarm. In clinical context: unnecessary escalation or ICU admission. Less dangerous than FP but still costly. |
| **Output (Logistic Regression)** | **FN = 10** |
| **Output (Random Forest)** | **FN = 0** — zero false alarms! |
| **Output (XGBoost)** | **FN = 0** — zero false alarms! |

---

### 7.5 Confusion Matrix (Full Grid)

**Logistic Regression — output.txt:**
```
                    Pred Died   Pred Survived
Actual Died           70            1
Actual Survived       10          1741
TP=1741  FP=1  TN=70  FN=10
```

**Random Forest — output.txt:**
```
                    Pred Died   Pred Survived
Actual Died           68            3
Actual Survived        0           1751
TP=1751  FP=3  TN=68  FN=0
```

**XGBoost — output.txt:**
```
                    Pred Died   Pred Survived
Actual Died           68            3
Actual Survived        0           1751
TP=1751  FP=3  TN=68  FN=0
```

---

## 8. Core Classification Metrics

### 8.1 Accuracy

| Item | Detail |
|------|--------|
| **Definition** | The proportion of **all predictions that are correct**: `(TP + TN) / (TP + TN + FP + FN)`. Simple but **misleading on imbalanced datasets** — a model that predicts "Survived" for everyone gets 96.1% accuracy without learning anything. |
| **Formula** | `Accuracy = (TP + TN) / Total` |
| **Code location** | `pipeline.py` line 347: `accuracy = accuracy_score(y_test, y_pred)` |
| **Output — Logistic Regression** | **0.9940 (99.4%)** — 1,811 of 1,822 patients correctly classified |
| **Output — Random Forest** | **0.9984 (~100%)** |
| **Output — XGBoost** | **0.9984 (~100%)** |

---

### 8.2 Precision

| Item | Detail |
|------|--------|
| **Definition** | Of all patients the model **predicted as Died/Survived**, what fraction **actually are**? Measures how trustworthy positive predictions are. Also called **Positive Predictive Value (PPV)**. |
| **Formula** | `Precision = TP / (TP + FP)` |
| **Code location** | `pipeline.py` line 344: `classification_report(...)`; extracted as `report["Died"]["precision"]` |
| **Output — "Died" class:** | |
| — Logistic Regression | **0.8750** — of patients flagged as "Died", 87.5% truly died |
| — Random Forest | **1.0000** — every "Died" flag was a real death (zero false alarms) |
| — XGBoost | **1.0000** |
| **Output — "Survived" class:** | |
| — Logistic Regression | **0.9994** |
| — Random Forest | **0.9983** |
| — XGBoost | **0.9983** |

---

### 8.3 Recall (Sensitivity / True Positive Rate)

| Item | Detail |
|------|--------|
| **Definition** | Of all patients who **actually Died/Survived**, what fraction did the model successfully detect? Measures the model's ability to find all true cases. Also called **Sensitivity** or **Hit Rate**. |
| **Formula** | `Recall = TP / (TP + FN)` |
| **Code location** | `pipeline.py` line 344: extracted as `report["Died"]["recall"]` |
| **Clinical importance** | **Recall for "Died" is the most critical metric** — missing a death (high FN) is more dangerous than a false alarm. |
| **Output — "Died" class:** | |
| — Logistic Regression | **0.9859** — caught 98.6% of all actual deaths (best of the 3 models!) |
| — Random Forest | **0.9577** — caught 95.8% of deaths |
| — XGBoost | **0.9577** — caught 95.8% of deaths |
| **Output — "Survived" class:** | |
| — Logistic Regression | **0.9943** |
| — Random Forest | **1.0000** |
| — XGBoost | **1.0000** |

---

### 8.4 F1 Score

| Item | Detail |
|------|--------|
| **Definition** | The **harmonic mean of Precision and Recall**. Balances both concerns into one number. Preferred over accuracy on imbalanced data. Ranges from 0 (worst) to 1 (perfect). |
| **Formula** | `F1 = 2 × (Precision × Recall) / (Precision + Recall)` |
| **Code location** | `pipeline.py` line 344: extracted as `report["Died"]["f1-score"]` |
| **Output — "Died" class:** | |
| — Logistic Regression | **0.9272** |
| — Random Forest | **0.9784** |
| — XGBoost | **0.9784** |
| **Output — "Survived" class:** | |
| — Logistic Regression | **0.9969** |
| — Random Forest | **0.9991** |
| — XGBoost | **0.9991** |

---

### 8.5 Support

| Item | Detail |
|------|--------|
| **Definition** | The **number of actual samples** in each class in the test set. Appears in `classification_report` output. |
| **In this project** | Test set: **71 Died** (support for "Died") + **1,751 Survived** (support for "Survived") = **1,822 total** |
| **Code location** | Visible in `run_log.txt` lines 94–101: `Died: support=71, Survived: support=1751, accuracy support=1822` |

---

## 9. Aggregate / Averaging Metrics

### 9.1 Macro Average (F1 Macro Avg)

| Item | Detail |
|------|--------|
| **Definition** | Computes F1 for each class **independently** and then takes the **unweighted average**. Treats all classes equally regardless of how many samples they have. Better for imbalanced datasets where minority class performance matters equally. |
| **Formula** | `F1_macro = (F1_Died + F1_Survived) / 2` |
| **Code location** | `pipeline.py` line 365: `report["macro avg"]["f1-score"]` |
| **Output — Logistic Regression** | **0.9620** → `(0.9272 + 0.9969) / 2 ≈ 0.962` ✓ |
| **Output — Random Forest** | **0.9888** |
| **Output — XGBoost** | **0.9888** |

---

### 9.2 Weighted Average (F1 Weighted Avg)

| Item | Detail |
|------|--------|
| **Definition** | Computes F1 for each class and averages them weighted by **class support** (number of samples). Dominated by the majority class. Tends to be high when the majority class performance is high. |
| **Formula** | `F1_weighted = (F1_Died × 71 + F1_Survived × 1751) / 1822` |
| **Code location** | `pipeline.py` line 366: `report["weighted avg"]["f1-score"]` |
| **Output — Logistic Regression** | **0.9941** |
| **Output — Random Forest** | **0.9983** |
| **Output — XGBoost** | **0.9983** |

---

## 10. Ranking / Discrimination Metrics

### 10.1 AUC-ROC (Area Under the ROC Curve)

| Item | Detail |
|------|--------|
| **Definition** | Measures the model's ability to **rank** patients correctly: the probability that a randomly chosen dead patient receives a **higher risk score** than a randomly chosen survivor. AUC = 1.0 = perfect discrimination; AUC = 0.5 = random guessing. |
| **ROC Curve** | Plots **True Positive Rate (Recall)** on the Y-axis vs. **False Positive Rate** on the X-axis across all possible thresholds. |
| **Formula** | `FPR = FP / (FP + TN)` |
| **Code location** | `pipeline.py` line 345: `auc_roc = roc_auc_score(y_test, y_prob)` |
| **Plot** | `plot1_roc_curves.png` — all three model curves overlaid |
| **Output:** | |
| — Logistic Regression | **0.9975** |
| — Random Forest | **0.9919** |
| — XGBoost | **0.9980** ← Best model selected on this metric |
| **Clinical benchmark** | AUC > 0.85 is considered clinically useful (from `run_log.txt` line 200) |
| **In app.py** | Hardcoded in comparison table: `[0.9975, 0.9919, 0.9980]` (line 442) |

---

### 10.2 PR-AUC (Area Under the Precision-Recall Curve)

| Item | Detail |
|------|--------|
| **Definition** | Area under the **Precision-Recall curve**. Plots **Precision** vs. **Recall** across thresholds. Unlike AUC-ROC, PR-AUC is **not affected by the large number of true negatives** and is therefore the preferred metric for **highly imbalanced datasets** where positive events (deaths) are rare. A random classifier achieves PR-AUC = prevalence (3.9%). |
| **Code location** | `pipeline.py` line 346: `pr_auc = average_precision_score(y_test, y_prob)` |
| **Plot** | `plot2_pr_curves.png` |
| **Output:** | |
| — Logistic Regression | **0.9999** |
| — Random Forest | **0.9993** |
| — XGBoost | **0.9999** |
| **Why so high?** | The Synthea synthetic dataset is artificially clean and well-structured, making separation between classes easier than in messy real-world hospital data. |

---

### 10.3 False Positive Rate (FPR)

| Item | Detail |
|------|--------|
| **Definition** | The fraction of **actual "Died"** patients incorrectly classified as "Survived". The X-axis of the ROC curve. Also called the **fall-out** or **1 − Specificity**. |
| **Formula** | `FPR = FP / (FP + TN)` |
| **Code location** | `pipeline.py` line 484: `fpr, tpr, _ = roc_curve(y_test, res["y_prob"])` |

---

### 10.4 ROC Curve

| Item | Detail |
|------|--------|
| **Definition** | A curve swept by varying the classification threshold from 0 to 1, plotting **(FPR, TPR)** at each threshold. A model with the curve hugging the top-left corner is better. The diagonal line represents a random classifier. |
| **Code location** | `pipeline.py` lines 484–485 in `step10_plots()` |
| **Plot** | `plot1_roc_curves.png` |

---

### 10.5 Precision-Recall Curve

| Item | Detail |
|------|--------|
| **Definition** | Plots **Precision vs. Recall** at every threshold, showing the trade-off between them. As you lower the threshold (catch more positives), recall rises but precision falls. |
| **Code location** | `pipeline.py` lines 500–501: `prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])` |
| **Plot** | `plot2_pr_curves.png` |

---

## 11. Calibration Terms

### 11.1 Calibration Curve (Reliability Diagram)

| Item | Detail |
|------|--------|
| **Definition** | Plots the **mean predicted probability** (X-axis) against the **actual fraction of positives** (Y-axis) in bins of predictions. A **perfectly calibrated** model's curve lies on the diagonal (y = x). Points above the diagonal = model underestimates probability; points below = overestimates. |
| **Code location** | `pipeline.py` lines 397–398: `frac_pos, mean_pred = calibration_curve(y_test, res["y_prob"], n_bins=10)` |
| **Plot** | `cm_calibration.png` — bottom row of the 2×3 grid |
| **App note** | `app.py` line 494–497: *"A well-calibrated model follows the diagonal. The XGBoost and LR models are very well calibrated, meaning predicted probabilities are close to true observed frequencies."* |
| **n_bins=10** | The probability range [0, 1] is divided into 10 equal bins; actual positive rates are measured within each bin. |

---

## 12. Feature Importance & Explainability Terms

### 12.1 SHAP (SHapley Additive exPlanations)

| Item | Detail |
|------|--------|
| **Definition** | A game-theory-based method that assigns each feature a **contribution value (SHAP value)** explaining how much that feature pushed the model's output toward or away from a prediction. SHAP values are **additive**: they sum to the difference between the model's prediction and the baseline (average) prediction. |
| **Positive SHAP** | → pushes prediction toward **survival** (higher probability) |
| **Negative SHAP** | → pushes prediction toward **death** (lower probability) |
| **Code location** | `pipeline.py` `step9_shap()` (lines 422–475); `app.py` lines 363–395 (per-patient waterfall) |
| **Explainer type** | `shap.TreeExplainer` for Random Forest & XGBoost; `shap.LinearExplainer` for Logistic Regression |
| **Best model SHAP** | Applied to **XGBoost** (the best model) |
| **Top 3 SHAP features** | **`was_hospitalized`, `d_dimer`, `age`** (from `clinical_interpretation.txt` line 7 and `run_log.txt` line 150) |

---

### 12.2 SHAP Beeswarm Plot

| Item | Detail |
|------|--------|
| **Definition** | Shows **all test patients** as dots on a horizontal axis of SHAP values. Each row is one feature. Dot color indicates the **feature value** (red = high, blue = low). Spread shows how much that feature varies in its impact. |
| **Code location** | `pipeline.py` lines 436–442: `shap.summary_plot(sv, X_test_df, ...)` |
| **Plot** | `shap_beeswarm.png` |
| **App description** | `app.py` line 513: *"Each dot = one test patient. Color = feature value (red=high, blue=low)."* |

---

### 12.3 SHAP Bar Plot (Mean |SHAP|)

| Item | Detail |
|------|--------|
| **Definition** | Shows the **mean absolute SHAP value** for each feature — the average magnitude of impact across all test patients. This is the global feature importance metric in SHAP. |
| **Formula** | `mean_abs_shap[feature] = mean(|SHAP values for that feature|)` |
| **Code location** | `pipeline.py` lines 445–451: `shap.summary_plot(..., plot_type="bar")` and line 453: `mean_abs_shap = np.abs(sv).mean(axis=0)` |
| **Plot** | `shap_bar.png` |

---

### 12.4 SHAP Dependence Plot

| Item | Detail |
|------|--------|
| **Definition** | A scatter plot showing the relationship between a **feature's value** (X-axis) and its **SHAP value** (Y-axis). Reveals non-linear relationships and thresholds (e.g., at what D-Dimer value does risk sharply increase). |
| **Code location** | `pipeline.py` lines 458–474 in `step9_shap()` — custom implementation using `scatter()` |
| **Generated for** | Top 3 SHAP features: `was_hospitalized`, `d_dimer`, `age` |
| **Plot** | `shap_dependence.png` |
| **Color encoding** | Dot color = feature value (using `coolwarm` colormap) |

---

### 12.5 SHAP Waterfall (Per-Patient)

| Item | Detail |
|------|--------|
| **Definition** | Shows for a **single patient** how each feature contributes to pushing the prediction away from the baseline. Starts from the average model output and adds/subtracts each feature's contribution to reach the final prediction. |
| **Code location** | `app.py` lines 363–393: SHAP values computed per patient form submission, displayed as horizontal bar chart via Plotly |
| **Chart title** | *"SHAP Contribution (green = towards survival, red = towards death)"* |

---

### 12.6 Random Forest Feature Importance (Gini Impurity)

| Item | Detail |
|------|--------|
| **Definition** | For each feature in a Random Forest, measures how much it **reduces impurity** (Gini index) across all splits across all trees. Higher = more important for making predictions. This is a model-native importance vs. SHAP which is model-agnostic. |
| **Code location** | `pipeline.py` line 528: `importances = pd.Series(model_rf.feature_importances_, index=feature_names)` |
| **Plot** | `plot4_rf_feature_importance.png` — top 15 features shown |

---

## 13. Logistic Regression Specific Terms

### 13.1 Coefficient (β)

| Item | Detail |
|------|--------|
| **Definition** | The weight learned by Logistic Regression for each feature. A positive coefficient increases the log-odds of survival; a negative one decreases it. The model is: `log(P/(1−P)) = β₀ + β₁x₁ + β₂x₂ + ...` |
| **Code location** | `pipeline.py` line 375: `model_lr.coef_[0]` |
| **Output — Top coefficients (run_log.txt):** | |
| `has_COPD` | 1.3985 → strong positive association with survival prediction |
| `crp` | 1.2580 |
| `has_asthma` | 0.8430 |
| `has_obesity` | 0.5918 |

> **Caution:** High coefficient for COPD/asthma doesn't mean these conditions improve survival — it may reflect a data artifact in the Synthea synthetic dataset (COPD patients may have had more structured care plans triggering survival labels).

---

### 13.2 Odds Ratio

| Item | Detail |
|------|--------|
| **Definition** | `exp(coefficient)` — the **multiplicative change in odds of survival** for a one-unit increase in the feature (with all other features held constant). Odds Ratio > 1 = increased odds of survival; < 1 = decreased odds. |
| **Formula** | `OR = e^β` |
| **Code location** | `pipeline.py` line 374: `np.exp(model_lr.coef_[0])` |
| **Output — Top 10 (run_log.txt lines 78–87):** | |
| `has_COPD` | OR = **4.05** → patients with COPD have 4× the odds of being classified as survived |
| `crp` | OR = **3.52** |
| `has_asthma` | OR = **2.32** |
| `has_obesity` | OR = **1.81** |
| `low_spo2` | OR = **1.07** |

---

## 14. Clinical / Domain-Specific Terms

### 14.1 SpO2 (Blood Oxygen Saturation)

| Item | Detail |
|------|--------|
| **Definition** | Peripheral oxygen saturation measured by pulse oximetry. Reflects how much oxygen is bound to hemoglobin. Normal: **95–100%**. |
| **LOINC code** | 59408-5 |
| **Clinical threshold** | `SpO2 < 94%` → hypoxia → `low_spo2 = 1` |
| **In code** | `pipeline.py` line 180: `lab_df["low_spo2"] = (lab_df["spo2"] < 94).astype(int)` |
| **Clinical alert (app.py)** | Line 348: `"🔴 Low SpO2 (< 94%) — critical hypoxia"` |

---

### 14.2 D-Dimer

| Item | Detail |
|------|--------|
| **Definition** | A fibrin degradation product. Elevated D-Dimer indicates active blood clot formation and breakdown (**thrombosis**). Normal: < 500 ng/mL. Key COVID-19 severity marker. |
| **LOINC code** | 48065-7 |
| **Clinical threshold** | `D-Dimer > 500 ng/mL` → thrombotic risk → `high_d_dimer = 1` |
| **Top 3 SHAP feature** | **#2 most important** predictor in XGBoost (from `clinical_interpretation.txt`) |
| **In code** | `pipeline.py` line 182: `lab_df["high_d_dimer"] = (lab_df["d_dimer"] > 500).astype(int)` |

---

### 14.3 CRP (C-Reactive Protein)

| Item | Detail |
|------|--------|
| **Definition** | An acute-phase protein released by the liver in response to **systemic inflammation**. Elevated in infections, autoimmune disease, and cytokine storm. Normal: < 3 mg/L. |
| **LOINC code** | 1988-5 |
| **Clinical threshold** | `CRP > 10 mg/L` → active inflammation → `high_crp = 1` |
| **Median in dataset** | 10.2 mg/L (borderline elevated — typical in COVID-19 cohorts) |
| **In code** | `pipeline.py` line 181: `lab_df["high_crp"] = (lab_df["crp"] > 10).astype(int)` |

---

### 14.4 Ferritin

| Item | Detail |
|------|--------|
| **Definition** | An iron-storage protein that rises sharply during severe inflammation. **Hyperferritinemia** (very high ferritin) is a hallmark of **cytokine storm** in severe COVID-19. Normal: 12–300 ng/mL. |
| **LOINC code** | 2276-4 |
| **Median in dataset** | 423.3 ng/mL — above normal upper limit (expected in COVID-19 hospitalized cohort) |

---

### 14.5 WBC (White Blood Cell Count)

| Item | Detail |
|------|--------|
| **Definition** | Total count of white blood cells (immune cells) in blood. Normal: 4.0–11.0 ×10³/µL. Elevated in bacterial infections; can be low or normal in viral infections. |
| **LOINC code** | 6690-2 |
| **Median in dataset** | 3.8 ×10³/µL — slightly below normal (leukopenia, common in COVID-19) |

---

### 14.6 Lymphocytes

| Item | Detail |
|------|--------|
| **Definition** | A type of white blood cell critical for adaptive immune response. **Lymphopenia** (low count) is a consistent finding in severe COVID-19. Normal: 1.0–4.8 ×10³/µL. |
| **LOINC code** | 26474-7 |
| **Clinical threshold** | `Lymphocytes < 1.0` → immune suppression → `low_lymphocytes = 1` |
| **In code** | `pipeline.py` line 183 |

---

### 14.7 IL-6 (Interleukin-6)

| Item | Detail |
|------|--------|
| **Definition** | A pro-inflammatory cytokine. Massively elevated in **cytokine storm**, a life-threatening immune overreaction seen in severe COVID-19. Normal: < 7 pg/mL. |
| **LOINC code** | 33204-9 |
| **Dataset note** | **No IL-6 data found** in the Synthea dataset → imputed with 0 for all patients (WARNING in `run_log.txt` line 30) |

---

### 14.8 Survival Label (Target Variable)

| Item | Detail |
|------|--------|
| **Definition** | The binary outcome we are predicting. `survived = 1` if the patient did **not** die within 60 days of COVID-19 diagnosis; `survived = 0` if they died within 60 days. |
| **Code location** | `pipeline.py` lines 96–100 in `step2_build_target()`: `died_within_60 = DEATHDATE.notna() & (days ≤ 60); df["survived"] = np.where(died_within_60, 0, 1)` |
| **60-day window rationale** | Captures in-hospital mortality + short-term post-discharge mortality attributable to COVID-19. Standard clinical endpoint for acute viral illness outcomes. |
| **Distribution** | 8,752 survived (96.1%) vs. 354 died (3.9%) |

---

### 14.9 Comorbidities

| Item | Detail |
|------|--------|
| **Definition** | Pre-existing medical conditions present **before** COVID-19 diagnosis, which increase the risk of severe illness. |
| **Inclusion window** | Only conditions with `START date ≤ COVID diagnosis date` are included (pre-COVID only) |
| **Code location** | `pipeline.py` line 122: `pre_covid = cond[cond["START"] <= cond["covid_diagnosis_date"]]` |
| **Prevalence in cohort (app.py line 638)** | Obesity: 38.1%, Diabetes: 31.6%, Hypertension: 24.8%, Heart Disease: 6.5%, COPD: 1.5%, Asthma: 1.9% |

---

## 15. Descriptive Statistics Terms

The following terms appear in the `run_log.txt` descriptive statistics table (Step 4).

### 15.1 Count

Number of **non-null observations** for each feature across all 9,106 patients. All counts = 9,106.00 → no missing values after imputation.

---

### 15.2 Mean

The **arithmetic average** of a feature. For binary features, mean = proportion of patients with value = 1.

| Feature | Mean | Interpretation |
|---------|------|---------------|
| age | 41.35 | Average patient age = 41 years |
| gender | 0.476 | 47.6% are Male |
| has_hypertension | 0.248 | 24.8% have hypertension |
| has_diabetes | 0.316 | 31.6% have diabetes |
| crp | 10.27 | Average CRP at/above inflammation threshold |
| was_hospitalized | 0.212 | 21.2% were hospitalized |
| is_icu | 0.041 | 4.1% admitted to ICU |

---

### 15.3 Standard Deviation (Std)

Measures **spread** around the mean. Higher std = more variability across patients.

| Feature | Std | Interpretation |
|---------|-----|---------------|
| age | 23.77 | Wide age range (0–110) |
| ferritin | 112.1 | Large variability in ferritin levels |
| crp | 0.897 | CRP tightly clustered around the median |
| comorbidity_count | 0.938 | Most patients have 0–2 comorbidities |

---

### 15.4 Min / Max

The **smallest and largest observed values**. Used to verify data quality (e.g., age = 0.01 years = very young patient; max age = 110 years).

| Feature | Min | Max |
|---------|-----|-----|
| age | 0.01 years | 110.13 years |
| ferritin | 300.2 ng/mL | 1197.6 ng/mL |
| wbc | 0.7 ×10³ | 10.5 ×10³ |
| crp | 8.0 mg/L | 18.0 mg/L |

---

### 15.5 Percentiles (25th, 50th/Median, 75th)

Divide the data into quarters. The **50th percentile = median** (middle value, robust to outliers).

| Feature | Q1 (25%) | Median (50%) | Q3 (75%) |
|---------|----------|-------------|----------|
| age | 21.9 | 41.0 | 58.8 |
| comorbidity_count | 0 | 1 | 2 |
| ferritin | 423.3 | 423.3 | 423.3 (imputed) |
| d_dimer | 0.3 | 0.3 | 0.3 (imputed) |

---

### 15.6 Imbalance Ratio

| Item | Detail |
|------|--------|
| **Definition** | The ratio of majority class size to minority class size. Quantifies how skewed the class distribution is. |
| **Formula** | `ratio = count(survived) / count(died)` |
| **Code location** | `pipeline.py` line 106: `print(f"  Ratio    : {survived / max(died, 1):.2f}:1")` |
| **Actual output** | **24.72:1** — for every 1 death, there are ~25 survivors (from `run_log.txt` line 20) |

---

## 16. High-Level Model Comparison Summary

This table consolidates all key metrics from `output.txt` into a single view:

| Metric | Logistic Regression | Random Forest | **XGBoost (Best)** |
|--------|---------------------|---------------|--------------------|
| **Accuracy** | 0.9940 | 0.9984 | **0.9984** |
| **AUC-ROC** | 0.9975 | 0.9919 | **0.9980** ★ |
| **PR-AUC** | **0.9999** | 0.9993 | **0.9999** |
| **F1 Macro Avg** | 0.9620 | **0.9888** | **0.9888** |
| **F1 Weighted** | 0.9941 | **0.9983** | **0.9983** |
| **Precision (Died)** | 0.8750 | **1.0000** | **1.0000** |
| **Recall (Died)** | **0.9859** | 0.9577 | 0.9577 |
| **F1 (Died)** | 0.9272 | **0.9784** | **0.9784** |
| **Precision (Survived)** | **0.9994** | 0.9983 | 0.9983 |
| **Recall (Survived)** | 0.9943 | **1.0000** | **1.0000** |
| **F1 (Survived)** | 0.9969 | **0.9991** | **0.9991** |
| **TP** | 1,741 | **1,751** | **1,751** |
| **TN** | **70** | 68 | 68 |
| **FP** | **1** | 3 | 3 |
| **FN** | 10 | **0** | **0** |

> ★ XGBoost was selected as best model based on **highest AUC-ROC (0.9980)**.
>
> 💡 **Key insight:** Logistic Regression achieves better **Recall(Died) = 0.9859** (misses fewest deaths) whereas Random Forest/XGBoost achieve **Precision(Died) = 1.0** (zero false alarms) and **Recall(Survived) = 1.0** (zero survivors missed). The choice depends on clinical priority.

---

## 📝 Quick Reference: The Metric Cheat Sheet

| Goal | Best Metric | Why |
|------|------------|-----|
| Overall correctness | Accuracy | Simple but biased toward majority class |
| Not missing deaths | **Recall (Died)** | Minimizes False Negatives (missed deaths) |
| No false death alarms | **Precision (Died)** | Minimizes False Positives (unnecessary ICU) |
| Balance both above | **F1 (Died)** | Harmonic mean of Precision + Recall |
| Ranking/discrimination | **AUC-ROC** | Threshold-independent ability to rank patients |
| Imbalanced rare events | **PR-AUC** | Better than AUC-ROC when deaths are rare |
| Model trustworthiness | **Calibration Curve** | Are probabilities actually what they say? |
| Explainability | **SHAP values** | Which features drove this prediction? |

---

*Generated from project codebase: `pipeline.py`, `app.py`, `output.txt`, `run_log.txt`, `clinical_interpretation.txt`*  
*Dataset: Synthea 10k COVID-19 synthetic dataset | 9,106 patients | 23 features*  
*Best model: XGBoost | AUC-ROC: 0.9980 | PR-AUC: 0.9999*
