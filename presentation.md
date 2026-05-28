# COVID-19 Patient Survival Prediction
## Using Machine Learning on Synthetic EHR Data (Synthea 10k)
### Graduation Project Presentation (Adapted from `model1_survival/`)

---

## Table of Contents

1. Project Overview & Motivation  
2. Dataset & Target Definition  
3. End-to-End Pipeline (as implemented)  
4. Feature Engineering (23 features)  
5. Preprocessing & Class Imbalance  
6. Model Training (3 models)  
7. Evaluation Results (test set)  
8. Explainability (SHAP)  
9. Streamlit Dashboard  
10. Limitations & Future Work  
11. Conclusion & Deliverables  

---

## 1) Project Overview & Motivation

### Problem Statement
Predict whether a COVID-19 patient will **survive** or **die within 60 days** of diagnosis using EHR-style data.

### Why This Matters
- Early identification of **high-risk patients** can support triage and monitoring decisions.
- ML can combine many weak signals (demographics, comorbidities, labs, encounters) into one risk score.

### What Was Built (in code)
An end-to-end pipeline in `model1_survival/pipeline.py` that:
- extracts a COVID cohort
- builds a 60-day mortality label
- engineers **23 features**
- trains **Logistic Regression**, **Random Forest**, and **XGBoost**
- evaluates on a held-out test set
- generates plots and SHAP explainability outputs
- persists model artifacts for a Streamlit app

---

## 2) Dataset & Target Definition

### Source
Synthea™ 10k COVID-19 synthetic dataset (`10k_synthea_covid19_csv/`).

### Files Used
| File | Purpose |
|------|---------|
| `patients.csv` | demographics + death date |
| `conditions.csv` | COVID diagnosis + comorbidities |
| `observations.csv` | labs (LOINC-coded) |
| `encounters.csv` | visits, hospitalization signals |
| `careplans.csv` | COVID/isolation care plans |

### Cohort Identification (code)
COVID condition rows are detected via:
- `DESCRIPTION` contains “COVID” (case-insensitive), **or**
- SNOMED code `840539006`

### Code (from `model1_survival/pipeline.py`)
```python
def step1_get_covid_patients(conditions_path):
    cond = load(conditions_path)
    cond = parse_dates(cond, ["START", "STOP"])

    is_covid = (
        cond["DESCRIPTION"].str.contains("COVID", case=False, na=False) |
        (cond["CODE"].astype(str) == "840539006")
    )
    covid_cond = cond[is_covid].copy()

    covid_dx = (
        covid_cond.groupby("PATIENT")["START"]
        .min()
        .reset_index()
        .rename(columns={"PATIENT": "patient_id", "START": "covid_diagnosis_date"})
    )
    return covid_dx
```

### Target Label (code)
Let `covid_diagnosis_date` = first COVID condition `START` per patient.

```
survived = 0  if DEATHDATE exists and (DEATHDATE - covid_diagnosis_date) <= 60 days
survived = 1  otherwise
```

### Code (from `model1_survival/pipeline.py`)
```python
def step2_build_target(patients_path, covid_dx):
    pts = load(patients_path)
    pts = parse_dates(pts, ["BIRTHDATE", "DEATHDATE"])
    pts = pts.rename(columns={"Id": "patient_id"})

    df = covid_dx.merge(
        pts[["patient_id", "BIRTHDATE", "DEATHDATE", "GENDER"]],
        on="patient_id", how="left"
    )

    died_within_60 = (
        df["DEATHDATE"].notna() &
        ((df["DEATHDATE"] - df["covid_diagnosis_date"]).dt.days <= 60)
    )
    df["survived"] = np.where(died_within_60, 0, 1)
    return df
```

### Class Distribution (from pipeline run)
| Class | Count | Percentage |
|------|------:|-----------:|
| Survived | 8,752 | 96.1% |
| Died within 60 days | 354 | 3.9% |
| Imbalance ratio | 24.72 : 1 | — |

---

## 3) End-to-End Pipeline (as implemented)

The pipeline executes these steps (see console log in `run_log.txt`):

```
Step 1  → Identify COVID patients (conditions.csv)
Step 2  → Build 60-day survival label (patients.csv + diagnosis date)
Step 3  → Feature engineering (demographics, comorbidities, labs, encounters, careplans)
Step 4  → Build feature matrix (9,106 × 23 + id/label)
Step 5  → Split + scale continuous features + SMOTE on training set
Step 6  → Train 3 models (LR, RF, XGBoost)
Step 7  → Evaluate models on test set (AUC-ROC, PR-AUC, F1, confusion matrices)
Step 8  → SHAP explainability for best model
Step 9  → Save all plots (ROC/PR, confusion, feature importance, labs/age)
Step 10 → Persist model + scaler + reports (output.txt, clinical_interpretation.txt)
```

### Key Design Decisions (from code)
| Decision | Value |
|----------|-------|
| Train/test split | 80% / 20% (stratified) |
| Random seed | 42 |
| Threshold | 0.35 on **survival probability** |
| Outputs folder | `model1_survival/outputs/` |

---

## 4) Feature Engineering (23 features)

The exact feature list (from `pipeline.py` / `app.py`) is:

### A) Demographics (2)
- `age` (years at diagnosis)
- `gender` (Male=1, Female=0)

### B) Comorbidities (7)
Computed from **pre-COVID** conditions (condition `START` <= diagnosis date):
- `has_hypertension`
- `has_diabetes`
- `has_obesity`
- `has_asthma`
- `has_COPD` (matches “copd” or “pulmonary”)
- `has_heart_disease` (matches “heart” or “cardiac”)
- `comorbidity_count` (sum of the 6 flags)

### C) Labs (11)
LOINC → feature mapping:
| LOINC | Feature |
|------:|---------|
| 48065-7 | `d_dimer` |
| 33204-9 | `il6` |
| 2276-4 | `ferritin` |
| 1988-5 | `crp` |
| 59408-5 | `spo2` |
| 6690-2 | `wbc` |
| 26474-7 | `lymphocytes` |

Lab selection rule (code): choose the **closest value before diagnosis** per patient (minimum `days_before`).

Derived clinical flags:
- `low_spo2` (`spo2` < 94)
- `high_crp` (`crp` > 10)
- `high_d_dimer` (`d_dimer` > 500)
- `low_lymphocytes` (`lymphocytes` < 1.0)

Imputation (code): per lab column, fill missing with **median** (if median is NaN, use 0).

### D) Encounters (2)
From encounters in window **0–14 days after diagnosis**:
- `was_hospitalized` (encounter class in `inpatient` or `emergency`)
- `is_icu` (description contains “icu”, “intensive”, or “critical”)

### E) Careplans (1)
- `had_care_plan` (careplan description contains “infectious”, “covid”, or “isolation”)

---

## 5) Preprocessing & Class Imbalance

### Train/Test Split
Test set size (from run): **1,822** patients  
Support in test report: **71 died**, **1,751 survived**

### Scaling
`StandardScaler` is applied to continuous features:
`age, d_dimer, il6, ferritin, crp, spo2, wbc, lymphocytes, comorbidity_count`

### Code (feature list + scaling + SMOTE, from `model1_survival/pipeline.py`)
```python
continuous_cols = [
    "age", "d_dimer", "il6", "ferritin", "crp",
    "spo2", "wbc", "lymphocytes", "comorbidity_count"
]

scaler   = StandardScaler()
cont_idx = [list(X.columns).index(c) for c in continuous_cols]

X_train_scaled = X_train.values.copy().astype(float)
X_test_scaled  = X_test.values.copy().astype(float)
X_train_scaled[:, cont_idx] = scaler.fit_transform(X_train.iloc[:, cont_idx].values)
X_test_scaled[:, cont_idx]  = scaler.transform(X_test.iloc[:, cont_idx].values)

sm = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
```

### SMOTE
Applied to **training set only** (after scaling). From run log:
| Class | Before | After |
|------|-------:|------:|
| Survived | 7,001 | 7,001 |
| Died | 283 | 7,001 |

### Class Weighting
- Logistic Regression / Random Forest: `class_weight="balanced"`
- XGBoost: `scale_pos_weight = (#survived in train) / (#died in train)`

---

## 6) Model Training (3 models)

| Model | Key params (from code) |
|------|-------------------------|
| Logistic Regression | `max_iter=1000`, `class_weight="balanced"` |
| Random Forest | `n_estimators=200`, `class_weight="balanced"` |
| XGBoost | `n_estimators=200`, `learning_rate=0.05`, `max_depth=5`, `scale_pos_weight` computed |

---

## 7) Evaluation Results (held-out test set)

### Threshold Logic (important)
The models predict **survival probability** \(P(\text{survived}=1)\).

```
If survival_prob >= 0.35  → Predict “Survived”
If survival_prob <  0.35  → Flag “High risk (Died)”
```

### Code (thresholding, from `model1_survival/pipeline.py`)
```python
def evaluate_model(model, model_name, X_test_scaled, y_test):
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # P(survived=1)
    y_pred = (y_prob >= THRESHOLD).astype(int)         # 1=Survived, 0=Died
    # ... metrics ...
```

### Model Comparison (from `model1_survival/outputs/output.txt`)
| Metric | Logistic Regression | Random Forest | XGBoost (Best by AUC-ROC) |
|--------|--------------------:|--------------:|---------------------------:|
| AUC-ROC | 0.9975 | 0.9919 | **0.9980** |
| PR-AUC | 0.9999 | 0.9993 | **0.9999** |
| Accuracy | 0.9940 | 0.9984 | **0.9984** |
| Precision (Died) | 0.8750 | **1.0000** | **1.0000** |
| Recall (Died) | **0.9859** | 0.9577 | 0.9577 |
| F1 (Died) | 0.9272 | **0.9784** | **0.9784** |
| F1 (Macro) | 0.9620 | 0.9888 | **0.9888** |

### Confusion Matrices (with “Died” as the positive clinical event)

Logistic Regression:
```
                  Pred Died   Pred Survived
Actual Died          70            1
Actual Survived      10          1741
```

Random Forest / XGBoost:
```
                  Pred Died   Pred Survived
Actual Died          68            3
Actual Survived       0          1751
```

### Generated Visualizations (saved under `model1_survival/outputs/`)
| File | Purpose |
|------|---------|
| `plot1_roc_curves.png` | ROC curves |
| `plot2_pr_curves.png` | Precision-Recall curves |
| `plot3_confusion_matrices.png` | confusion matrices |
| `cm_calibration.png` | calibration curves |
| `plot4_rf_feature_importance.png` | RF top-15 feature importance |
| `plot5_age_distribution.png` | age histogram by outcome |
| `plot6_lab_boxplots.png` | lab boxplots by outcome |

---

## 8) Explainability (SHAP)

### What SHAP Does
SHAP attributes per-feature contributions to the model’s output, helping explain **why** a patient is scored as low/high survival probability.

### Best Model Explainability
The pipeline runs SHAP on the selected best model (by AUC-ROC). From the run log, the top 3 SHAP features were:
1. `was_hospitalized`
2. `d_dimer`
3. `age`

### SHAP Figures Saved
- `shap_beeswarm.png`
- `shap_bar.png`
- `shap_dependence.png`

---

## 9) Streamlit Dashboard (`model1_survival/app.py`)

### Pages
1. **Patient Risk Calculator**: enter patient info → survival probability + risk band + alerts + SHAP explanation.
2. **Model Performance**: view ROC/PR/confusion/calibration plots.
3. **SHAP Explainability**: global SHAP plots + key takeaways.
4. **Dataset Insights**: cohort stats, imbalance visualization, distributions.

### Risk Stratification (as implemented)
| Survival probability | Risk level |
|---------------------:|------------|
| < 0.35 | High risk |
| 0.35 – 0.65 | Moderate |
| ≥ 0.65 | Lower risk |

### Code (Streamlit prediction path, from `model1_survival/app.py`)
```python
FEATURE_NAMES = [
    "age", "gender",
    "has_hypertension", "has_diabetes", "has_obesity",
    "has_asthma", "has_COPD", "has_heart_disease", "comorbidity_count",
    "d_dimer", "il6", "ferritin", "crp", "spo2", "wbc", "lymphocytes",
    "low_spo2", "high_crp", "high_d_dimer", "low_lymphocytes",
    "was_hospitalized", "is_icu", "had_care_plan",
]

CONTINUOUS = [
    "age", "d_dimer", "il6", "ferritin",
    "crp", "spo2", "wbc", "lymphocytes", "comorbidity_count"
]
CONT_IDX = [FEATURE_NAMES.index(c) for c in CONTINUOUS]

def build_feature_vector(inputs: dict) -> np.ndarray:
    vec = np.array([[inputs[f] for f in FEATURE_NAMES]], dtype=float)
    vec[:, CONT_IDX] = scaler.transform(vec[:, CONT_IDX])
    return vec

X_input = build_feature_vector(inputs)
survival_prob = model.predict_proba(X_input)[0, 1]
prediction = "SURVIVED" if survival_prob >= 0.35 else "HIGH RISK — DIED"
```

---

## 10) Limitations & Future Work

### Limitations (scope + data)
- Synthetic dataset (Synthea): realistic structure, but not guaranteed to match real hospital distributions.
- Some labs may be sparse; median imputation can create near-constant columns if a lab is missing for most patients.
- No external validation dataset; results are only on held-out synthetic test data.

### Future Work
- Validate on real-world EHR data (with governance + ethics approvals).
- Add temporal features (lab trends over time) rather than one closest pre-diagnosis measurement.
- Evaluate fairness and subgroup performance (age, sex, etc.).
- Deploy as an API for integration (and log monitoring + drift detection).

---

## 11) Conclusion & Deliverables

### Key Outcome
On the held-out test set, **XGBoost** achieved **AUC-ROC = 0.9980** and **PR-AUC = 0.9999**.

### What’s Produced by the Pipeline
| Artifact | Location |
|----------|----------|
| Saved model | `model1_survival/survival_model.pkl` |
| Saved scaler | `model1_survival/survival_scaler.pkl` |
| Metrics report | `model1_survival/outputs/output.txt` |
| Clinical summary | `model1_survival/outputs/clinical_interpretation.txt` |
| Plots + SHAP images | `model1_survival/outputs/` |

### How to Run
Pipeline:
```
python run_pipeline.py
```

Dashboard:
```
streamlit run model1_survival/app.py
```
