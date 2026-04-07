import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE

DATA_DIR  = r"f:\Graduation-Project\10k_synthea_covid19_csv"
OUT_DIR   = r"f:\Graduation-Project\model1_survival\outputs"
MODEL_DIR = r"f:\Graduation-Project\model1_survival"
RANDOM_STATE = 42
THRESHOLD    = 0.35

os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
COLORS = {"survived": "#4CAF81", "died": "#EF5350"}


def load(path):
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def parse_dates(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def step1_get_covid_patients(conditions_path):
    cond = load(conditions_path)
    cond = parse_dates(cond, ["START", "STOP"])

    is_covid = (
        cond["DESCRIPTION"].str.contains("COVID", case=False, na=False) |
        (cond["CODE"].astype(str) == "840539006")
    )
    covid_cond = cond[is_covid].copy()
    print(f"  COVID condition rows : {len(covid_cond):,}")

    covid_dx = (
        covid_cond.groupby("PATIENT")["START"]
        .min()
        .reset_index()
        .rename(columns={"PATIENT": "patient_id", "START": "covid_diagnosis_date"})
    )
    print(f"  Unique COVID patients: {len(covid_dx):,}")
    return covid_dx


def step2_build_target(patients_path, covid_dx):
    print("\n" + "="*60)
    print("STEP 2 - Build survival label")
    print("="*60)

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

    survived = df["survived"].sum()
    died     = (df["survived"] == 0).sum()
    print(f"  Survived : {survived:,} patients")
    print(f"  Died     : {died:,} patients")
    print(f"  Ratio    : {survived / max(died, 1):.2f}:1")
    return df


def feat_demographics(base_df):
    feat = base_df[["patient_id", "covid_diagnosis_date", "BIRTHDATE", "GENDER"]].copy()
    feat["age"]    = (feat["covid_diagnosis_date"] - feat["BIRTHDATE"]).dt.days / 365.25
    feat["gender"] = (feat["GENDER"] == "M").astype(int)
    return feat[["patient_id", "age", "gender"]]


def feat_comorbidities(conditions_path, base_df):
    cond = load(conditions_path)
    cond = parse_dates(cond, ["START"])
    cond = cond.rename(columns={"PATIENT": "patient_id"})
    cond = cond.merge(base_df[["patient_id", "covid_diagnosis_date"]], on="patient_id", how="inner")
    pre_covid = cond[cond["START"] <= cond["covid_diagnosis_date"]].copy()

    desc = pre_covid["DESCRIPTION"].str.lower().fillna("")
    defs = {
        "has_hypertension" : desc.str.contains("hypertension"),
        "has_diabetes"     : desc.str.contains("diabetes"),
        "has_obesity"      : desc.str.contains("obesity"),
        "has_asthma"       : desc.str.contains("asthma"),
        "has_COPD"         : desc.str.contains("copd|pulmonary"),
        "has_heart_disease": desc.str.contains("heart|cardiac"),
    }
    for col, mask in defs.items():
        pre_covid[col] = mask.astype(int)

    cols = list(defs.keys())
    agg = pre_covid.groupby("patient_id")[cols].max().reset_index()
    agg["comorbidity_count"] = agg[cols].sum(axis=1)
    return agg


def feat_labs(observations_path, base_df):
    LOINC_MAP = {
        "48065-7": "d_dimer",
        "33204-9": "il6",
        "2276-4" : "ferritin",
        "1988-5" : "crp",
        "59408-5": "spo2",
        "6690-2" : "wbc",
        "26474-7": "lymphocytes",
    }

    print("  Loading observations (large file)...")
    obs = load(observations_path)
    obs = parse_dates(obs, ["DATE"])
    obs = obs.rename(columns={"PATIENT": "patient_id"})
    obs = obs[obs["CODE"].astype(str).isin(LOINC_MAP)].copy()
    obs["lab_name"] = obs["CODE"].astype(str).map(LOINC_MAP)
    obs["VALUE"]    = pd.to_numeric(obs["VALUE"], errors="coerce")

    obs = obs.merge(base_df[["patient_id", "covid_diagnosis_date"]], on="patient_id", how="inner")
    obs = obs[obs["DATE"] <= obs["covid_diagnosis_date"]].copy()
    obs["days_before"] = (obs["covid_diagnosis_date"] - obs["DATE"]).dt.days
    obs = obs.sort_values("days_before")
    closest = obs.groupby(["patient_id", "lab_name"]).first().reset_index()

    lab_wide = closest.pivot(index="patient_id", columns="lab_name", values="VALUE").reset_index()
    lab_wide.columns.name = None
    for lab in LOINC_MAP.values():
        if lab not in lab_wide.columns:
            lab_wide[lab] = np.nan

    lab_df = base_df[["patient_id"]].merge(lab_wide, on="patient_id", how="left")
    for lab in LOINC_MAP.values():
        median_val = lab_df[lab].median()
        if pd.isna(median_val):
            median_val = 0
        lab_df[lab] = lab_df[lab].fillna(median_val)

    lab_df["low_spo2"]        = (lab_df["spo2"] < 94).astype(int)
    lab_df["high_crp"]        = (lab_df["crp"] > 10).astype(int)
    lab_df["high_d_dimer"]    = (lab_df["d_dimer"] > 500).astype(int)
    lab_df["low_lymphocytes"] = (lab_df["lymphocytes"] < 1.0).astype(int)
    return lab_df


def feat_encounters(encounters_path, base_df):
    enc = load(encounters_path)
    enc = parse_dates(enc, ["START"])
    enc = enc.rename(columns={"PATIENT": "patient_id"})
    enc = enc.merge(base_df[["patient_id", "covid_diagnosis_date"]], on="patient_id", how="inner")

    if enc["START"].dt.tz is not None:
        enc["START"] = enc["START"].dt.tz_localize(None)

    enc["days_from_dx"] = (enc["START"] - enc["covid_diagnosis_date"]).dt.days
    enc_window = enc[(enc["days_from_dx"] >= 0) & (enc["days_from_dx"] <= 14)].copy()

    desc   = enc_window["DESCRIPTION"].str.lower().fillna("")
    eclass = enc_window["ENCOUNTERCLASS"].str.lower().fillna("")

    enc_window["was_hospitalized"] = eclass.isin(["inpatient", "emergency"]).astype(int)
    enc_window["is_icu"]           = desc.str.contains("icu|intensive|critical").astype(int)

    enc_agg = enc_window.groupby("patient_id")[["was_hospitalized", "is_icu"]].max().reset_index()
    enc_df  = base_df[["patient_id"]].merge(enc_agg, on="patient_id", how="left")
    enc_df["was_hospitalized"] = enc_df["was_hospitalized"].fillna(0).astype(int)
    enc_df["is_icu"]           = enc_df["is_icu"].fillna(0).astype(int)
    return enc_df


def feat_careplans(careplans_path, base_df):
    cp = load(careplans_path)
    cp = cp.rename(columns={"PATIENT": "patient_id"})
    desc = cp["DESCRIPTION"].str.lower().fillna("")
    cp["had_care_plan"] = desc.str.contains("infectious|covid|isolation").astype(int)
    cp_agg = cp.groupby("patient_id")["had_care_plan"].max().reset_index()
    cp_df  = base_df[["patient_id"]].merge(cp_agg, on="patient_id", how="left")
    cp_df["had_care_plan"] = cp_df["had_care_plan"].fillna(0).astype(int)
    return cp_df


def step4_build_feature_matrix(base_df, demo_df, comorbidity_df, labs_df, enc_df, cp_df):
    final = base_df[["patient_id", "survived"]].copy()
    for feat_df in [demo_df, comorbidity_df, labs_df, enc_df, cp_df]:
        final = final.merge(feat_df, on="patient_id", how="left")

    feature_cols = [
        "age", "gender",
        "has_hypertension", "has_diabetes", "has_obesity",
        "has_asthma", "has_COPD", "has_heart_disease", "comorbidity_count",
        "d_dimer", "il6", "ferritin", "crp", "spo2", "wbc", "lymphocytes",
        "low_spo2", "high_crp", "high_d_dimer", "low_lymphocytes",
        "was_hospitalized", "is_icu", "had_care_plan",
    ]
    final = final[["patient_id"] + feature_cols + ["survived"]]
    print(f"  Feature matrix shape: {final.shape}")
    print(f"  Survived: {(final['survived']==1).sum()} | Died: {(final['survived']==0).sum()}")
    return final


def step5_preprocess(final_df):
    continuous_cols = [
        "age", "d_dimer", "il6", "ferritin", "crp",
        "spo2", "wbc", "lymphocytes", "comorbidity_count"
    ]

    X = final_df.drop(columns=["patient_id", "survived"])
    y = final_df["survived"]

    class_counts    = y.value_counts()
    min_class_count = class_counts.min()
    print(f"  Class distribution: {class_counts.to_dict()}")

    if min_class_count < 2:
        print(f"  WARNING: Only {min_class_count} minority sample(s). Forcing into training set.")
        minority_idx = y[y == class_counts.idxmin()].index
        majority_df  = final_df[~final_df.index.isin(minority_idx)]
        X_maj = majority_df.drop(columns=["patient_id", "survived"])
        y_maj = majority_df["survived"]
        X_min = X.loc[minority_idx]
        y_min = y.loc[minority_idx]
        X_maj_tr, X_maj_te, y_maj_tr, y_maj_te = train_test_split(
            X_maj, y_maj, test_size=0.20, random_state=RANDOM_STATE
        )
        X_train = pd.concat([X_maj_tr, X_min])
        y_train = pd.concat([y_maj_tr, y_min])
        X_test  = X_maj_te
        y_test  = y_maj_te
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
        )

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    scaler   = StandardScaler()
    cont_idx = [list(X.columns).index(c) for c in continuous_cols]

    X_train_scaled = X_train.values.copy().astype(float)
    X_test_scaled  = X_test.values.copy().astype(float)
    X_train_scaled[:, cont_idx] = scaler.fit_transform(X_train.iloc[:, cont_idx].values)
    X_test_scaled[:, cont_idx]  = scaler.transform(X_test.iloc[:, cont_idx].values)

    train_min = Counter(y_train).most_common()[-1][1]
    if train_min >= 2:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
        dist = Counter(y_train_res)
        print(f"  After SMOTE - Survived: {dist[1]} | Died: {dist[0]}")
    else:
        print(f"  WARNING: Only {train_min} minority sample(s) in train set. Skipping SMOTE.")
        X_train_res, y_train_res = X_train_scaled, y_train

    return X_train_res, y_train_res, X_train_scaled, X_test_scaled, y_train, y_test, scaler, X


def step6_train_models(X_train_res, y_train_res, y_train):
    if len(np.unique(y_train_res)) < 2:
        raise ValueError("Training set has only 1 class. Dataset too small or imbalanced.")

    print("  Training Logistic Regression...")
    model_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    model_lr.fit(X_train_res, y_train_res)

    print("  Training Random Forest...")
    model_rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      random_state=RANDOM_STATE, n_jobs=-1)
    model_rf.fit(X_train_res, y_train_res)

    print("  Training XGBoost...")
    neg = (y_train == 1).sum()
    pos = max((y_train == 0).sum(), 1)
    model_xgb = XGBClassifier(
        n_estimators=200, scale_pos_weight=neg / pos,
        learning_rate=0.05, max_depth=5,
        random_state=RANDOM_STATE, eval_metric="logloss",
        use_label_encoder=False, verbosity=0,
    )
    model_xgb.fit(X_train_res, y_train_res)

    print("  All models trained.")
    return model_lr, model_rf, model_xgb


def evaluate_model(model, model_name, X_test_scaled, y_test):
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    print(f"\n--- {model_name} (threshold={THRESHOLD}) ---")

    if len(np.unique(y_test)) < 2:
        print(f"  WARNING: Test set has only 1 class. AUC metrics unavailable.")
        acc = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {acc:.4f}")
        return {
            "name": model_name, "y_prob": y_prob, "y_pred": y_pred,
            "auc_roc": np.nan, "pr_auc": np.nan, "accuracy": acc,
            "precision_died": np.nan, "recall_died": np.nan, "f1_died": np.nan,
            "precision_survived": np.nan, "recall_survived": np.nan, "f1_survived": np.nan,
            "f1_macro": np.nan, "f1_weighted": np.nan, "report": {},
        }

    report   = classification_report(y_test, y_pred, target_names=["Died", "Survived"], output_dict=True)
    auc_roc  = roc_auc_score(y_test, y_prob)
    pr_auc   = average_precision_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)

    print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))
    print(f"  Accuracy: {accuracy:.4f} | AUC-ROC: {auc_roc:.4f} | PR-AUC: {pr_auc:.4f}")

    return {
        "name"              : model_name,
        "y_prob"            : y_prob,
        "y_pred"            : y_pred,
        "auc_roc"           : auc_roc,
        "pr_auc"            : pr_auc,
        "accuracy"          : accuracy,
        "precision_died"    : report["Died"]["precision"],
        "recall_died"       : report["Died"]["recall"],
        "f1_died"           : report["Died"]["f1-score"],
        "precision_survived": report["Survived"]["precision"],
        "recall_survived"   : report["Survived"]["recall"],
        "f1_survived"       : report["Survived"]["f1-score"],
        "f1_macro"          : report["macro avg"]["f1-score"],
        "f1_weighted"       : report["weighted avg"]["f1-score"],
        "report"            : report,
    }


def print_odds_ratios(model_lr, feature_names):
    odds_df = pd.DataFrame({
        "Feature"    : feature_names,
        "Odds Ratio" : np.exp(model_lr.coef_[0]),
        "Coefficient": model_lr.coef_[0],
    }).sort_values("Odds Ratio", ascending=False)
    print("\n  Top 10 features by Odds Ratio (Logistic Regression):")
    print(odds_df.head(10).to_string(index=False))
    return odds_df


def plot_per_model(results_list, y_test):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Confusion Matrices & Calibration Curves", fontsize=16, fontweight="bold")

    for col, res in enumerate(results_list):
        cm = confusion_matrix(y_test, res["y_pred"], labels=[0, 1])
        ax_cm = axes[0, col]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
        ax_cm.set_title(f'{res["name"]}\nConfusion Matrix', fontsize=12, fontweight="bold")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")

        ax_cal = axes[1, col]
        if len(np.unique(y_test)) >= 2:
            frac_pos, mean_pred = calibration_curve(y_test, res["y_prob"], n_bins=10)
            ax_cal.plot(mean_pred, frac_pos, "s-", label=res["name"], color="#5C6BC0")
        ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1.5)
        ax_cal.set_title(f'{res["name"]}\nCalibration Curve', fontsize=12, fontweight="bold")
        ax_cal.set_xlabel("Mean Predicted Probability")
        ax_cal.set_ylabel("Fraction of Positives")
        ax_cal.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cm_calibration.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_comparison_table(results_list):
    def fmt(v):
        return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A  "
    header = f"{'Model':<25} {'AUC-ROC':>8} {'PR-AUC':>8} {'Recall(Died)':>13} {'F1(Died)':>10}"
    print(header)
    for r in results_list:
        print(f"{r['name']:<25} {fmt(r['auc_roc']):>8} {fmt(r['pr_auc']):>8}"
              f" {fmt(r['recall_died']):>13} {fmt(r['f1_died']):>10}")


def step9_shap(best_model, best_name, X_train_res, X_test_scaled, feature_names):
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

    if "Logistic" in best_name:
        X_train_df  = pd.DataFrame(X_train_res, columns=feature_names)
        explainer   = shap.LinearExplainer(best_model, X_train_df)
        shap_values = explainer.shap_values(X_test_df)
    else:
        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_df)

    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_test_df, feature_names=list(feature_names), show=False)
    plt.title(f"SHAP Beeswarm - {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "shap_beeswarm.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_test_df, feature_names=list(feature_names), plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "shap_bar.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")

    mean_abs_shap = np.abs(sv).mean(axis=0)
    top3_idx      = np.argsort(mean_abs_shap)[::-1][:3]
    top3_features = [feature_names[i] for i in top3_idx]
    print(f"  Top 3 SHAP features: {top3_features}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, feat in enumerate(top3_features):
        feat_idx = list(feature_names).index(feat)
        x_vals   = X_test_df[feat].values
        s_vals   = sv[:, feat_idx]
        sc = axes[i].scatter(x_vals, s_vals, c=x_vals, cmap="coolwarm", alpha=0.6, edgecolors="none", s=20)
        axes[i].axhline(0, color="black", linewidth=1, linestyle="--")
        axes[i].set_xlabel(feat, fontsize=11)
        axes[i].set_ylabel("SHAP value", fontsize=11)
        axes[i].set_title(f"Dependence: {feat}", fontsize=12, fontweight="bold")
        plt.colorbar(sc, ax=axes[i], label=feat)
    plt.suptitle(f"SHAP Dependence Plots - {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "shap_dependence.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")
    return sv, top3_features


def step10_plots(results_list, y_test, final_df, feature_names, model_rf):
    palette = ["#5C6BC0", "#26A69A", "#EF5350"]

    if len(np.unique(y_test)) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for res, color in zip(results_list, palette):
            fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{res['name']} (AUC={res['auc_roc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves - All Models", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "plot1_roc_curves.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {p}")

        fig, ax = plt.subplots(figsize=(8, 6))
        for res, color in zip(results_list, palette):
            prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
            ax.plot(rec, prec, color=color, lw=2, label=f"{res['name']} (PR-AUC={res['pr_auc']:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves - All Models", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(OUT_DIR, "plot2_pr_curves.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {p}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, res, color in zip(axes, results_list, palette):
        cm = confusion_matrix(y_test, res["y_pred"], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
        ax.set_title(res["name"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(f"Confusion Matrices (threshold={THRESHOLD})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "plot3_confusion_matrices.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")

    importances = pd.Series(model_rf.feature_importances_, index=feature_names)
    top15 = importances.sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top15.index, top15.values, color="#5C6BC0", edgecolor="white")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 15 Feature Importances - Random Forest", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "plot4_rf_feature_importance.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")

    age_survived = final_df[final_df["survived"] == 1]["age"]
    age_died     = final_df[final_df["survived"] == 0]["age"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(age_survived, bins=30, alpha=0.6, color=COLORS["survived"],
            label=f"Survived (n={len(age_survived):,})", edgecolor="white")
    ax.hist(age_died, bins=30, alpha=0.6, color=COLORS["died"],
            label=f"Died (n={len(age_died):,})", edgecolor="white")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.set_title("Age Distribution by Outcome", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "plot5_age_distribution.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")

    lab_thresholds = {"d_dimer": 500, "crp": 10, "spo2": 94, "ferritin": 300}
    outcome_label  = final_df["survived"].map({0: "Died", 1: "Survived"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (lab, thresh) in enumerate(lab_thresholds.items()):
        ax = axes.flatten()[i]
        plot_df = pd.DataFrame({"Value": final_df[lab], "Outcome": outcome_label})
        sns.boxplot(x="Outcome", y="Value", data=plot_df, ax=ax,
                    palette={"Died": COLORS["died"], "Survived": COLORS["survived"]},
                    width=0.5, fliersize=3)
        ax.axhline(thresh, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({thresh})")
        ax.set_title(f"{lab.upper()} by Outcome", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
    fig.suptitle("Lab Values by Outcome", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "plot6_lab_boxplots.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def fmt(v):
    return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"


def save_output_txt(results_list, y_test, best_name):
    path = os.path.join(OUT_DIR, "output.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  COVID-19 SURVIVAL MODEL - EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  Models evaluated : {', '.join(r['name'] for r in results_list)}\n")
        f.write(f"  Best model       : {best_name}\n")
        f.write(f"  Decision threshold: {THRESHOLD}  (prob >= {THRESHOLD} -> Survived)\n\n")

        for res in results_list:
            cm = confusion_matrix(y_test, res["y_pred"], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            f.write("-" * 70 + "\n")
            f.write(f"  MODEL: {res['name']}\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Accuracy          : {fmt(res['accuracy'])}\n")
            f.write(f"  AUC-ROC           : {fmt(res['auc_roc'])}\n")
            f.write(f"  PR-AUC            : {fmt(res['pr_auc'])}\n")
            f.write(f"  F1 Macro Avg      : {fmt(res['f1_macro'])}\n")
            f.write(f"  F1 Weighted Avg   : {fmt(res['f1_weighted'])}\n\n")
            f.write(f"  --- Class: DIED (0) ---\n")
            f.write(f"  Precision : {fmt(res['precision_died'])}\n")
            f.write(f"  Recall    : {fmt(res['recall_died'])}\n")
            f.write(f"  F1 Score  : {fmt(res['f1_died'])}\n\n")
            f.write(f"  --- Class: SURVIVED (1) ---\n")
            f.write(f"  Precision : {fmt(res['precision_survived'])}\n")
            f.write(f"  Recall    : {fmt(res['recall_survived'])}\n")
            f.write(f"  F1 Score  : {fmt(res['f1_survived'])}\n\n")
            f.write(f"  --- Confusion Matrix ---\n")
            f.write(f"  {'':18} Pred Died  Pred Survived\n")
            f.write(f"  {'Actual Died':18} {tn:^10d} {fp:^13d}\n")
            f.write(f"  {'Actual Survived':18} {fn:^10d} {tp:^13d}\n")
            f.write(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}\n\n")

        f.write("=" * 70 + "\n")
        f.write("  COMPARISON TABLE\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Model':<25} {'Acc':>8} {'AUC-ROC':>9} {'PR-AUC':>8} {'F1(Died)':>10} {'Recall(Died)':>13}\n")
        f.write("-" * 70 + "\n")
        for res in results_list:
            marker = " *" if res["name"] == best_name else ""
            f.write(
                f"{res['name']:<25} {fmt(res['accuracy']):>8} {fmt(res['auc_roc']):>9} "
                f"{fmt(res['pr_auc']):>8} {fmt(res['f1_died']):>10} {fmt(res['recall_died']):>13}{marker}\n"
            )
        f.write("\n  * = Best model (highest AUC-ROC)\n")

    print(f"  Metrics saved -> {path}")


def step11_save_model(best_model, scaler):
    model_path  = os.path.join(MODEL_DIR, "survival_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "survival_scaler.pkl")
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Model  saved -> {model_path}")
    print(f"  Scaler saved -> {scaler_path}")


def step12_clinical_interpretation(top3_features, results_list, final_df):
    best = max(results_list, key=lambda r: r["auc_roc"] if not np.isnan(r["auc_roc"]) else r["accuracy"])
    text = f"""
Best model: {best['name']}
AUC-ROC   : {fmt(best['auc_roc'])}
PR-AUC    : {fmt(best['pr_auc'])}
Recall (Died): {fmt(best['recall_died'])}

Top 3 predictive features: {top3_features[0]}, {top3_features[1]}, {top3_features[2]}

Critical thresholds:
  SpO2 < 94%      -> High mortality risk
  D-Dimer > 500   -> Thrombotic complications
  CRP > 10 mg/L   -> Systemic inflammation

Highest-risk profile:
  Age > 65, low SpO2, high D-Dimer, high CRP, low lymphocytes, ICU, diabetes/heart disease.

Usage:
  Score < {THRESHOLD} -> HIGH RISK (escalate care)
  Score > 0.65  -> LOWER RISK (standard care)
  Re-score every 24-48 hours.

WARNING: Trained on SYNTHETIC data. Validate before clinical use.
"""
    print(text)
    report_path = os.path.join(OUT_DIR, "clinical_interpretation.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Clinical report saved -> {report_path}")


def main():
    patients_path     = os.path.join(DATA_DIR, "patients.csv")
    conditions_path   = os.path.join(DATA_DIR, "conditions.csv")
    observations_path = os.path.join(DATA_DIR, "observations.csv")
    encounters_path   = os.path.join(DATA_DIR, "encounters.csv")
    careplans_path    = os.path.join(DATA_DIR, "careplans.csv")

    print("\n" + "="*60)
    print("STEP 1 - Get COVID patients")
    print("="*60)
    covid_dx = step1_get_covid_patients(conditions_path)

    base_df = step2_build_target(patients_path, covid_dx)

    print("\n" + "="*60)
    print("STEP 3 - Feature engineering")
    print("="*60)
    demo_df        = feat_demographics(base_df)
    comorbidity_df = feat_comorbidities(conditions_path, base_df)
    labs_df        = feat_labs(observations_path, base_df)
    enc_df         = feat_encounters(encounters_path, base_df)
    cp_df          = feat_careplans(careplans_path, base_df)

    print("\n" + "="*60)
    print("STEP 4 - Build feature matrix")
    print("="*60)
    final_df = step4_build_feature_matrix(base_df, demo_df, comorbidity_df, labs_df, enc_df, cp_df)

    print("\n" + "="*60)
    print("STEP 5 - Preprocess and split")
    print("="*60)
    (X_train_res, y_train_res,
     X_train_scaled, X_test_scaled,
     y_train, y_test, scaler, X) = step5_preprocess(final_df)
    feature_names = list(X.columns)

    print("\n" + "="*60)
    print("STEP 6 - Train models")
    print("="*60)
    model_lr, model_rf, model_xgb = step6_train_models(X_train_res, y_train_res, y_train)
    print_odds_ratios(model_lr, feature_names)

    print("\n" + "="*60)
    print("STEP 7 - Evaluate models")
    print("="*60)
    res_lr  = evaluate_model(model_lr,  "Logistic Regression", X_test_scaled, y_test)
    res_rf  = evaluate_model(model_rf,  "Random Forest",       X_test_scaled, y_test)
    res_xgb = evaluate_model(model_xgb, "XGBoost",             X_test_scaled, y_test)
    results_list = [res_lr, res_rf, res_xgb]

    plot_per_model(results_list, y_test)
    print_comparison_table(results_list)

    best_result = max(results_list, key=lambda r: r["auc_roc"] if not np.isnan(r["auc_roc"]) else r["accuracy"])
    best_name   = best_result["name"]
    best_model  = {"Logistic Regression": model_lr, "Random Forest": model_rf, "XGBoost": model_xgb}[best_name]
    print(f"\n  Best model: {best_name} (AUC-ROC={fmt(best_result['auc_roc'])})")

    print("\n" + "="*60)
    print("STEP 8 - SHAP explainability")
    print("="*60)
    sv, top3_features = step9_shap(best_model, best_name, X_train_res, X_test_scaled, feature_names)

    print("\n" + "="*60)
    print("STEP 9 - Generate plots")
    print("="*60)
    step10_plots(results_list, y_test, final_df, feature_names, model_rf)

    print("\n" + "="*60)
    print("STEP 10 - Save results")
    print("="*60)
    step11_save_model(best_model, scaler)
    save_output_txt(results_list, y_test, best_name)
    step12_clinical_interpretation(top3_features, results_list, final_df)

    print(f"\n  PIPELINE COMPLETE - outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()