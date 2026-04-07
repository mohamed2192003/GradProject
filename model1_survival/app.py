"""
COVID-19 Survival Prediction — Streamlit GUI
============================================
Multi-page dashboard:
  1. Patient Risk Calculator  — real-time prediction + SHAP waterfall
  2. Model Performance        — ROC, PR, confusion matrices
  3. SHAP Explainability      — global feature importance plots
  4. Dataset Insights         — cohort statistics + lab distributions
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = r"f:\Graduation-Project\model1_survival"
OUTPUTS    = os.path.join(BASE, "outputs")
DATA_DIR   = r"f:\Graduation-Project\10k_synthea_covid19_csv"

MODEL_PATH  = os.path.join(BASE, "survival_model.pkl")
SCALER_PATH = os.path.join(BASE, "survival_scaler.pkl")

# ── Feature metadata ──────────────────────────────────────────────────────────
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

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 Survival Predictor",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background: #0f1117; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, #161b27 0%, #0f1117 100%);
      border-right: 1px solid #2d3748;
  }

  /* Cards */
  .metric-card {
      background: linear-gradient(135deg, #1e2533 0%, #252d3d 100%);
      border: 1px solid #2d3748;
      border-radius: 12px;
      padding: 20px 24px;
      text-align: center;
  }
  .metric-card h2 { font-size: 2.2rem; margin: 0; }
  .metric-card p  { color: #94a3b8; margin: 4px 0 0; font-size: .9rem; }

  /* Risk badges */
  .risk-low  { background:#064e3b; color:#6ee7b7; border:1px solid #059669;
               padding:6px 18px; border-radius:20px; font-weight:700; }
  .risk-med  { background:#78350f; color:#fcd34d; border:1px solid #d97706;
               padding:6px 18px; border-radius:20px; font-weight:700; }
  .risk-high { background:#7f1d1d; color:#fca5a5; border:1px solid #dc2626;
               padding:6px 18px; border-radius:20px; font-weight:700; }

  /* Section headings */
  .section-title {
      font-size: 1.4rem; font-weight: 700; color: #e2e8f0;
      border-left: 4px solid #6366f1; padding-left: 12px;
      margin: 24px 0 16px;
  }

  /* Probability bar fill */
  .prob-bar-wrap {
      background:#1e2533; border-radius:8px;
      height:22px; width:100%; margin-top:8px;
  }
  .prob-bar-fill {
      height:22px; border-radius:8px;
      transition: width 0.5s ease;
  }

  /* Feature input labels */
  label { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model & scaler (cached) ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_artifacts()


# ── Helper: build & scale feature vector ─────────────────────────────────────
def build_feature_vector(inputs: dict) -> np.ndarray:
    """Build a (1, 23) scaled numpy array from input dict."""
    vec = np.array([[inputs[f] for f in FEATURE_NAMES]], dtype=float)
    vec[:, CONT_IDX] = scaler.transform(vec[:, CONT_IDX])
    return vec


# ── Helper: load a plot image ─────────────────────────────────────────────────
def load_img(filename: str):
    path = os.path.join(OUTPUTS, filename)
    if os.path.exists(path):
        return Image.open(path)
    return None


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫁 COVID-19\nSurvival Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🩺 Patient Risk Calculator",
         "📊 Model Performance",
         "🔍 SHAP Explainability",
         "📈 Dataset Insights"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🧪 Normal Lab Reference")
    st.markdown(
        "<p style='color:#94a3b8;font-size:0.82rem;margin-top:-8px;'>"
        "Average values for healthy adults</p>",
        unsafe_allow_html=True,
    )

    LAB_NORMS = [
        ("SpO2",        "95 – 100 %",        97.0,  "%"),
        ("D-Dimer",     "< 500 ng/mL",       250.0, "ng/mL"),
        ("CRP",         "< 3 mg/L",          1.5,   "mg/L"),
        ("WBC",         "4.0 – 11.0 ×10³",  7.0,   "×10³/µL"),
        ("Lymphocytes", "1.0 – 4.8 ×10³",   2.5,   "×10³/µL"),
        ("Ferritin",    "12 – 300 ng/mL",    100.0, "ng/mL"),
        ("IL-6",        "< 7 pg/mL",         2.0,   "pg/mL"),
    ]

    for name, normal_range, avg, unit in LAB_NORMS:
        st.markdown(
            f"""
            <div style="
                background:#1e2533;
                border:1px solid #2d3748;
                border-radius:8px;
                padding:8px 12px;
                margin-bottom:6px;
            ">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="color:#e2e8f0;font-weight:600;font-size:0.9rem;">{name}</span>
                    <span style="color:#6ee7b7;font-size:0.78rem;font-weight:700;">✓ Normal</span>
                </div>
                <div style="color:#94a3b8;font-size:0.78rem;margin-top:2px;">
                    Range: <span style="color:#a5b4fc;">{normal_range}</span>
                </div>
                <div style="color:#94a3b8;font-size:0.78rem;">
                    Avg: <span style="color:#e2e8f0;font-weight:600;">{avg} {unit}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<p style='color:#64748b;font-size:0.75rem;margin-top:4px;'>"
        "⚠ Values vary by lab & population. Use clinical judgment.</p>",
        unsafe_allow_html=True,
    )

# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Patient Risk Calculator
# ══════════════════════════════════════════════════════════════════════════════╗
if page == "🩺 Patient Risk Calculator":
    st.markdown("# 🩺 Patient Risk Calculator")
    st.markdown("Enter patient data to get a real-time COVID-19 survival prediction.")
    st.markdown("---")

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("patient_form"):
        col_a, col_b, col_c = st.columns(3)

        # ─ Demographics ───────────────────────────────────────────────────────
        with col_a:
            st.markdown('<div class="section-title">Demographics</div>',
                        unsafe_allow_html=True)
            age    = st.slider("Age (years)", 0, 110, 55)
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True)

        # ─ Comorbidities ──────────────────────────────────────────────────────
        with col_b:
            st.markdown('<div class="section-title">Comorbidities</div>',
                        unsafe_allow_html=True)
            has_hypertension  = st.checkbox("Hypertension")
            has_diabetes      = st.checkbox("Diabetes")
            has_obesity       = st.checkbox("Obesity")
            has_asthma        = st.checkbox("Asthma")
            has_COPD          = st.checkbox("COPD / Pulmonary disease")
            has_heart_disease = st.checkbox("Heart / Cardiac disease")

        # ─ Hospital / Care ────────────────────────────────────────────────────
        with col_c:
            st.markdown('<div class="section-title">Hospital Status</div>',
                        unsafe_allow_html=True)
            was_hospitalized = st.checkbox("Hospitalized (inpatient/emergency)")
            is_icu           = st.checkbox("ICU / Intensive care")
            had_care_plan    = st.checkbox("COVID care plan active")

        st.markdown("---")

        # ─ Lab values ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Laboratory Values</div>',
                    unsafe_allow_html=True)

        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            d_dimer  = st.number_input("D-Dimer (ng/mL)",  0.0, 5000.0, 300.0, step=10.0)
            ferritin = st.number_input("Ferritin (ng/mL)", 0.0, 5000.0, 423.0, step=10.0)
        with lc2:
            crp      = st.number_input("CRP (mg/L)",       0.0, 400.0,  10.2,  step=0.5)
            wbc      = st.number_input("WBC (10³/µL)",     0.0, 50.0,   3.8,   step=0.1)
        with lc3:
            spo2         = st.number_input("SpO2 (%)",          50.0, 100.0, 98.0,  step=0.5)
            lymphocytes  = st.number_input("Lymphocytes (10³/µL)", 0.0, 15.0, 1.5,   step=0.1)
        with lc4:
            il6 = st.number_input("IL-6 (pg/mL)", 0.0, 10000.0, 0.0, step=1.0)
            st.markdown("")

        submitted = st.form_submit_button("🔮 Predict Survival", use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if submitted:
        # Derived binary flags
        comorbidity_count = sum([
            has_hypertension, has_diabetes, has_obesity,
            has_asthma, has_COPD, has_heart_disease
        ])
        low_spo2_flag        = 1 if spo2 < 94 else 0
        high_crp_flag        = 1 if crp > 10 else 0
        high_d_dimer_flag    = 1 if d_dimer > 500 else 0
        low_lymphocytes_flag = 1 if lymphocytes < 1.0 else 0

        inputs = {
            "age"              : age,
            "gender"           : 1 if gender == "Male" else 0,
            "has_hypertension" : int(has_hypertension),
            "has_diabetes"     : int(has_diabetes),
            "has_obesity"      : int(has_obesity),
            "has_asthma"       : int(has_asthma),
            "has_COPD"         : int(has_COPD),
            "has_heart_disease": int(has_heart_disease),
            "comorbidity_count": comorbidity_count,
            "d_dimer"          : d_dimer,
            "il6"              : il6,
            "ferritin"         : ferritin,
            "crp"              : crp,
            "spo2"             : spo2,
            "wbc"              : wbc,
            "lymphocytes"      : lymphocytes,
            "low_spo2"         : low_spo2_flag,
            "high_crp"         : high_crp_flag,
            "high_d_dimer"     : high_d_dimer_flag,
            "low_lymphocytes"  : low_lymphocytes_flag,
            "was_hospitalized" : int(was_hospitalized),
            "is_icu"           : int(is_icu),
            "had_care_plan"    : int(had_care_plan),
        }

        X_input = build_feature_vector(inputs)
        survival_prob = model.predict_proba(X_input)[0, 1]
        death_prob    = 1.0 - survival_prob
        prediction    = "SURVIVED" if survival_prob >= 0.35 else "HIGH RISK — DIED"

        st.markdown("---")
        st.markdown("## 🎯 Prediction Result")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color:{'#6ee7b7' if survival_prob >= 0.35 else '#fca5a5'}">
                    {survival_prob*100:.1f}%
                </h2>
                <p>Survival Probability</p>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color:#fca5a5">{death_prob*100:.1f}%</h2>
                <p>Mortality Probability</p>
            </div>""", unsafe_allow_html=True)
        with r3:
            risk_class = "low" if survival_prob >= 0.65 else ("med" if survival_prob >= 0.35 else "high")
            risk_label = "LOW RISK" if risk_class == "low" else ("MODERATE" if risk_class == "med" else "HIGH RISK")
            st.markdown(f"""
            <div class="metric-card">
                <h2><span class="risk-{risk_class}">{risk_label}</span></h2>
                <p>Risk Classification</p>
            </div>""", unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="color:#a5b4fc">{comorbidity_count}</h2>
                <p>Active Comorbidities</p>
            </div>""", unsafe_allow_html=True)

        # Survival probability gauge
        st.markdown("#### Survival Probability")
        bar_color   = "#6ee7b7" if survival_prob >= 0.65 else ("#fcd34d" if survival_prob >= 0.35 else "#fca5a5")
        st.markdown(f"""
        <div class="prob-bar-wrap">
          <div class="prob-bar-fill" style="width:{survival_prob*100:.1f}%;background:{bar_color};"></div>
        </div>
        <p style="color:#94a3b8;font-size:.85rem;margin-top:4px;">
          Threshold = 0.35 | Survival Prob = {survival_prob:.3f}
        </p>""", unsafe_allow_html=True)

        # Clinical flags
        st.markdown("#### ⚡ Active Clinical Alerts")
        alerts = []
        if low_spo2_flag:     alerts.append("🔴 **Low SpO2** (< 94%) — critical hypoxia")
        if high_crp_flag:     alerts.append("🟠 **High CRP** (> 10 mg/L) — systemic inflammation")
        if high_d_dimer_flag: alerts.append("🟠 **High D-Dimer** (> 500 ng/mL) — thrombotic risk")
        if low_lymphocytes_flag: alerts.append("🟠 **Low Lymphocytes** (< 1.0) — immune suppression")
        if is_icu:            alerts.append("🔴 **ICU admission** — critical condition")
        if age > 65:          alerts.append("🟡 **Age > 65** — elevated baseline risk")

        if alerts:
            for a in alerts:
                st.markdown(f"- {a}")
        else:
            st.success("✅ No critical clinical flags triggered")

        # SHAP explanation for this patient
        st.markdown("#### 🔍 Feature Contribution (SHAP Waterfall)")
        try:
            X_df       = pd.DataFrame(X_input, columns=FEATURE_NAMES)
            explainer  = shap.TreeExplainer(model)
            sv         = explainer.shap_values(X_df)
            if isinstance(sv, list):
                sv = sv[1]
            sv_row = sv[0]

            shap_df = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "SHAP"   : sv_row,
                "Value"  : [inputs[f] for f in FEATURE_NAMES],
            }).sort_values("SHAP", key=abs, ascending=True).tail(12)

            colors = ["#fca5a5" if v < 0 else "#6ee7b7" for v in shap_df["SHAP"]]
            fig = go.Figure(go.Bar(
                x=shap_df["SHAP"],
                y=[f"{r.Feature} = {r.Value:.2f}" for _, r in shap_df.iterrows()],
                orientation="h",
                marker_color=colors,
            ))
            fig.update_layout(
                title="SHAP Contribution (green = towards survival, red = towards death)",
                xaxis_title="SHAP Value",
                plot_bgcolor="#1e2533",
                paper_bgcolor="#0f1117",
                font_color="#e2e8f0",
                height=420,
                margin=dict(l=180, r=20, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP could not be computed: {e}")

        # Clinical recommendation
        st.markdown("---")
        if survival_prob < 0.35:
            st.error("""
            ⚠️ **CLINICAL RECOMMENDATION: HIGH RISK — ESCALATE IMMEDIATELY**
            
            This patient profile matches high-mortality patterns. Recommended actions:
            - Immediate intensivist review
            - Aggressive oxygen therapy / mechanical ventilation assessment
            - Anticoagulation workup (elevated D-Dimer)
            - Cytokine storm management protocol
            - ICU admission consideration
            """)
        elif survival_prob < 0.65:
            st.warning("""
            ⚠️ **CLINICAL RECOMMENDATION: MODERATE RISK — MONITOR CLOSELY**
            
            - Close monitoring every 4–6 hours
            - Repeat labs in 12–24 hours
            - Consider step-up to high-dependency unit
            - Re-run risk score if condition changes
            """)
        else:
            st.success("""
            ✅ **CLINICAL RECOMMENDATION: LOWER RISK — STANDARD MONITORING**
            
            - Standard COVID ward monitoring
            - Repeat risk assessment in 24–48 hours
            - Consider home isolation if criteria met
            - Patient and family education on warning signs
            """)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════╗
elif page == "📊 Model Performance":
    st.markdown("# 📊 Model Performance")
    st.markdown("Evaluation of all three models on the held-out **20% test set** (n = 1,822 patients).")
    st.markdown("---")

    # Comparison table
    st.markdown("### Comparison Table (threshold = 0.35)")
    perf_df = pd.DataFrame({
        "Model"          : ["Logistic Regression", "Random Forest", "XGBoost ★ (Best)"],
        "AUC-ROC"        : [0.9975, 0.9919, 0.9980],
        "PR-AUC"         : [0.9999, 0.9993, 0.9999],
        "Recall (Died)"  : [0.9859, 0.9577, 0.9577],
        "F1 (Died)"      : [0.9272, 0.9784, 0.9784],
        "Accuracy"       : ["99%", "~100%", "~100%"],
    })
    st.dataframe(
        perf_df.style
            .highlight_max(subset=["AUC-ROC","PR-AUC","Recall (Died)","F1 (Died)"],
                           color="#1a3a2a")
            .format({"AUC-ROC": "{:.4f}", "PR-AUC": "{:.4f}",
                     "Recall (Died)": "{:.4f}", "F1 (Died)": "{:.4f}"}),
        use_container_width=True,
    )

    # AUC chart
    st.markdown("### AUC-ROC & PR-AUC at a Glance")
    models = ["Logistic Regression", "Random Forest", "XGBoost"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="AUC-ROC", x=models, y=[0.9975, 0.9919, 0.9980],
                         marker_color="#6366f1"))
    fig.add_trace(go.Bar(name="PR-AUC",  x=models, y=[0.9999, 0.9993, 0.9999],
                         marker_color="#22d3ee"))
    fig.update_layout(
        barmode="group", plot_bgcolor="#1e2533", paper_bgcolor="#0f1117",
        font_color="#e2e8f0", yaxis=dict(range=[0.98, 1.0]),
        legend=dict(bgcolor="#1e2533"), height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Plot 1: ROC curves
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ROC Curves")
        img = load_img("plot1_roc_curves.png")
        if img: st.image(img, use_container_width=True)
    with c2:
        st.markdown("### Precision-Recall Curves")
        img = load_img("plot2_pr_curves.png")
        if img: st.image(img, use_container_width=True)

    st.markdown("### Confusion Matrices (all 3 models)")
    img = load_img("plot3_confusion_matrices.png")
    if img: st.image(img, use_container_width=True)

    st.markdown("### Confusion Matrix + Calibration Curves")
    img = load_img("cm_calibration.png")
    if img: st.image(img, use_container_width=True)

    st.info("""
    **Reading the calibration curve:** A well-calibrated model follows the diagonal.
    The XGBoost and LR models are very well calibrated, meaning predicted probabilities
    are close to true observed frequencies.
    """)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SHAP Explainability
# ══════════════════════════════════════════════════════════════════════════════╗
elif page == "🔍 SHAP Explainability":
    st.markdown("# 🔍 SHAP Explainability (XGBoost)")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) reveals **why** the model makes each prediction.
    - **Positive SHAP value** → pushes prediction toward *survival*
    - **Negative SHAP value** → pushes prediction toward *death*
    """)
    st.markdown("---")

    st.markdown("### 🐝 Beeswarm Summary Plot")
    st.markdown("Each dot = one test patient. Color = feature value (red=high, blue=low).")
    img = load_img("shap_beeswarm.png")
    if img: st.image(img, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Global Feature Importance (Mean |SHAP|)")
    img = load_img("shap_bar.png")
    if img: st.image(img, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📉 SHAP Dependence Plots — Top 3 Features")
    st.markdown("Shows how each feature value affects the survival probability.")
    img = load_img("shap_dependence.png")
    if img: st.image(img, use_container_width=True)

    # Key takeaways
    st.markdown("---")
    st.markdown("### 💡 Key Takeaways from SHAP")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🏥 Hospitalization**
        
        The single strongest predictor of death. Patients admitted as inpatient
        or emergency had dramatically lower survival probability in the SHAP analysis.
        """)
    with col2:
        st.markdown("""
        **🩸 D-Dimer**
        
        Elevated D-Dimer (>500 ng/mL) is the second most important signal,
        reflecting the profound thrombotic complications seen in severe COVID-19.
        """)
    with col3:
        st.markdown("""
        **👴 Age**
        
        Age has a strong dose-response effect — mortality risk rises steeply
        after age 65, with the sharpest increase above 75 years.
        """)


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Dataset Insights
# ══════════════════════════════════════════════════════════════════════════════╗
elif page == "📈 Dataset Insights":
    st.markdown("# 📈 Dataset Insights")
    st.markdown("Cohort statistics from the Synthea 10k COVID-19 synthetic dataset.")
    st.markdown("---")

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""<div class="metric-card">
            <h2 style="color:#a5b4fc">9,106</h2><p>COVID-19 Patients</p>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="metric-card">
            <h2 style="color:#6ee7b7">8,752</h2><p>Survived (96.1%)</p>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="metric-card">
            <h2 style="color:#fca5a5">354</h2><p>Died within 60 days (3.9%)</p>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""<div class="metric-card">
            <h2 style="color:#fcd34d">24.7:1</h2><p>Class Imbalance Ratio</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Outcome pie chart
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Outcome Distribution")
        fig = go.Figure(go.Pie(
            labels=["Survived", "Died"],
            values=[8752, 354],
            marker_colors=["#4CAF81", "#EF5350"],
            hole=0.45,
        ))
        fig.update_layout(
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font_color="#e2e8f0", height=300,
            legend=dict(bgcolor="#1e2533"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Class Balancing (SMOTE)")
        fig = go.Figure(go.Bar(
            x=["Before SMOTE\n(Survived)", "Before SMOTE\n(Died)",
               "After SMOTE\n(Survived)", "After SMOTE\n(Died)"],
            y=[7001, 283, 7001, 7001],
            marker_color=["#4CAF81", "#EF5350", "#4CAF81", "#f87171"],
            text=[7001, 283, 7001, 7001],
            textposition="auto",
        ))
        fig.update_layout(
            plot_bgcolor="#1e2533", paper_bgcolor="#0f1117",
            font_color="#e2e8f0", height=300, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Age distribution
    st.markdown("### Age Distribution by Outcome")
    img = load_img("plot5_age_distribution.png")
    if img: st.image(img, use_container_width=True)

    # Lab boxplots
    st.markdown("### Lab Values by Outcome")
    img = load_img("plot6_lab_boxplots.png")
    if img: st.image(img, use_container_width=True)

    # Random Forest feature importance
    st.markdown("### Top 15 Features — Random Forest Importance")
    img = load_img("plot4_rf_feature_importance.png")
    if img: st.image(img, use_container_width=True)

    # Comorbidity breakdown
    st.markdown("---")
    st.markdown("### Comorbidity Prevalence in COVID-19 Cohort")
    comorbidity_data = {
        "Condition"   : ["Obesity", "Diabetes", "Hypertension", "Heart Disease",
                         "COPD/Pulmonary", "Asthma"],
        "Prevalence %": [38.1, 31.6, 24.8, 6.5, 1.5, 1.9],
    }
    df_comorbid = pd.DataFrame(comorbidity_data).sort_values("Prevalence %")
    fig = px.bar(
        df_comorbid, x="Prevalence %", y="Condition", orientation="h",
        color="Prevalence %", color_continuous_scale="Purples",
    )
    fig.update_layout(
        plot_bgcolor="#1e2533", paper_bgcolor="#0f1117",
        font_color="#e2e8f0", height=320,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)
