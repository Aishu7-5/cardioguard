"""
🧠 Cardiovascular Stroke Prediction System
==========================================
Full Streamlit web application with:
  • User authentication (login / signup)
  • Dashboard with dataset stats
  • 20-parameter prediction form
  • Patient history tracking
  • Clean clinical UI
"""

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib, json, hashlib, os, warnings
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StrokeSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0a1628 0%, #112240 100%);
    border-right: 1px solid #1e3a5f;
  }
  [data-testid="stSidebar"] * { color: #ccd6f6 !important; }
  [data-testid="stSidebar"] .stButton>button {
    background: rgba(100,255,218,0.08);
    border: 1px solid #64ffda44;
    color: #64ffda !important;
    border-radius: 6px;
    width: 100%;
    font-weight: 600;
    transition: all .2s;
  }
  [data-testid="stSidebar"] .stButton>button:hover {
    background: rgba(100,255,218,0.18);
    border-color: #64ffda;
  }

  /* ── Main background ── */
  .main { background: #0d1b2a; color: #ccd6f6; }
  .block-container { padding-top: 1.5rem; }

  /* ── Cards ── */
  .card {
    background: linear-gradient(135deg, #112240 0%, #0d1b2a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }
  .metric-card {
    background: linear-gradient(135deg, #112240 0%, #172a45 100%);
    border: 1px solid #233d5e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }
  .metric-card h2 { color: #64ffda; font-size: 2rem; margin: 0; }
  .metric-card p  { color: #8892b0; margin: 0; font-size: .85rem; }

  /* ── Risk badge ── */
  .risk-low      { background:#0d3b2e; color:#64ffda; border:1px solid #64ffda66; }
  .risk-moderate { background:#3b2d0a; color:#ffd166; border:1px solid #ffd16666; }
  .risk-high     { background:#3b0d0d; color:#ff6b6b; border:1px solid #ff6b6b66; }
  .risk-badge {
    display:inline-block; border-radius:20px; padding:.3rem 1.2rem;
    font-size:1.1rem; font-weight:700; letter-spacing:.05em;
  }

  /* ── Form labels ── */
  .stSelectbox label, .stNumberInput label, .stSlider label { color: #8892b0 !important; }

  /* ── Inputs ── */
  .stNumberInput input, .stTextInput input, .stSelectbox select {
    background: #0a1628 !important;
    color: #ccd6f6 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
  }

  /* ── Primary button ── */
  .stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #64ffda, #00b4d8) !important;
    color: #0a1628 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    padding: .6rem 2rem !important;
  }

  /* ── Section headers ── */
  .section-header {
    color: #64ffda;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: .15em;
    text-transform: uppercase;
    padding-bottom: .4rem;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 1rem;
  }

  /* ── Table ── */
  .dataframe { background: #112240 !important; }

  /* Gauge chart area */
  .gauge-container { text-align:center; padding: .5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────
for key, default in {
    "authenticated": False,
    "username": "",
    "page": "login",
    "history": [],
    "users": {"admin": hashlib.sha256("admin123".encode()).hexdigest()},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model    = tf.keras.models.load_model("stroke_model.keras")
        scaler   = joblib.load("scaler.pkl")
        with open("model_metadata.json") as f:
            meta = json.load(f)
        return model, scaler, meta
    except Exception as e:
        return None, None, None

model, scaler, meta = load_artifacts()

# ─────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def engineer_input(d: dict) -> np.ndarray:
    """Replicate feature engineering for a single patient dict."""
    bmi = d["bmi"]
    age = d["age"]
    sys_bp = d["systolic_bp"]
    dia_bp = d["diastolic_bp"]
    gluc   = d["avg_glucose_level"]
    chol   = d["cholesterol"]
    hdl    = d["hdl"]

    age_norm = (age - 18) / 72
    bmi_norm = (bmi - 15) / 40

    risk_score = min(10, (
        d["hypertension"]   * 2.0 +
        d["heart_disease"]  * 2.0 +
        d["family_history"] * 1.5 +
        (1.0 if bmi > 30 else 0) +
        (1.5 if gluc > 125 else 0) +
        (1.0 if chol > 240 else 0) +
        (1.0 if hdl < 40 else 0)
    ))

    lifestyle_risk = (
        d["smoking_status"]    * 1.5 +
        d["alcohol_intake"]    * 0.8 +
        (3 - d["physical_activity"]) * 1.2 +
        d["stress_level"]      * 0.3
    )

    age_group = 0 if age < 30 else (1 if age < 45 else (2 if age < 60 else (3 if age < 75 else 4)))

    pulse_pressure        = sys_bp - dia_bp
    mean_arterial_pressure = dia_bp + pulse_pressure / 3
    chol_hdl_ratio        = min(20, chol / max(1, hdl))

    bmi_cat = 0 if bmi < 18.5 else (1 if bmi < 25 else (2 if bmi < 30 else (3 if bmi < 35 else 4)))

    glucose_bmi_interaction = (gluc * bmi) / 1000

    bp_severity = (
        3 if (sys_bp >= 180 or dia_bp >= 120) else
        2 if (sys_bp >= 160 or dia_bp >= 100) else
        1 if (sys_bp >= 130 or dia_bp >= 80)  else 0
    )

    base = [
        d["age"], d["gender"], d["bmi"], d["height"], d["weight"],
        d["hypertension"], d["heart_disease"], d["ever_married"], d["family_history"],
        d["avg_glucose_level"], d["fasting_blood_sugar"], d["cholesterol"],
        d["hdl"], d["ldl"],
        d["smoking_status"], d["alcohol_intake"], d["physical_activity"],
        d["stress_level"], d["systolic_bp"], d["diastolic_bp"],
        risk_score, lifestyle_risk, age_group, pulse_pressure,
        mean_arterial_pressure, chol_hdl_ratio, bmi_cat,
        glucose_bmi_interaction, bp_severity,
    ]
    return np.array(base, dtype=np.float32).reshape(1, -1)

def predict_stroke(patient: dict):
    if model is None:
        return None, None, None
    x = engineer_input(patient)
    x_scaled = scaler.transform(x)
    prob = float(model.predict(x_scaled, verbose=0)[0][0])
    risk = "HIGH" if prob >= 0.60 else ("MODERATE" if prob >= 0.30 else "LOW")
    return prob, risk, x[0]

def gauge_chart(probability: float):
    pct = probability * 100
    color = "#ff6b6b" if pct >= 60 else ("#ffd166" if pct >= 30 else "#64ffda")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        delta={"reference": 30, "increasing": {"color": "#ff6b6b"},
               "decreasing": {"color": "#64ffda"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#8892b0"},
            "bar":  {"color": color, "thickness": 0.22},
            "bgcolor": "#112240",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],   "color": "#0d3b2e"},
                {"range": [30, 60],  "color": "#3b2d0a"},
                {"range": [60, 100], "color": "#3b0d0d"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75, "value": pct
            },
        },
        title={"text": "Stroke Probability", "font": {"size": 16, "color": "#8892b0"}},
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#0d1b2a",
        font={"color": "#ccd6f6"},
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────
#  PAGES
# ─────────────────────────────────────────────────────────────────────────

# ══ AUTH PAGE ══════════════════════════════════════════════════════════════
def auth_page():
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding:2rem 0 1rem'>
          <div style='font-size:3rem'>🧠</div>
          <h1 style='color:#64ffda; font-size:2rem; font-weight:700; margin:.5rem 0 0'>StrokeSense AI</h1>
          <p style='color:#8892b0; margin-bottom:2rem'>Clinical Stroke Risk Prediction System</p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        with tab_login:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            uname = st.text_input("Username", key="li_user")
            pw    = st.text_input("Password", type="password", key="li_pw")
            if st.button("Login", use_container_width=True, type="primary"):
                if uname in st.session_state.users and st.session_state.users[uname] == hash_pw(pw):
                    st.session_state.authenticated = True
                    st.session_state.username = uname
                    st.session_state.page = "dashboard"
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption("Demo credentials → username: **admin** | password: **admin123**")

        with tab_signup:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            nu = st.text_input("Choose Username", key="su_user")
            np_= st.text_input("Choose Password", type="password", key="su_pw")
            nc = st.text_input("Confirm Password", type="password", key="su_cf")
            if st.button("Create Account", use_container_width=True, type="primary"):
                if not nu:
                    st.warning("Enter a username")
                elif nu in st.session_state.users:
                    st.error("Username already exists")
                elif np_ != nc:
                    st.error("Passwords do not match")
                elif len(np_) < 6:
                    st.warning("Password must be ≥ 6 characters")
                else:
                    st.session_state.users[nu] = hash_pw(np_)
                    st.success("Account created! Please login.")
            st.markdown("</div>", unsafe_allow_html=True)

# ══ SIDEBAR ════════════════════════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:.8rem 0; border-bottom:1px solid #1e3a5f; margin-bottom:1rem;'>
          <div style='font-size:1.4rem; font-weight:700; color:#64ffda'>🧠 StrokeSense AI</div>
          <div style='font-size:.8rem; color:#8892b0; margin-top:.2rem'>
            Welcome, <b style='color:#ccd6f6'>{st.session_state.username}</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        pages = [
            ("📊", "Dashboard",  "dashboard"),
            ("🔬", "Predict",    "predict"),
            ("📋", "History",    "history"),
        ]
        for icon, label, key in pages:
            active = "background:rgba(100,255,218,.12);" if st.session_state.page == key else ""
            if st.button(f"{icon}  {label}", key=f"nav_{key}",
                         use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.markdown("---")
        if model is None:
            st.error("⚠️ Model not loaded\nRun train_model.py first")
        else:
            st.success(f"✅ Model ready\nAUC = {meta.get('test_auc','N/A')}")

        st.markdown("---")
        if st.button("🔒 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.page = "login"
            st.rerun()

# ══ DASHBOARD ══════════════════════════════════════════════════════════════
def dashboard_page():
    st.markdown("## 📊 Dashboard")

    # Dataset stats
    try:
        df = pd.read_csv("stroke_clinical_dataset.csv")
        has_data = True
    except:
        has_data = False

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Total Records",    f"{len(df):,}"       if has_data else "N/A",  "Dataset size"),
        ("Stroke Cases",     f"{df.stroke.sum():,}" if has_data else "N/A", "Positive class"),
        ("Stroke Rate",      f"{df.stroke.mean()*100:.1f}%" if has_data else "N/A", "Class ratio"),
        ("Model AUC",        str(meta.get("test_auc","N/A")) if meta else "N/A", "Performance"),
    ]
    for col, (title, val, sub) in zip([c1,c2,c3,c4], metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <p style='color:#64ffda;font-size:.75rem;font-weight:700;letter-spacing:.1em'>{title.upper()}</p>
              <h2>{val}</h2>
              <p>{sub}</p>
            </div>""", unsafe_allow_html=True)

    if has_data:
        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Age Distribution by Stroke Status**")
            fig = px.histogram(
                df, x="age", color="stroke",
                color_discrete_map={0:"#1e3a5f", 1:"#ff6b6b"},
                labels={"stroke": "Stroke", "age": "Age"},
                barmode="overlay", opacity=0.75, nbins=35,
            )
            fig.update_layout(
                paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
                font_color="#ccd6f6", legend_title="Stroke",
                height=300, margin=dict(l=10,r=10,t=10,b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("**Risk Factor Prevalence**")
            factors = {
                "Hypertension":    df.hypertension.mean(),
                "Heart Disease":   df.heart_disease.mean(),
                "Family History":  df.family_history.mean(),
                "Obesity (BMI>30)": (df.bmi > 30).mean(),
                "High Glucose":    (df.avg_glucose_level > 125).mean(),
                "Current Smoker":  (df.smoking_status == 2).mean(),
            }
            fig2 = go.Figure(go.Bar(
                x=list(factors.values()),
                y=list(factors.keys()),
                orientation="h",
                marker_color="#64ffda",
            ))
            fig2.update_traces(text=[f"{v*100:.1f}%" for v in factors.values()],
                               textposition="outside")
            fig2.update_layout(
                paper_bgcolor="#0d1b2a", plot_bgcolor="#112240",
                font_color="#ccd6f6", xaxis=dict(tickformat=".0%"),
                height=300, margin=dict(l=10,r=60,t=10,b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Correlation heatmap
        st.markdown("**Feature Correlation with Stroke**")
        corr_cols = ["age","bmi","systolic_bp","avg_glucose_level","cholesterol",
                     "hypertension","heart_disease","stress_level","stroke"]
        corr = df[corr_cols].corr()["stroke"].drop("stroke").sort_values(ascending=False)

        fig3 = go.Figure(go.Bar(
            x=corr.values,
            y=corr.index,
            orientation="h",
            marker_color=["#ff6b6b" if v > 0 else "#64ffda" for v in corr.values],
        ))
        fig3.update_layout(
            paper_bgcolor="#0d1b2a", plot_bgcolor="#112240",
            font_color="#ccd6f6", height=280,
            margin=dict(l=10,r=20,t=10,b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ══ PREDICTION PAGE ════════════════════════════════════════════════════════
def predict_page():
    st.markdown("## 🔬 Stroke Risk Prediction")
    st.caption("Enter all 20 clinical parameters to compute stroke probability.")

    if model is None:
        st.error("⚠️  Model not loaded. Please run `train_model.py` first.")
        return

    # ── Input form ────────────────────────────────────────────────────────
    with st.form("prediction_form"):

        # ── Demographics ──────────────────────────────────────────────────
        st.markdown("<div class='section-header'>👤 Demographics</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        age          = c1.number_input("Age (years)",  18, 90,  55)
        gender       = c2.selectbox("Gender",        ["Female","Male"])
        height       = c3.number_input("Height (cm)", 140, 220, 168)
        weight       = c4.number_input("Weight (kg)",  40, 200,  75)
        bmi_auto     = weight / (height / 100) ** 2
        c1.metric("Auto-BMI", f"{bmi_auto:.1f}")
        ever_married = c2.selectbox("Ever Married",  ["No","Yes"])
        family_hist  = c3.selectbox("Family History of Stroke", ["No","Yes"])

        # ── Vitals ────────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>💓 Blood Pressure & Vitals</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        systolic_bp  = c1.number_input("Systolic BP (mmHg)",   80,  220, 122)
        diastolic_bp = c2.number_input("Diastolic BP (mmHg)",  50,  140,  78)
        hypertension = c3.selectbox("Hypertension Diagnosed",  ["No","Yes"])
        heart_disease= c4.selectbox("Heart Disease",           ["No","Yes"])

        # ── Metabolic ─────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🩸 Metabolic Panel</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        avg_glucose    = c1.number_input("Avg Glucose (mg/dL)",  55, 450,  99)
        fasting_bs     = c2.number_input("Fasting Blood Sugar",  60, 400,  95)
        cholesterol    = c3.number_input("Total Cholesterol",   100, 400, 200)
        hdl            = c4.number_input("HDL (mg/dL)",          20, 100,  55)
        c1, c2 = st.columns(2)
        ldl            = c1.number_input("LDL (mg/dL)",          40, 300, 130)

        # ── Lifestyle ─────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🏃 Lifestyle Factors</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        smoking   = c1.selectbox("Smoking Status", ["Never","Former","Current"])
        alcohol   = c2.selectbox("Alcohol Intake", ["None","Light","Moderate","Heavy"])
        activity  = c3.selectbox("Physical Activity", ["Sedentary","Light","Moderate","Active"])
        stress    = c4.slider("Stress Level (1–10)", 1, 10, 5)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ PREDICT STROKE RISK", type="primary",
                                          use_container_width=True)

    # ── Run prediction ────────────────────────────────────────────────────
    if submitted:
        patient = {
            "age":                age,
            "gender":             1 if gender == "Male" else 0,
            "height":             float(height),
            "weight":             float(weight),
            "bmi":                round(bmi_auto, 2),
            "hypertension":       1 if hypertension == "Yes" else 0,
            "heart_disease":      1 if heart_disease == "Yes" else 0,
            "ever_married":       1 if ever_married == "Yes" else 0,
            "family_history":     1 if family_hist == "Yes" else 0,
            "avg_glucose_level":  float(avg_glucose),
            "fasting_blood_sugar":int(fasting_bs),
            "cholesterol":        int(cholesterol),
            "hdl":                int(hdl),
            "ldl":                int(ldl),
            "smoking_status":     ["Never","Former","Current"].index(smoking),
            "alcohol_intake":     ["None","Light","Moderate","Heavy"].index(alcohol),
            "physical_activity":  ["Sedentary","Light","Moderate","Active"].index(activity),
            "stress_level":       int(stress),
            "systolic_bp":        int(systolic_bp),
            "diastolic_bp":       int(diastolic_bp),
        }

        prob, risk, raw_features = predict_stroke(patient)

        if prob is None:
            st.error("Prediction failed – model not available.")
            return

        # ── Save to history ────────────────────────────────────────────────
        record = {
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            "age":         age,
            "gender":      gender,
            "bmi":         round(bmi_auto, 1),
            "systolic_bp": systolic_bp,
            "probability": round(prob * 100, 1),
            "risk":        risk,
        }
        st.session_state.history.insert(0, record)

        # ── Display results ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        col_gauge, col_detail = st.columns([1, 1.2])

        with col_gauge:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)

        with col_detail:
            badge_class = {"LOW":"risk-low","MODERATE":"risk-moderate","HIGH":"risk-high"}[risk]
            risk_advice = {
                "LOW": "Maintain healthy lifestyle. Routine check-ups recommended.",
                "MODERATE": "Consult a physician. Address modifiable risk factors urgently.",
                "HIGH": "Immediate medical consultation strongly advised. High cardiovascular risk detected.",
            }

            st.markdown(f"""
            <div class='card' style='margin-top:1rem'>
              <p style='color:#8892b0;font-size:.8rem;margin-bottom:.5rem'>RISK CLASSIFICATION</p>
              <div class='risk-badge {badge_class}'>{risk} RISK</div>
              <hr style='border-color:#1e3a5f;margin:1rem 0'>
              <p style='color:#8892b0;font-size:.85rem;margin:0'>
                <b style='color:#ccd6f6'>Stroke Probability:</b>
                <span style='color:#64ffda;font-size:1.3rem;font-weight:700'>
                  &nbsp;{prob*100:.1f}%
                </span>
              </p>
              <p style='color:#8892b0;font-size:.82rem;margin-top:.8rem;line-height:1.6'>
                ⚕️ {risk_advice[risk]}
              </p>
            </div>
            """, unsafe_allow_html=True)

            # Key risk flags
            flags = []
            if patient["hypertension"]:         flags.append("🔴 Hypertension")
            if patient["heart_disease"]:         flags.append("🔴 Heart Disease")
            if patient["family_history"]:        flags.append("🟡 Family History")
            if avg_glucose > 125:                flags.append("🔴 High Glucose")
            if cholesterol > 240:                flags.append("🟡 High Cholesterol")
            if bmi_auto > 30:                    flags.append("🟡 Obesity")
            if systolic_bp > 160:                flags.append("🔴 Severe Hypertension")
            if patient["smoking_status"] == 2:   flags.append("🟡 Current Smoker")
            if stress >= 8:                      flags.append("🟡 High Stress")

            if flags:
                st.markdown("**Active Risk Flags:**")
                for f in flags:
                    st.markdown(f"&nbsp;&nbsp;{f}")
            else:
                st.success("✅ No major acute risk flags detected.")

# ══ HISTORY PAGE ═══════════════════════════════════════════════════════════
def history_page():
    st.markdown("## 📋 Prediction History")

    if not st.session_state.history:
        st.info("No predictions yet. Go to the **Predict** tab to get started.")
        return

    df_h = pd.DataFrame(st.session_state.history)

    # Summary cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <p style='color:#64ffda;font-size:.75rem;font-weight:700'>TOTAL PREDICTIONS</p>
            <h2>{len(df_h)}</h2></div>""", unsafe_allow_html=True)
    with c2:
        high_count = (df_h.risk == "HIGH").sum()
        st.markdown(f"""<div class='metric-card'>
            <p style='color:#ff6b6b;font-size:.75rem;font-weight:700'>HIGH RISK</p>
            <h2 style='color:#ff6b6b'>{high_count}</h2></div>""", unsafe_allow_html=True)
    with c3:
        avg_prob = df_h.probability.mean()
        st.markdown(f"""<div class='metric-card'>
            <p style='color:#ffd166;font-size:.75rem;font-weight:700'>AVG PROBABILITY</p>
            <h2 style='color:#ffd166'>{avg_prob:.1f}%</h2></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Table
    st.dataframe(
        df_h.style.background_gradient(subset=["probability"], cmap="Reds"),
        use_container_width=True,
    )

    # Trend chart
    if len(df_h) > 1:
        fig = px.line(
            df_h[::-1], x="timestamp", y="probability",
            color="risk",
            color_discrete_map={"LOW":"#64ffda","MODERATE":"#ffd166","HIGH":"#ff6b6b"},
            markers=True, title="Probability Trend",
        )
        fig.update_layout(
            paper_bgcolor="#0d1b2a", plot_bgcolor="#112240",
            font_color="#ccd6f6", height=280,
            margin=dict(l=10,r=10,t=40,b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    auth_page()
else:
    sidebar()
    page = st.session_state.page
    if page == "dashboard":
        dashboard_page()
    elif page == "predict":
        predict_page()
    elif page == "history":
        history_page()
