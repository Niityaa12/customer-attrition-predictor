import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Attrition Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Font & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: #0f1117; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1d2e 0%, #12151f 100%);
    border-right: 1px solid #2a2d3e;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #a0aec0 !important;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.03em;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 40%, #f093fb 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}
.hero-banner h1 { color: #fff; font-size: 2rem; font-weight: 700; margin: 0; }
.hero-banner p  { color: rgba(255,255,255,0.85); font-size: 0.95rem; margin: 0.4rem 0 0; }

/* ── Metric cards ── */
.metric-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
}
.metric-card .label { color: #718096; font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card .value { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }
.metric-card .sub   { color: #4a5568; font-size: 0.72rem; margin-top: 0.1rem; }

/* ── Result cards ── */
.result-existing {
    background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 100%);
    border: 1px solid #40916c;
    border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 8px 24px rgba(64, 145, 108, 0.25);
}
.result-attrited {
    background: linear-gradient(135deg, #4a0e0e 0%, #7b2d2d 100%);
    border: 1px solid #e63946;
    border-radius: 16px; padding: 2rem; text-align: center;
    box-shadow: 0 8px 24px rgba(230, 57, 70, 0.25);
}
.result-label { font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.1em; color: rgba(255,255,255,0.7); }
.result-title { font-size: 1.9rem; font-weight: 800; color: #fff; margin: 0.5rem 0; }
.result-conf  { font-size: 0.9rem; color: rgba(255,255,255,0.75); }

/* ── Section headers ── */
.section-hdr {
    color: #a0aec0; font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
    padding: 0 0 0.6rem; border-bottom: 1px solid #2a2d3e;
    margin-bottom: 0.8rem;
}

/* ── Probability bar ── */
.prob-bar-wrap { background: #1a1d2e; border-radius: 50px; overflow: hidden;
    height: 10px; margin: 0.6rem 0 1.4rem; }
.prob-bar-fill { height: 100%; border-radius: 50px;
    background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.6s ease; }

/* ── Input widget tweaks ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background: #1a1d2e !important;
    border-color: #2a2d3e !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div > div { background: #667eea !important; }

/* ── Tab ── */
.stTabs [data-baseweb="tab-list"]  { background: #1a1d2e; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"]       { background: transparent; color: #718096;
    border-radius: 8px; font-size: 0.85rem; font-weight: 500; }
.stTabs [aria-selected="true"]     { background: #667eea !important; color: #fff !important; }

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; padding: 0.6rem 2rem !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    width: 100%;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.55) !important;
}

/* ── Tip box ── */
.tip-box {
    background: #1a1d2e; border-left: 3px solid #667eea;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    color: #a0aec0; font-size: 0.8rem; line-height: 1.5;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "naive_bayes.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    st.error(f"Could not load model: {e}")
    st.stop()

# ── Feature names & categories ─────────────────────────────────────────────────
FEATURE_NAMES = list(model.feature_names_in_)

GENDER_OPTIONS         = ["F", "M"]
EDUCATION_OPTIONS      = ["College", "Doctorate", "Graduate", "High School",
                           "Post-Graduate", "Uneducated", "Unknown"]
MARITAL_OPTIONS        = ["Divorced", "Married", "Single", "Unknown"]
INCOME_OPTIONS         = ["$120K +", "$40K - $60K", "$60K - $80K",
                           "$80K - $120K", "Less than $40K", "Unknown"]
CARD_OPTIONS           = ["Blue", "Gold", "Platinum", "Silver"]

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="color:#667eea;font-weight:700;font-size:1.05rem;'
                'margin-bottom:0.2rem;">⚙️ Customer Profile</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#718096;font-size:0.75rem;margin-bottom:1.2rem;">'
                'Fill in the customer details to predict churn risk.</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">📋 Demographics</div>', unsafe_allow_html=True)
    customer_age       = st.slider("Customer Age", 18, 80, 42)
    gender             = st.selectbox("Gender", GENDER_OPTIONS)
    dependent_count    = st.slider("Dependents", 0, 5, 2)
    education_level    = st.selectbox("Education Level", EDUCATION_OPTIONS, index=2)
    marital_status     = st.selectbox("Marital Status", MARITAL_OPTIONS, index=1)
    income_category    = st.selectbox("Income Category", INCOME_OPTIONS, index=4)

    st.markdown('<br><div class="section-hdr">💳 Card & Account</div>', unsafe_allow_html=True)
    card_category            = st.selectbox("Card Category", CARD_OPTIONS, index=0)
    months_on_book           = st.slider("Months on Book", 12, 60, 36)
    total_relationship_count = st.slider("Total Products Held", 1, 6, 3)
    credit_limit             = st.number_input("Credit Limit ($)", 1000, 35000, 4000, step=500)
    total_revolving_bal      = st.number_input("Total Revolving Balance ($)", 0, 3000, 800, step=100)
    avg_open_to_buy          = st.number_input("Avg Open To Buy ($)", 0, 35000, 3200, step=100)
    avg_utilization_ratio    = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.27, step=0.01)

    st.markdown('<br><div class="section-hdr">📊 Activity (12 Months)</div>', unsafe_allow_html=True)
    months_inactive          = st.slider("Months Inactive", 0, 6, 2)
    contacts_count           = st.slider("Contacts Count", 0, 6, 2)
    total_trans_amt          = st.number_input("Total Transaction Amt ($)", 500, 20000, 4500, step=100)
    total_trans_ct           = st.slider("Total Transaction Count", 10, 140, 55)
    total_amt_chng_q4_q1     = st.number_input("Amt Change Q4→Q1", 0.0, 4.0, 0.75, step=0.01, format="%.2f")
    total_ct_chng_q4_q1      = st.number_input("Count Change Q4→Q1", 0.0, 4.0, 0.70, step=0.01, format="%.2f")

    st.markdown('<br>')
    predict_btn = st.button("🔍  Predict Attrition Risk", use_container_width=True)

# ── Main layout ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>💳 Customer Attrition Predictor</h1>
  <p>Gaussian Naïve Bayes · 40 features · Binary classification</p>
</div>
""", unsafe_allow_html=True)

# Stats row
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.markdown('<div class="metric-card"><div class="label">Model</div>'
                '<div class="value" style="font-size:1.1rem">Gaussian NB</div>'
                '<div class="sub">sklearn 1.6.1</div></div>', unsafe_allow_html=True)
with col_s2:
    st.markdown('<div class="metric-card"><div class="label">Features</div>'
                '<div class="value">40</div><div class="sub">Numeric + One-hot</div></div>',
                unsafe_allow_html=True)
with col_s3:
    st.markdown('<div class="metric-card"><div class="label">Churn Rate</div>'
                '<div class="value">16.0%</div><div class="sub">1,300 / 8,101 train</div></div>',
                unsafe_allow_html=True)
with col_s4:
    st.markdown('<div class="metric-card"><div class="label">Classes</div>'
                '<div class="value">2</div><div class="sub">Existing / Attrited</div></div>',
                unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯  Prediction", "📊  Feature Explorer", "ℹ️  About Model"])

# ─────────────────────────── TAB 1 : PREDICTION ───────────────────────────────
with tab1:
    def build_feature_vector():
        """Build the 40-dim feature vector from sidebar inputs."""

        # NB classifier features (the weird long ones – these are pre-computed
        # probabilities embedded into the dataset; we use the class prior as proxy
        # since we don't have the original pipeline)
        nb1 = model.class_prior_[0]  # prob of class 0 – reasonable proxy
        nb2 = model.class_prior_[1]

        vec = {
            "CLIENTNUM":                          700000000,   # placeholder
            "Customer_Age":                       customer_age,
            "Dependent_count":                    dependent_count,
            "Months_on_book":                     months_on_book,
            "Total_Relationship_Count":           total_relationship_count,
            "Months_Inactive_12_mon":             months_inactive,
            "Contacts_Count_12_mon":              contacts_count,
            "Credit_Limit":                       credit_limit,
            "Total_Revolving_Bal":                total_revolving_bal,
            "Avg_Open_To_Buy":                    avg_open_to_buy,
            "Total_Amt_Chng_Q4_Q1":               total_amt_chng_q4_q1,
            "Total_Trans_Amt":                    total_trans_amt,
            "Total_Trans_Ct":                     total_trans_ct,
            "Total_Ct_Chng_Q4_Q1":                total_ct_chng_q4_q1,
            "Avg_Utilization_Ratio":              avg_utilization_ratio,
            # NB classifier embedded features
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1": nb1,
            "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2": nb2,
            # Gender
            "Gender_F": 1 if gender == "F" else 0,
            "Gender_M": 1 if gender == "M" else 0,
            # Education
            "Education_Level_College":         1 if education_level == "College"       else 0,
            "Education_Level_Doctorate":       1 if education_level == "Doctorate"     else 0,
            "Education_Level_Graduate":        1 if education_level == "Graduate"      else 0,
            "Education_Level_High School":     1 if education_level == "High School"   else 0,
            "Education_Level_Post-Graduate":   1 if education_level == "Post-Graduate" else 0,
            "Education_Level_Uneducated":      1 if education_level == "Uneducated"    else 0,
            "Education_Level_Unknown":         1 if education_level == "Unknown"       else 0,
            # Marital status
            "Marital_Status_Divorced": 1 if marital_status == "Divorced" else 0,
            "Marital_Status_Married":  1 if marital_status == "Married"  else 0,
            "Marital_Status_Single":   1 if marital_status == "Single"   else 0,
            "Marital_Status_Unknown":  1 if marital_status == "Unknown"  else 0,
            # Income
            "Income_Category_$120K +":          1 if income_category == "$120K +"           else 0,
            "Income_Category_$40K - $60K":      1 if income_category == "$40K - $60K"       else 0,
            "Income_Category_$60K - $80K":      1 if income_category == "$60K - $80K"       else 0,
            "Income_Category_$80K - $120K":     1 if income_category == "$80K - $120K"      else 0,
            "Income_Category_Less than $40K":   1 if income_category == "Less than $40K"    else 0,
            "Income_Category_Unknown":          1 if income_category == "Unknown"            else 0,
            # Card
            "Card_Category_Blue":     1 if card_category == "Blue"     else 0,
            "Card_Category_Gold":     1 if card_category == "Gold"     else 0,
            "Card_Category_Platinum": 1 if card_category == "Platinum" else 0,
            "Card_Category_Silver":   1 if card_category == "Silver"   else 0,
        }
        return np.array([vec[f] for f in FEATURE_NAMES]).reshape(1, -1)

    if predict_btn:
        X = build_feature_vector()
        pred       = model.predict(X)[0]
        proba      = model.predict_proba(X)[0]
        conf_exist = proba[1] * 100
        conf_attr  = proba[0] * 100

        result_col, spacer, tips_col = st.columns([2, 0.2, 1.5])

        with result_col:
            if pred == 1:
                st.markdown(f"""
                <div class="result-existing">
                  <div class="result-label">Prediction Result</div>
                  <div class="result-title">✅ Existing Customer</div>
                  <div class="result-conf">Likely to stay with {conf_exist:.1f}% confidence</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-attrited">
                  <div class="result-label">Prediction Result</div>
                  <div class="result-title">⚠️ Attrition Risk</div>
                  <div class="result-conf">Churn probability: {conf_attr:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p style="color:#718096;font-size:0.8rem;margin-bottom:0.3rem;">'
                        'Probability of being an <b style="color:#40916c">Existing Customer</b></p>',
                        unsafe_allow_html=True)
            bar_w = int(conf_exist)
            st.markdown(f'<div class="prob-bar-wrap"><div class="prob-bar-fill" '
                        f'style="width:{bar_w}%;background:linear-gradient(90deg,#40916c,#74c69d)"></div></div>',
                        unsafe_allow_html=True)

            st.markdown('<p style="color:#718096;font-size:0.8rem;margin-bottom:0.3rem;">'
                        'Probability of <b style="color:#e63946">Attrition</b></p>',
                        unsafe_allow_html=True)
            bar_w2 = int(conf_attr)
            st.markdown(f'<div class="prob-bar-wrap"><div class="prob-bar-fill" '
                        f'style="width:{bar_w2}%;background:linear-gradient(90deg,#c9184a,#ff4d6d)"></div></div>',
                        unsafe_allow_html=True)

            # Probability table
            prob_df = pd.DataFrame({
                "Class": ["Attrited Customer", "Existing Customer"],
                "Probability": [f"{conf_attr:.2f}%", f"{conf_exist:.2f}%"],
                "Raw Score": [f"{proba[0]:.4f}", f"{proba[1]:.4f}"],
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

        with tips_col:
            st.markdown('<div class="section-hdr">🔑 Key Risk Signals</div>',
                        unsafe_allow_html=True)

            risk_factors = []
            if months_inactive >= 3:
                risk_factors.append(("🔴", f"High inactivity ({months_inactive} months)"))
            if total_trans_ct < 40:
                risk_factors.append(("🔴", f"Low transaction count ({total_trans_ct})"))
            if contacts_count >= 4:
                risk_factors.append(("🟡", f"High contact frequency ({contacts_count})"))
            if total_relationship_count <= 2:
                risk_factors.append(("🟡", f"Few products held ({total_relationship_count})"))
            if avg_utilization_ratio < 0.1:
                risk_factors.append(("🟡", "Very low card utilization"))
            if total_trans_amt < 2000:
                risk_factors.append(("🔴", f"Low spend (${total_trans_amt:,})"))
            if total_ct_chng_q4_q1 < 0.5:
                risk_factors.append(("🔴", "Transaction count declining"))

            positive = []
            if total_trans_ct > 70:
                positive.append(("🟢", f"High transaction activity ({total_trans_ct})"))
            if total_relationship_count >= 4:
                positive.append(("🟢", f"Multiple products ({total_relationship_count})"))
            if months_inactive <= 1:
                positive.append(("🟢", "Recently active"))

            if risk_factors:
                st.markdown("**Risks detected:**")
                for icon, msg in risk_factors:
                    st.markdown(f'<p style="color:#e2e8f0;font-size:0.82rem;margin:0.2rem 0">'
                                f'{icon} {msg}</p>', unsafe_allow_html=True)

            if positive:
                st.markdown("<br>**Positive signals:**")
                for icon, msg in positive:
                    st.markdown(f'<p style="color:#e2e8f0;font-size:0.82rem;margin:0.2rem 0">'
                                f'{icon} {msg}</p>', unsafe_allow_html=True)

            if not risk_factors and not positive:
                st.markdown('<p style="color:#a0aec0;font-size:0.82rem">No strong signals detected.</p>',
                            unsafe_allow_html=True)

            st.markdown("""
            <div class="tip-box">
            💡 <b>Key attrition drivers:</b> Low transaction count & amount, 
            long inactivity periods, and high contact frequency are the 
            strongest predictors of churn in this dataset.
            </div>""", unsafe_allow_html=True)

    else:
        # Idle state
        st.markdown("""
        <div style="background:#1a1d2e;border:1px dashed #2a2d3e;border-radius:16px;
             padding:3rem;text-align:center;margin-top:1rem;">
          <div style="font-size:3rem">💳</div>
          <div style="color:#a0aec0;font-size:1rem;margin-top:0.8rem;">
            Configure the customer profile in the sidebar,<br>then click <b style="color:#667eea">Predict Attrition Risk</b>.
          </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────── TAB 2 : FEATURE EXPLORER ─────────────────────────────
with tab2:
    st.markdown("### 📊 Model Feature Statistics")
    st.markdown('<p style="color:#718096;font-size:0.85rem">Class-conditional means learned by '
                'the Gaussian Naïve Bayes model for each feature.</p>', unsafe_allow_html=True)

    means_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Mean (Attrited)":  np.round(model.theta_[0], 4),
        "Mean (Existing)":  np.round(model.theta_[1], 4),
        "Δ Difference":     np.round(np.abs(model.theta_[1] - model.theta_[0]), 4),
    }).sort_values("Δ Difference", ascending=False)

    top_n = st.slider("Show top N most discriminative features", 5, 40, 15)
    st.dataframe(means_df.head(top_n), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚖️ Class Priors")
    prior_col1, prior_col2 = st.columns(2)
    with prior_col1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">P(Attrited)</div>
          <div class="value">{model.class_prior_[0]:.4f}</div>
          <div class="sub">~{model.class_prior_[0]*100:.1f}% of training data</div>
        </div>""", unsafe_allow_html=True)
    with prior_col2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">P(Existing)</div>
          <div class="value">{model.class_prior_[1]:.4f}</div>
          <div class="sub">~{model.class_prior_[1]*100:.1f}% of training data</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────── TAB 3 : ABOUT ────────────────────────────────────
with tab3:
    st.markdown("### ℹ️ About This Model")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#1a1d2e;border-radius:12px;padding:1.4rem;border:1px solid #2a2d3e">
        <p style="color:#667eea;font-weight:700;margin-bottom:0.8rem">🧠 Algorithm</p>
        <p style="color:#a0aec0;font-size:0.85rem;line-height:1.7">
        <b style="color:#e2e8f0">Gaussian Naïve Bayes</b> assumes each feature is 
        normally distributed within each class and that all features are conditionally 
        independent given the class label. It estimates per-class mean (θ) and variance 
        (σ²) from training data, then applies Bayes' theorem at prediction time.
        </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:#1a1d2e;border-radius:12px;padding:1.4rem;border:1px solid #2a2d3e">
        <p style="color:#667eea;font-weight:700;margin-bottom:0.8rem">📦 Dataset</p>
        <p style="color:#a0aec0;font-size:0.85rem;line-height:1.7">
        Trained on credit card customer data with <b style="color:#e2e8f0">8,101 customers</b> — 
        1,300 attrited (~16%) and 6,801 existing (~84%). Features span demographics, 
        card behaviour, transaction patterns and one-hot encoded categorical variables.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#1a1d2e;border-radius:12px;padding:1.4rem;border:1px solid #2a2d3e">
    <p style="color:#667eea;font-weight:700;margin-bottom:0.8rem">📋 All Features (40)</p>
    </div>
    """, unsafe_allow_html=True)

    feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
    cols = [feat_col1, feat_col2, feat_col3, feat_col4]
    chunk = len(FEATURE_NAMES) // 4 + 1
    for i, col in enumerate(cols):
        with col:
            for feat in FEATURE_NAMES[i*chunk:(i+1)*chunk]:
                st.markdown(f'<p style="color:#a0aec0;font-size:0.75rem;margin:0.15rem 0">'
                            f'• {feat}</p>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 0.5rem;color:#4a5568;font-size:0.75rem">
  Built with Streamlit · Gaussian Naïve Bayes · sklearn
</div>
""", unsafe_allow_html=True)
