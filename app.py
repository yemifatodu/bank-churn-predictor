import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ | Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    /* ── Reset & Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F0F4FA;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1628 0%, #0D1F3C 60%, #102444 100%);
        border-right: 1px solid #C9A84C33;
    }
    [data-testid="stSidebar"] * {
        color: #CBD5E0 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #C9A84C !important;
    }
    [data-testid="stSidebar"] a {
        color: #C9A84C !important;
        text-decoration: none;
    }
    [data-testid="stSidebar"] a:hover {
        color: #E8C96A !important;
        text-decoration: underline;
    }
    [data-testid="stSidebar"] hr {
        border-color: #C9A84C33 !important;
    }

    /* ── Main background ── */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #F0F4FA;
    }

    /* ── Top header bar ── */
    .top-bar {
        background: linear-gradient(135deg, #0A1628 0%, #0D1F3C 100%);
        padding: 24px 36px;
        border-radius: 12px;
        margin-bottom: 28px;
        border-bottom: 3px solid #C9A84C;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .top-bar-title {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: 0.5px;
        margin: 0;
    }
    .top-bar-title span {
        color: #C9A84C;
    }
    .top-bar-subtitle {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 4px;
        font-weight: 400;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .top-bar-badge {
        background: #C9A84C22;
        border: 1px solid #C9A84C66;
        color: #C9A84C;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── Section label ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 12px;
        padding-left: 2px;
    }

    /* ── Cards ── */
    .card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 24px 28px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 4px rgba(10,22,40,0.06);
        margin-bottom: 20px;
    }
    .card-title {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: #0A1628;
        border-left: 3px solid #C9A84C;
        padding-left: 10px;
        margin-bottom: 20px;
    }

    /* ── KPI chips ── */
    .kpi-row {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }
    .kpi-chip {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 22px;
        flex: 1;
        min-width: 140px;
        box-shadow: 0 1px 3px rgba(10,22,40,0.05);
    }
    .kpi-chip-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0A1628;
        line-height: 1;
    }
    .kpi-chip-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #64748B;
        margin-top: 5px;
    }
    .kpi-chip-accent {
        width: 28px;
        height: 3px;
        background: #C9A84C;
        border-radius: 2px;
        margin-bottom: 10px;
    }

    /* ── Input labels ── */
    label, .stSlider label, .stSelectbox label, .stNumberInput label {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }

    /* ── Slider accent ── */
    [data-testid="stSlider"] > div > div > div {
        background: #C9A84C !important;
    }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0A1628 0%, #0D1F3C 100%) !important;
        color: #C9A84C !important;
        border: 1px solid #C9A84C !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        padding: 14px 0 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(10,22,40,0.2) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #C9A84C 0%, #E8C96A 100%) !important;
        color: #0A1628 !important;
        box-shadow: 0 6px 20px rgba(201,168,76,0.35) !important;
    }

    /* ── Result verdict cards ── */
    .verdict-high {
        background: linear-gradient(135deg, #FFF1F1 0%, #FEE2E2 100%);
        border: 1.5px solid #DC2626;
        border-left: 5px solid #DC2626;
        border-radius: 12px;
        padding: 24px 28px;
    }
    .verdict-low {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        border: 1.5px solid #16A34A;
        border-left: 5px solid #16A34A;
        border-radius: 12px;
        padding: 24px 28px;
    }
    .verdict-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .verdict-prob {
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1;
        margin: 10px 0 4px 0;
    }
    .verdict-sublabel {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        opacity: 0.7;
    }

    /* ── Risk meter bar ── */
    .risk-bar-wrap {
        background: #E2E8F0;
        border-radius: 6px;
        height: 10px;
        margin: 14px 0 6px 0;
        overflow: hidden;
    }
    .risk-bar-fill-high {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #F59E0B, #DC2626);
    }
    .risk-bar-fill-low {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #34D399, #16A34A);
    }

    /* ── Action plan ── */
    .action-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 18px 22px;
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .action-icon {
        font-size: 1.2rem;
        margin-top: 1px;
    }
    .action-text strong {
        font-size: 0.85rem;
        font-weight: 700;
        color: #0A1628;
        display: block;
        margin-bottom: 2px;
    }
    .action-text span {
        font-size: 0.78rem;
        color: #64748B;
    }

    /* ── Divider ── */
    .gold-divider {
        height: 1px;
        background: linear-gradient(90deg, #C9A84C44, #C9A84C, #C9A84C44);
        margin: 28px 0;
        border: none;
    }

    /* ── Success/error overrides ── */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
    }

    /* ── Metric overrides ── */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        color: #64748B !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #0A1628 !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model        = joblib.load('models/bank_churn_rf_model.pkl')
    preprocessor = joblib.load('models/bank_preprocessor.pkl')
    return model, preprocessor

try:
    model, preprocessor = load_models()
    model_ok = True
except Exception as e:
    model_ok = False
    model_error = str(e)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
        <div style="font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:700; color:#C9A84C;">
            ChurnIQ
        </div>
        <div style="font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; color:#64748B; margin-top:2px;">
            Risk Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📌 Platform")
    st.markdown("""
    <div style="font-size:0.82rem; color:#94A3B8; line-height:1.8;">
    ChurnIQ uses a trained <strong style="color:#C9A84C;">Random Forest</strong> ensemble 
    model to assess customer attrition risk from 10 behavioral and demographic signals.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 Model Card")
    st.markdown("""
    <div style="font-size:0.8rem; color:#94A3B8; line-height:2;">
    <b style="color:#CBD5E0;">Algorithm</b><br/>Random Forest Classifier<br/><br/>
    <b style="color:#CBD5E0;">Accuracy</b><br/>86.80%<br/><br/>
    <b style="color:#CBD5E0;">Features</b><br/>10 customer attributes<br/><br/>
    <b style="color:#CBD5E0;">Dataset</b><br/>Bank Customer Segmentation<br/>10,000 records
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🔗 Resources")
    st.markdown("""
    <div style="font-size:0.82rem; line-height:2.4;">
    <a href="https://github.com/yemifatodu/bank-churn-predictor" target="_blank">⟶ Source Code</a><br/>
    <a href="https://yemifatodu.online" target="_blank">⟶ Portfolio</a><br/>
    <a href="https://linkedin.com/in/yemifatodu" target="_blank">⟶ LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.7rem; color:#475569; line-height:1.6;">
    Built by <strong style="color:#C9A84C;">Yemi Fatodu</strong><br/>
    Data Scientist · BI Specialist<br/>Full-Stack Product Builder<br/><br/>
    <em style="color:#64748B;">Open to freelance & contract engagements</em>
    </div>
    """, unsafe_allow_html=True)

# ── Header bar ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div>
        <div class="top-bar-title">🏦 Churn<span>IQ</span></div>
        <div class="top-bar-subtitle">Customer Attrition Risk Intelligence Platform</div>
    </div>
    <div class="top-bar-badge">● Live Model</div>
</div>
""", unsafe_allow_html=True)

# ── Model status ──────────────────────────────────────────────────────────────
if not model_ok:
    st.error(f"❌ Model load error: {model_error}")
    st.info("Run `python train.py` inside your venv to regenerate model files.")
    st.stop()

# ── KPI strip ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-row">
    <div class="kpi-chip">
        <div class="kpi-chip-accent"></div>
        <div class="kpi-chip-value">86.80%</div>
        <div class="kpi-chip-label">Model Accuracy</div>
    </div>
    <div class="kpi-chip">
        <div class="kpi-chip-accent"></div>
        <div class="kpi-chip-value">10K</div>
        <div class="kpi-chip-label">Training Records</div>
    </div>
    <div class="kpi-chip">
        <div class="kpi-chip-accent"></div>
        <div class="kpi-chip-value">200</div>
        <div class="kpi-chip-label">Estimators</div>
    </div>
    <div class="kpi-chip">
        <div class="kpi-chip-accent"></div>
        <div class="kpi-chip-value">10</div>
        <div class="kpi-chip-label">Feature Signals</div>
    </div>
    <div class="kpi-chip">
        <div class="kpi-chip-accent"></div>
        <div class="kpi-chip-value">RF</div>
        <div class="kpi-chip-label">Algorithm</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Customer Profile Input</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card"><div class="card-title">Personal & Demographic Signals</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.slider('Credit Score', 300, 850, 650,
            help="300 = Poor · 850 = Excellent")
        age          = st.slider('Age', 18, 100, 40)
        geography    = st.selectbox('Country / Region', ['France', 'Spain', 'Germany'])

    with col2:
        gender       = st.selectbox('Gender', ['Male', 'Female'])
        tenure       = st.slider('Tenure (Years)', 0, 10, 5,
            help="Years as a bank customer")
        num_products = st.selectbox('Products Held', [1, 2, 3, 4])

    with col3:
        balance          = st.number_input('Account Balance ($)', 0, 500000, 50000, step=1000)
        estimated_salary = st.number_input('Estimated Salary ($)', 0, 500000, 100000, step=1000)
        has_cr_card      = st.selectbox('Credit Card Holder', [1, 0],
            format_func=lambda x: 'Yes' if x == 1 else 'No')
        is_active        = st.selectbox('Active Member', [1, 0],
            format_func=lambda x: 'Yes' if x == 1 else 'No')

    st.markdown('</div>', unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button('⟶  RUN RISK ASSESSMENT', type='primary', use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if predict_clicked:
    input_data = pd.DataFrame({
        'CreditScore':     [credit_score],
        'Geography':       [geography],
        'Gender':          [gender],
        'Age':             [age],
        'Tenure':          [tenure],
        'Balance':         [balance],
        'NumOfProducts':   [num_products],
        'HasCrCard':       [has_cr_card],
        'IsActiveMember':  [is_active],
        'EstimatedSalary': [estimated_salary]
    })

    try:
        input_processed = preprocessor.transform(input_data)
        prediction      = model.predict(input_processed)[0]
        probability     = model.predict_proba(input_processed)[0][1]
        confidence      = max(probability, 1 - probability)

        st.markdown('<hr class="gold-divider"/>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Risk Assessment Output</div>', unsafe_allow_html=True)

        # ── Verdict card ──
        res_col1, res_col2 = st.columns([1.4, 1])

        with res_col1:
            if prediction == 1:
                bar_pct = int(probability * 100)
                st.markdown(f"""
                <div class="verdict-high">
                    <div class="verdict-sublabel">⚠ Attrition Risk Verdict</div>
                    <div class="verdict-title" style="color:#DC2626;">HIGH RISK — LIKELY TO CHURN</div>
                    <div class="verdict-prob" style="color:#DC2626;">{probability:.1%}</div>
                    <div class="verdict-sublabel">Churn Probability Score</div>
                    <div class="risk-bar-wrap">
                        <div class="risk-bar-fill-high" style="width:{bar_pct}%;"></div>
                    </div>
                    <div style="font-size:0.72rem;color:#64748B;">Risk threshold: 50% · Model confidence: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                bar_pct = int((1 - probability) * 100)
                st.markdown(f"""
                <div class="verdict-low">
                    <div class="verdict-sublabel">✔ Attrition Risk Verdict</div>
                    <div class="verdict-title" style="color:#16A34A;">LOW RISK — LIKELY TO RETAIN</div>
                    <div class="verdict-prob" style="color:#16A34A;">{probability:.1%}</div>
                    <div class="verdict-sublabel">Churn Probability Score</div>
                    <div class="risk-bar-wrap">
                        <div class="risk-bar-fill-low" style="width:{bar_pct}%;"></div>
                    </div>
                    <div style="font-size:0.72rem;color:#64748B;">Risk threshold: 50% · Model confidence: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.metric("Churn Probability",  f"{probability:.2%}")
            st.metric("Model Confidence",   f"{confidence:.2%}")
            st.metric("Verdict",            "⚠ CHURN" if prediction == 1 else "✔ RETAIN")

        # ── Feature importance chart ──
        st.markdown('<hr class="gold-divider"/>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Model Explainability — Feature Drivers</div>', unsafe_allow_html=True)

        try:
            feature_names = (preprocessor.get_feature_names_out()
                             if hasattr(preprocessor, 'get_feature_names_out')
                             else input_data.columns.tolist())
            importance    = model.feature_importances_
            feat_df       = pd.DataFrame({
                'Feature':    feature_names[:len(importance)],
                'Importance': importance
            }).sort_values('Importance', ascending=True).tail(8)

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#FFFFFF')
            ax.set_facecolor('#F8FAFC')

            colors = ['#C9A84C' if i == len(feat_df) - 1 else '#0D1F3C'
                      for i in range(len(feat_df))]
            bars = ax.barh(feat_df['Feature'], feat_df['Importance'],
                           color=colors, height=0.55, edgecolor='none')

            ax.set_title('Top Feature Importance Drivers', fontsize=11,
                         fontweight='700', color='#0A1628', pad=14,
                         fontfamily='sans-serif')
            ax.set_xlabel('Importance Score', fontsize=9, color='#64748B')
            ax.tick_params(axis='y', labelsize=9, labelcolor='#334155')
            ax.tick_params(axis='x', labelsize=8, labelcolor='#94A3B8')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#E2E8F0')
            ax.spines['bottom'].set_color('#E2E8F0')
            ax.grid(axis='x', color='#E2E8F0', linewidth=0.8, linestyle='--')
            ax.set_axisbelow(True)

            for bar in bars:
                w = bar.get_width()
                ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                        f'{w:.3f}', va='center', fontsize=8, color='#334155')

            gold_patch = mpatches.Patch(color='#C9A84C', label='Top driver')
            navy_patch = mpatches.Patch(color='#0D1F3C', label='Other drivers')
            ax.legend(handles=[gold_patch, navy_patch], fontsize=8,
                      framealpha=0, loc='lower right')

            plt.tight_layout()
            st.pyplot(fig)
        except Exception:
            st.info("Feature importance chart unavailable for this model version.")

        # ── Action plan ──
        st.markdown('<hr class="gold-divider"/>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Recommended Action Plan</div>', unsafe_allow_html=True)

        if prediction == 1:
            actions = [
                ("🎯", "Activate Retention Campaign",
                 "Dispatch a personalised offer — fee waiver, interest rate bonus, or loyalty upgrade — within 48 hours."),
                ("📞", "Relationship Manager Outreach",
                 "Schedule a proactive call to understand pain points and reinforce relationship value."),
                ("💰", "Premium Tier Incentive",
                 "Offer a complimentary upgrade to premium banking tier for 3 months to re-engage."),
                ("📊", "Heightened Transaction Monitoring",
                 "Flag this account for daily activity review and early-warning alerts."),
                ("📧", "Personalised Digital Engagement",
                 "Trigger a targeted email sequence with relevant product recommendations."),
            ]
        else:
            actions = [
                ("✅", "Maintain Engagement Cadence",
                 "Continue current touchpoint strategy — do not over-communicate and risk fatigue."),
                ("📈", "Cross-Sell Opportunity",
                 "Profile this customer for additional product offerings (savings, investment, insurance)."),
                ("🎁", "Loyalty Recognition",
                 "Acknowledge tenure milestones with a personalised loyalty reward or recognition message."),
                ("📱", "Digital Channel Deepening",
                 "Encourage mobile banking adoption for higher stickiness and lower cost-to-serve."),
                ("📊", "Periodic Health Check",
                 "Schedule a quarterly account review to monitor for any sentiment shift."),
            ]

        for icon, title, desc in actions:
            st.markdown(f"""
            <div class="action-card">
                <div class="action-icon">{icon}</div>
                <div class="action-text">
                    <strong>{title}</strong>
                    <span>{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Input summary ──
        with st.expander("📋 View Full Input Summary"):
            st.dataframe(input_data.T.rename(columns={0: 'Value'}),
                         use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")