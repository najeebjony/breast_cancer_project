"""
================================================================================
Breast Cancer Clinical Prognosis & Analytics Web Application
Developed by Najeeb Ullah | Machine Learning Engineer
================================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Prognosis & Analytics | Developed by Najeeb Ullah",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #e0e6ed;
        margin: 6px 0 0 0;
        font-size: 1.05rem;
    }
    
    /* Branding Badge */
    .branding-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.92rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Metric Card Styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3c72;
    }
    .metric-lbl {
        color: #64748b;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    
    /* Result Box */
    .result-box-favorable {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #4caf50;
        border-radius: 12px;
        padding: 22px;
        margin-top: 15px;
    }
    .result-box-caution {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border: 2px solid #ff9800;
        border-radius: 12px;
        padding: 22px;
        margin-top: 15px;
    }
    .result-box-highrisk {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border: 2px solid #f44336;
        border-radius: 12px;
        padding: 22px;
        margin-top: 15px;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 24px;
        margin-top: 50px;
        border-top: 1px solid #e2e8f0;
        color: #64748b;
        font-size: 0.95rem;
    }
    .custom-footer strong {
        color: #1e3c72;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA & MODEL ASSET LOADERS
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    data_path = os.path.join(os.path.dirname(__file__), 'cleaned_breast_cancer.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        return df
    return None

@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), 'best_breast_cancer_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_metadata():
    meta_path = os.path.join(os.path.dirname(__file__), 'model_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

df = load_dataset()
pipeline = load_trained_model()
metadata = load_metadata()

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & BRANDING
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/breast-cancer-ribbon.png", width=70)
    st.title("OncoAnalytics")
    st.markdown("**Clinical Decision Support System**")
    
    # Mandatory Developer Attribution in Sidebar
    st.markdown(
        """
        <div style="background-color: #f1f5f9; padding: 12px; border-radius: 8px; border-left: 4px solid #1e3c72; margin-bottom: 20px;">
            <span style="font-size: 0.8rem; color: #64748b; text-transform: uppercase;">Developed By:</span><br>
            <strong style="color: #1e3c72; font-size: 1.02rem;">Najeeb Ullah</strong><br>
            <span style="font-size: 0.85rem; color: #475569;">Machine Learning Engineer</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("Navigation")
    app_mode = st.radio(
        "Select Workspace View:",
        ["Breast Cancer Predictor", "Deep Dashboard / Analytics"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Clinical Guidelines")
    st.info(
        "This tool utilizes validated machine learning algorithms trained on clean clinical-pathological features. "
        "Intended for informational & research decision-support only."
    )
    
    if metadata and 'winning_model' in metadata:
        st.markdown(f"**Active Model:** `{metadata['winning_model']}`")
        if 'results' in metadata and len(metadata['results']) > 0:
            top_f1 = metadata['results'][0].get('F1-Score', 'N/A')
            st.markdown(f"**Holdout F1-Score:** `{top_f1}`")

# -----------------------------------------------------------------------------
# 4. VIEW 1: BREAST CANCER PREDICTOR
# -----------------------------------------------------------------------------
if app_mode == "Breast Cancer Predictor":
    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>🎗️ Patient Survival Prognosis Predictor</h1>
            <p>Enter patient clinical and histological characteristics to evaluate long-term survival probability</p>
            <div class="branding-badge">Developed by Najeeb Ullah | Machine Learning Engineer</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if pipeline is None:
        st.error("Model asset `best_breast_cancer_model.pkl` not found. Please run `python train_model.py` to train and serialize the model.")
    else:
        st.markdown("### 📋 Patient Clinical Attributes")
        st.write("Adjust the parameters below based on standard clinical examinations and biopsy reports.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 1. Patient Demographics & Stage")
            age = st.slider(
                "Age at Diagnosis (Years)",
                min_value=20, max_value=95, value=55, step=1,
                help="Patient's age at initial cancer diagnosis"
            )
            menopausal = st.selectbox(
                "Menopausal State",
                options=["Post", "Pre"],
                index=0,
                help="Inferred menopausal state based on age and endocrine status"
            )
            tumor_stage = st.selectbox(
                "Tumor Stage (TNM System)",
                options=[1.0, 2.0, 3.0, 4.0],
                index=1,
                format_func=lambda x: f"Stage {int(x)}"
            )
            laterality = st.selectbox(
                "Primary Tumor Laterality",
                options=["Left", "Right"],
                index=0
            )

        with col2:
            st.markdown("#### 2. Tumor Pathology & Burden")
            tumor_size = st.slider(
                "Tumor Size (mm)",
                min_value=1, max_value=150, value=25, step=1,
                help="Maximum diameter of the primary tumor in millimeters"
            )
            lymph_nodes = st.slider(
                "Positive Lymph Nodes Examined",
                min_value=0, max_value=35, value=1, step=1,
                help="Number of regional axillary lymph nodes identified with metastasis"
            )
            histologic_grade = st.selectbox(
                "Neoplasm Histologic Grade",
                options=[1.0, 2.0, 3.0],
                index=1,
                format_func=lambda x: f"Grade {int(x)} ({'Well' if x==1 else 'Moderately' if x==2 else 'Poorly'} Differentiated)"
            )
            cellularity = st.selectbox(
                "Tumor Cellularity",
                options=["High", "Moderate", "Low"],
                index=0,
                help="Percentage of tumor area occupied by invasive neoplastic cells"
            )

        with col3:
            st.markdown("#### 3. Receptor Status & Treatments")
            er_status = st.selectbox(
                "ER Status (Estrogen Receptor)",
                options=["Positive", "Negative"],
                index=0,
                help="Presence of estrogen receptors on breast cancer cells"
            )
            pr_status = st.selectbox(
                "PR Status (Progesterone Receptor)",
                options=["Positive", "Negative"],
                index=0,
                help="Presence of progesterone receptors"
            )
            her2_status = st.selectbox(
                "HER2 Status (Growth Factor Receptor)",
                options=["Negative", "Positive"],
                index=0,
                help="Overexpression of human epidermal growth factor receptor 2"
            )
            surgery = st.selectbox(
                "Type of Breast Surgery",
                options=["Mastectomy", "Breast Conserving"],
                index=0
            )
            
            c_chemo, c_radio, c_hormone = st.columns(3)
            with c_chemo:
                chemo = st.selectbox("Chemo", options=["No", "Yes"], index=0)
            with c_radio:
                radio = st.selectbox("Radio", options=["Yes", "No"], index=0)
            with c_hormone:
                hormone = st.selectbox("Hormone", options=["Yes", "No"], index=0)

        # Construct input dataframe
        input_data = pd.DataFrame([{
            'Age at Diagnosis': float(age),
            'Tumor Size': float(tumor_size),
            'Tumor Stage': float(tumor_stage),
            'Lymph nodes examined positive': float(lymph_nodes),
            'Neoplasm Histologic Grade': float(histologic_grade),
            'Type of Breast Surgery': surgery,
            'Chemotherapy': chemo,
            'Radio Therapy': radio,
            'Hormone Therapy': hormone,
            'ER Status': er_status,
            'PR Status': pr_status,
            'HER2 Status': her2_status,
            'Inferred Menopausal State': menopausal,
            'Primary Tumor Laterality': laterality,
            'Cellularity': cellularity
        }])

        st.markdown("---")
        predict_btn = st.button("🔍 Generate Comprehensive Survival Prognosis", use_container_width=True, type="primary")

        if predict_btn or True:  # Real-time reactive inference
            try:
                prediction = pipeline.predict(input_data)[0]
                probabilities = pipeline.predict_proba(input_data)[0]
                
                prob_deceased = probabilities[0] * 100
                prob_survived = probabilities[1] * 100
                
                st.subheader("📊 Clinical Prognosis & Outcome Assessment")
                
                res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])
                
                with res_col1:
                    if prob_survived >= 60.0:
                        st.markdown(
                            f"""
                            <div class="result-box-favorable">
                                <h3 style="color:#2e7d32; margin:0;">🌟 Favorable Prognosis</h3>
                                <p style="font-size:1.1rem; margin:8px 0;"><strong>Predicted Class:</strong> Long-Term Survivor / Living</p>
                                <h1 style="color:#1b5e20; margin:0; font-size:2.6rem;">{prob_survived:.1f}%</h1>
                                <p style="margin:4px 0 0 0; color:#388e3c;">Estimated Probability of Long-Term Survival</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    elif prob_survived >= 40.0:
                        st.markdown(
                            f"""
                            <div class="result-box-caution">
                                <h3 style="color:#e65100; margin:0;">⚠️ Intermediate / Guarded Prognosis</h3>
                                <p style="font-size:1.1rem; margin:8px 0;"><strong>Predicted Outcome:</strong> Moderate Risk Profile</p>
                                <h1 style="color:#bf360c; margin:0; font-size:2.6rem;">{prob_survived:.1f}%</h1>
                                <p style="margin:4px 0 0 0; color:#d84315;">Estimated Probability of Long-Term Survival</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="result-box-highrisk">
                                <h3 style="color:#c62828; margin:0;">🚨 Elevated Risk Prognosis</h3>
                                <p style="font-size:1.1rem; margin:8px 0;"><strong>Predicted Outcome:</strong> High Clinical Risk Profile</p>
                                <h1 style="color:#b71c1c; margin:0; font-size:2.6rem;">{prob_deceased:.1f}%</h1>
                                <p style="margin:4px 0 0 0; color:#c62828;">Estimated Mortality Risk / Needs Close Surveillance</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with res_col2:
                    st.markdown("#### Outcome Probability Matrix")
                    st.metric(
                        label="Probability of Long-Term Survival",
                        value=f"{prob_survived:.1f}%",
                        delta=f"{prob_survived - 33.4:.1f}% vs. Population Baseline"
                    )
                    st.metric(
                        label="Probability of Mortality / Adverse Event",
                        value=f"{prob_deceased:.1f}%",
                        delta=f"{prob_deceased - 66.6:.1f}% vs. Population Baseline",
                        delta_color="inverse"
                    )

                with res_col3:
                    st.markdown("#### Key Factor Assessment")
                    # Layman explanation
                    factors = []
                    if age >= 70:
                        factors.append("⚠️ Advanced Age (≥70 years) significantly elevates baseline risk.")
                    elif age < 50:
                        factors.append("✅ Younger age group (<50) offers favorable physiological reserve.")
                    
                    if lymph_nodes == 0:
                        factors.append("✅ Node-negative status (0 positive nodes) is a strong positive indicator.")
                    elif lymph_nodes >= 4:
                        factors.append("⚠️ Higher nodal involvement (≥4 nodes) indicates lymphatic spread.")
                    
                    if tumor_size <= 20:
                        factors.append("✅ Small primary tumor diameter (≤20mm, T1).")
                    elif tumor_size > 50:
                        factors.append("⚠️ Bulky tumor volume (>50mm, T3/T4).")
                        
                    if er_status == "Positive" and pr_status == "Positive":
                        factors.append("✅ Positive hormone receptor status (ER+/PR+) enables effective targeted endocrine therapies.")
                    elif er_status == "Negative" and pr_status == "Negative" and her2_status == "Negative":
                        factors.append("⚠️ Triple-negative receptor phenotype typically exhibits more aggressive course.")
                    
                    if len(factors) == 0:
                        factors.append("ℹ️ Balanced clinical factor distribution across all parameters.")
                        
                    for f in factors:
                        st.write(f)
                        
            except Exception as e:
                st.error(f"Prediction error encountered: {e}")

# -----------------------------------------------------------------------------
# 5. VIEW 2: DEEP DASHBOARD / ANALYTICS
# -----------------------------------------------------------------------------
else:
    # Header
    st.markdown(
        """
        <div class="main-header">
            <h1>📈 Deep Clinical Analytics & Exploratory Data Dashboard</h1>
            <p>Comprehensive population-level clinical insights, survival distributions, and correlation patterns</p>
            <div class="branding-badge">Developed by Najeeb Ullah | Machine Learning Engineer</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if df is None:
        st.error("Dataset `cleaned_breast_cancer.csv` not found in workspace.")
    else:
        # Top KPI Metrics Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        total_patients = len(df)
        overall_survival_rate = df['Overall Survival Status'].mean() * 100
        relapse_rate = (df['Relapse Free Status'] == 'Recurred').mean() * 100 if 'Relapse Free Status' in df.columns else 0
        avg_age = df['Age at Diagnosis'].mean()
        
        with kpi1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">{total_patients:,}</div>
                    <div class="metric-lbl">Total Clinical Cohort</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">{overall_survival_rate:.1f}%</div>
                    <div class="metric-lbl">Overall Survival Rate</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">{relapse_rate:.1f}%</div>
                    <div class="metric-lbl">Disease Recurrence Rate</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">{avg_age:.1f} yrs</div>
                    <div class="metric-lbl">Average Age at Diagnosis</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Visual Analytics Section 1: Demographics & Staging
        st.subheader("1. Survival Distribution by Age and Anatomical Stage")
        col_c1, col_c2 = st.columns(2)
        
        # Create Age Groups
        df_dash = df.copy()
        df_dash['Age_Group'] = pd.cut(
            df_dash['Age at Diagnosis'],
            bins=[0, 45, 60, 75, 120],
            labels=['Young (<45)', 'Middle (45-60)', 'Senior (60-75)', 'Elderly (>75)']
        )
        
        with col_c1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            age_survival = df_dash.groupby('Age_Group', observed=False)['Overall Survival Status'].mean() * 100
            bars = ax.bar(age_survival.index, age_survival.values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black', alpha=0.85)
            ax.set_ylabel("Survival Rate (%)", fontsize=10)
            ax.set_title("Survival Rate across Age Groups", fontsize=12, fontweight='bold', pad=10)
            ax.set_ylim(0, 60)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
            st.pyplot(fig)
            plt.close()

        with col_c2:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            stage_clean = df_dash['Tumor Stage'].round().clip(0, 4).astype(int)
            stage_survival = df_dash.groupby(stage_clean)['Overall Survival Status'].mean() * 100
            bars = ax.bar([f"Stage {s}" for s in stage_survival.index], stage_survival.values, color=sns.color_palette("Reds_r", len(stage_survival)), edgecolor='black', alpha=0.85)
            ax.set_ylabel("Survival Rate (%)", fontsize=10)
            ax.set_title("Survival Rate by Clinical Tumor Stage", fontsize=12, fontweight='bold', pad=10)
            ax.set_ylim(0, 60)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
            st.pyplot(fig)
            plt.close()

        # Visual Analytics Section 2: Receptors & Lymph Nodes
        st.markdown("---")
        st.subheader("2. Biomarker Receptor Profiling & Lymphatic Spread")
        col_c3, col_c4 = st.columns(2)
        
        with col_c3:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            receptors = ['ER Status', 'PR Status', 'HER2 Status']
            rates_pos = [df_dash[df_dash[r] == 'Positive']['Overall Survival Status'].mean() * 100 for r in receptors]
            rates_neg = [df_dash[df_dash[r] == 'Negative']['Overall Survival Status'].mean() * 100 for r in receptors]
            
            x = np.arange(len(receptors))
            width = 0.35
            ax.bar(x - width/2, rates_pos, width, label='Positive (+)', color='#2ecc71', alpha=0.85, edgecolor='black')
            ax.bar(x + width/2, rates_neg, width, label='Negative (-)', color='#e74c3c', alpha=0.85, edgecolor='black')
            ax.set_xticks(x)
            ax.set_xticklabels(['Estrogen (ER)', 'Progesterone (PR)', 'HER2 Neu'])
            ax.set_ylabel("Survival Rate (%)", fontsize=10)
            ax.set_title("Survival Rate: Receptor Positive vs. Negative", fontsize=12, fontweight='bold', pad=10)
            ax.set_ylim(0, 55)
            ax.legend()
            st.pyplot(fig)
            plt.close()

        with col_c4:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.scatterplot(
                data=df_dash,
                x='Tumor Size',
                y='Lymph nodes examined positive',
                hue='Overall Survival Status',
                palette={0: '#e74c3c', 1: '#2ecc71'},
                alpha=0.6,
                ax=ax
            )
            ax.set_title("Tumor Size vs. Positive Lymph Nodes Burden", fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel("Tumor Size (mm)")
            ax.set_ylabel("Positive Lymph Nodes")
            ax.set_xlim(0, 120)
            ax.set_ylim(-2, 35)
            ax.legend(title='Outcome', labels=['Deceased (0)', 'Survived (1)'])
            st.pyplot(fig)
            plt.close()

        # Visual Analytics Section 3: Feature Correlation Heatmap
        st.markdown("---")
        st.subheader("3. Clinical Feature Correlation Matrix")
        
        num_cols_for_corr = [
            'Age at Diagnosis', 'Tumor Size', 'Tumor Stage',
            'Lymph nodes examined positive', 'Neoplasm Histologic Grade', 'Overall Survival Status'
        ]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        corr = df_dash[num_cols_for_corr].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.8})
        ax.set_title("Pearson Correlation Coefficients among Clinical Indicators", fontsize=12, fontweight='bold', pad=10)
        st.pyplot(fig)
        plt.close()

        # Model Benchmark Comparison Table from Metadata
        if metadata and 'results' in metadata:
            st.markdown("---")
            st.subheader("4. Machine Learning Model Benchmark Comparison")
            st.write("Results of 5-Fold Stratified Cross-Validation with `GridSearchCV` on the Holdout Test Cohort (n=502):")
            
            res_table = pd.DataFrame(metadata['results'])
            st.dataframe(
                res_table[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Best_CV_F1']],
                use_container_width=True
            )

# -----------------------------------------------------------------------------
# 6. MANDATORY PROMINENT FOOTER BRANDING
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="custom-footer">
        <p style="margin: 0; font-size: 1.05rem;">
            🎗️ <strong>Breast Cancer Machine Learning Prognosis System</strong>
        </p>
        <p style="margin: 6px 0; font-size: 1rem; font-weight: 600; color: #1e3c72;">
            Developed by Najeeb Ullah | Machine Learning Engineer
        </p>
        <p style="margin: 0; font-size: 0.82rem; color: #94a3b8;">
            Trained with Stratified 5-Fold GridSearchCV Hyperparameter Tuning on Clean Clinical Records.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
