"""
Script to generate the comprehensive Jupyter Notebook: EDA_and_Feature_Engineering.ipynb
Developed by Najeeb Ullah | Machine Learning Engineer
"""

import json
import os

def build_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.2"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(code):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in code.strip().split("\n")]
        })

    # Header & Metadata
    add_md("""# Comprehensive Exploratory Data Analysis, Feature Engineering & Machine Learning Pipeline for Breast Cancer Survival Prognosis
### **Developed by Najeeb Ullah | Machine Learning Engineer**

---

## 1. Executive Summary & Clinical Context
Breast cancer prognosis is a complex medical challenge influenced by tumor biology, anatomical spread, patient demographics, and therapeutic interventions. In real-world clinical decision support, models built on dense genomic assays or complex clustered features are often impractical for frontline clinicians and confusing for patients.

**Objective of this Study:**
1. **Layman Feature Selection**: Isolate only intuitive clinical and pathological attributes that patients and doctors easily understand.
2. **Deep Exploratory Data Analysis**: Uncover clinical relationships, survival distributions, and feature correlations.
3. **Clinical Feature Engineering**: Construct meaningful composite metrics (Age groups, Tumor stage categories, Lymph node burden, Multi-therapy index).
4. **Machine Learning & Hyperparameter Tuning**: Benchmark 4 algorithms (**Random Forest**, **XGBoost**, **AdaBoost**, **Decision Tree**) using stratified 5-fold cross-validation and `GridSearchCV`.
5. **Model Evaluation & Deployment Serialization**: Select the optimal model based on balanced metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and serialize it for interactive web deployment.
""")

    # Imports
    add_code("""# Core Data Science & Visualization Libraries
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Ecosystem & XGBoost
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import joblib

# Visual aesthetics
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120
""")

    # Data Ingestion
    add_md("""---
## 2. Data Ingestion & Dataset Audit
We load the curated clinical dataset (`cleaned_breast_cancer.csv`) and examine its dimensions, structural integrity, and column representations.
""")

    add_code("""# Load dataset
DATA_PATH = 'cleaned_breast_cancer.csv'
df = pd.read_csv(DATA_PATH)

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing Values: {df.isnull().sum().sum()}")
df.head(5)
""")

    # Target Analysis
    add_md("""---
## 3. Target Variable Formulation & Distribution
In this clinical study, our target variable is **`Overall Survival Status`**:
- **`1` (Survived / Living)**: Patient survived during the observed study period.
- **`0` (Deceased / Did Not Survive)**: Patient did not survive.

We also examine **`Relapse Free Status`** (Recurred vs. Not Recurred) as a secondary outcome indicator.
""")

    add_code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall Survival Status Distribution
survival_counts = df['Overall Survival Status'].value_counts()
colors = ['#e74c3c', '#2ecc71']
axes[0].pie(survival_counts, labels=['Deceased (0)', 'Survived (1)'], autopct='%1.1f%%',
            startangle=140, colors=colors, explode=(0, 0.05), shadow=True)
axes[0].set_title('Overall Survival Status Distribution', fontsize=13, fontweight='bold')

# Relapse Free Status Distribution
if 'Relapse Free Status' in df.columns:
    sns.countplot(data=df, x='Relapse Free Status', ax=axes[1], palette='Set2')
    axes[1].set_title('Relapse Free Status Distribution', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Relapse Status')
    axes[1].set_ylabel('Patient Count')

plt.tight_layout()
plt.show()

print("Overall Survival Status Counts:")
print(df['Overall Survival Status'].value_counts())
print("\nSurvival Rate:", f"{df['Overall Survival Status'].mean() * 100:.2f}%")
""")

    # Feature Selection
    add_md("""---
## 4. Feature Selection for Laymen
To create an intuitive, practical model that non-technical users can interact with:
1. **Exclude Genomic & Clustered Features**: `Pam50 + Claudin-low subtype`, `Integrative Cluster`, `3-Gene classifier subtype`, `Nottingham prognostic index`, `Mutation Count`, `Cohort`, `Oncotree Code`.
2. **Exclude Target Leakage Attributes**: `Overall Survival (Years)`, `Relapse Free Status (Years)`, `Relapse Free Status`.
3. **Retain 15 Understandable Clinical Attributes**:
   - **Demographics**: `Age at Diagnosis`, `Inferred Menopausal State`
   - **Tumor Anatomical Characteristics**: `Tumor Size` (mm), `Tumor Stage` (1-4), `Primary Tumor Laterality` (Left/Right)
   - **Histopathology**: `Neoplasm Histologic Grade` (1-3), `Cellularity` (Low/Moderate/High)
   - **Lymph Node Burden**: `Lymph nodes examined positive`
   - **Hormone Receptor Status**: `ER Status`, `PR Status`, `HER2 Status`
   - **Therapeutic Interventions**: `Type of Breast Surgery` (Mastectomy/Breast Conserving), `Chemotherapy`, `Radio Therapy`, `Hormone Therapy`
""")

    add_code("""# Feature Selection
NUMERIC_FEATURES = [
    'Age at Diagnosis',
    'Tumor Size',
    'Tumor Stage',
    'Lymph nodes examined positive',
    'Neoplasm Histologic Grade'
]

CATEGORICAL_FEATURES = [
    'Type of Breast Surgery',
    'Chemotherapy',
    'Radio Therapy',
    'Hormone Therapy',
    'ER Status',
    'PR Status',
    'HER2 Status',
    'Inferred Menopausal State',
    'Primary Tumor Laterality',
    'Cellularity'
]

TARGET_COL = 'Overall Survival Status'

# Retain only selected features + target
analysis_df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]].copy()
print(f"Filtered clinical dataframe shape: {analysis_df.shape}")
analysis_df.describe().T[['mean', 'std', 'min', '50%', 'max']]
""")

    # Feature Engineering
    add_md("""---
## 5. Clinical Feature Engineering
To simplify complex quantitative relationships for clinicians and patients, we engineer 4 intuitive indicators:
- **`Age_Group`**: Stratified into `<45` (Young), `45-60` (Middle), `60-75` (Senior), and `>75` (Elderly).
- **`Tumor_Size_Category`**: Staged into T1 (`<=20mm`, Small), T2 (`20-50mm`, Medium), and T3/T4 (`>50mm`, Large).
- **`Has_Positive_Nodes`**: Binary flag indicating presence of positive lymph nodes.
- **`Therapy_Count`**: Composite multi-modality treatment score (Chemotherapy + Radio Therapy + Hormone Therapy).
""")

    add_code("""# Age Groups
analysis_df['Age_Group'] = pd.cut(
    analysis_df['Age at Diagnosis'],
    bins=[0, 45, 60, 75, 120],
    labels=['Young (<45)', 'Middle (45-60)', 'Senior (60-75)', 'Elderly (>75)']
)

# Tumor Size Categories
analysis_df['Tumor_Size_Category'] = pd.cut(
    analysis_df['Tumor Size'],
    bins=[-1, 20, 50, 500],
    labels=['Small (<=20mm)', 'Medium (20-50mm)', 'Large (>50mm)']
)

# Binary Lymph Node Involvement
analysis_df['Has_Positive_Nodes'] = (analysis_df['Lymph nodes examined positive'] > 0).astype(int)

# Multi-modality Therapy Index
analysis_df['Therapy_Count'] = (
    (analysis_df['Chemotherapy'] == 'Yes').astype(int) +
    (analysis_df['Radio Therapy'] == 'Yes').astype(int) +
    (analysis_df['Hormone Therapy'] == 'Yes').astype(int)
)

analysis_df[['Age_Group', 'Tumor_Size_Category', 'Has_Positive_Nodes', 'Therapy_Count']].head()
""")

    # Visualizations
    add_md("""---
## 6. Deep Visual Insights & Exploratory Data Analysis
We now generate high-resolution visual insights to examine clinical patterns driving long-term survival.
""")

    add_code("""# Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 7))
corr_cols = NUMERIC_FEATURES + ['Has_Positive_Nodes', 'Therapy_Count', TARGET_COL]
corr_matrix = analysis_df[corr_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.3, vmax=0.3, ax=ax, linewidths=0.5)
ax.set_title('Clinical Features Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()
""")

    add_code("""# Survival Rate Breakdown by Key Clinical Dimensions
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# 1. Survival Rate by Age Group
sns.barplot(data=analysis_df, x='Age_Group', y=TARGET_COL, ax=axes[0, 0], palette='Blues_d', ci=None)
axes[0, 0].set_title('Survival Rate by Age Group', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Survival Rate')
axes[0, 0].set_ylim(0, 0.6)

# 2. Survival Rate by Tumor Stage
stage_rounded = analysis_df['Tumor Stage'].round().astype(int).clip(0, 4)
sns.barplot(x=stage_rounded, y=analysis_df[TARGET_COL], ax=axes[0, 1], palette='Reds_d', ci=None)
axes[0, 1].set_title('Survival Rate by Tumor Stage', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Tumor Stage')
axes[0, 1].set_ylabel('Survival Rate')
axes[0, 1].set_ylim(0, 0.6)

# 3. Survival Rate by Hormone Receptor Status
receptors_df = analysis_df.melt(
    id_vars=[TARGET_COL],
    value_vars=['ER Status', 'PR Status', 'HER2 Status'],
    var_name='Receptor', value_name='Status'
)
sns.barplot(data=receptors_df, x='Receptor', y=TARGET_COL, hue='Status', ax=axes[1, 0], palette='coolwarm', ci=None)
axes[1, 0].set_title('Survival by Receptor Status (ER, PR, HER2)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Survival Rate')

# 4. Survival Rate by Breast Surgery Type
sns.barplot(data=analysis_df, x='Type of Breast Surgery', y=TARGET_COL, ax=axes[1, 1], palette='Purples_d', ci=None)
axes[1, 1].set_title('Survival Rate by Surgery Type', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Survival Rate')

plt.tight_layout()
plt.show()
""")

    # Modeling & Tuning
    add_md("""---
## 7. Machine Learning Modeling & GridSearchCV Hyperparameter Tuning
We split the data using an 80/20 stratified split, construct an automated `ColumnTransformer` preprocessor (`StandardScaler` for numeric, `OneHotEncoder` for categorical), and systematically tune 4 algorithms:
1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **AdaBoost Classifier**
4. **XGBoost Classifier**
""")

    add_code("""# Prepare feature matrix X and target y
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
y = df[TARGET_COL].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CATEGORICAL_FEATURES)
    ]
)

# Algorithm hyperparameter spaces
models_config = {
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [3, 5, 8, 12, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [4, 6, 8, 12],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__class_weight': ['balanced', None]
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [50, 100, 150],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss'),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__subsample': [0.8, 1.0],
            'classifier__scale_pos_weight': [1, 2]
        }
    }
}
""")

    add_code("""# Run GridSearchCV across all 4 algorithms
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
benchmark_results = []
best_estimators = {}

for model_name, config in models_config.items():
    print(f"Optimizing {model_name}...")
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', config['model'])
    ])
    
    grid = GridSearchCV(pipe, param_grid=config['params'], cv=cv, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_pipe = grid.best_estimator_
    best_estimators[model_name] = best_pipe
    
    # Evaluate on test set
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    benchmark_results.append({
        'Algorithm': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4),
        'Best_CV_F1': round(grid.best_score_, 4),
        'Best_Params': grid.best_params_
    })

comparison_df = pd.DataFrame(benchmark_results).sort_values(by=['F1-Score', 'ROC-AUC', 'Accuracy'], ascending=False)
display(comparison_df[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']])
""")

    # Model Evaluation & ROC Curve
    add_md("""---
## 8. Comparative Evaluation, ROC Curves & Feature Importance
We visualize model ROC curves and inspect feature importance from our winning model.
""")

    add_code("""# Plot ROC Curves for all algorithms
plt.figure(figsize=(9, 6))

for model_name, pipe in best_estimators.items():
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance', alpha=0.6)
plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
""")

    add_code("""# Feature Importance for the Winning Model
best_model_name = comparison_df.iloc[0]['Algorithm']
best_pipe = best_estimators[best_model_name]
print(f"Top Performing Model: {best_model_name}")

# Extract feature names from ColumnTransformer
cat_encoder = best_pipe.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
all_feature_names = list(NUMERIC_FEATURES) + list(cat_feature_names)

classifier = best_pipe.named_steps['classifier']
if hasattr(classifier, 'feature_importances_'):
    importances = classifier.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_df.head(12), x='Importance', y='Feature', palette='viridis')
    plt.title(f'Top 12 Feature Importances ({best_model_name})', fontsize=13, fontweight='bold')
    plt.xlabel('Relative Importance Score')
    plt.tight_layout()
    plt.show()
""")

    # Serialization
    add_md("""---
## 9. Model Serialization
The best performing model and its complete preprocessing pipeline are saved to `best_breast_cancer_model.pkl` for immediate inference inside the Streamlit web application.
""")

    add_code("""# Serialize winning pipeline
EXPORT_PATH = 'best_breast_cancer_model.pkl'
joblib.dump(best_pipe, EXPORT_PATH)
print(f"[OK] Successfully saved winning model to '{EXPORT_PATH}'")

# Developed by Najeeb Ullah | Machine Learning Engineer
print("Pipeline complete. Developed by Najeeb Ullah | Machine Learning Engineer")
""")

    output_path = os.path.join(r"d:\breast_cancer_project2026", "EDA_and_Feature_Engineering.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Jupyter Notebook generated at: {output_path}")

if __name__ == "__main__":
    build_notebook()
