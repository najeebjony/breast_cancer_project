"""
================================================================================
Deep Analysis, Overfitting/Underfitting Detection & Confusion Matrix Evaluation
Breast Cancer Survival Prognosis
Developed by Najeeb Ullah | Machine Learning Engineer
================================================================================

This script generates the Deep_Analysis_and_Model_Evaluation.ipynb notebook.
Run: python Deep_Analysis_and_Model_Evaluation.py
"""

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3"
}

cells = []

# ============================================================
# CELL 1: Title & Introduction
# ============================================================
cells.append(new_markdown_cell("""# 🔬 Deep Analysis, Overfitting/Underfitting Detection & Model Evaluation
### **Developed by Najeeb Ullah | Machine Learning Engineer**

---

## Notebook Objectives

This notebook provides a **comprehensive deep-dive analysis** of the Breast Cancer Survival Prognosis ML pipeline:

1. **Dataset Audit & Quality Assessment** — Schema, missing values, duplicates, outliers
2. **Deep Exploratory Data Analysis (EDA)** — Distributions, survival breakdowns, feature interactions
3. **Class Imbalance Analysis** — Target distribution & impact on model performance
4. **Feature Correlation & Multicollinearity** — Heatmaps, VIF analysis
5. **Model Training with Train/Validation/Test Split** — Proper 3-way split
6. **Overfitting & Underfitting Detection** — Train vs. Validation vs. Test scores comparison
7. **Learning Curves** — Visual diagnosis of bias-variance tradeoff
8. **Confusion Matrices** — For all 4 models with detailed classification reports
9. **ROC Curves & Precision-Recall Curves** — Full diagnostic suite
10. **Feature Importance Analysis** — What drives the model predictions
11. **Final Verdict & Recommendations** — Summary of findings
"""))

# ============================================================
# CELL 2: Imports
# ============================================================
cells.append(new_code_cell("""# Core Libraries
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Scikit-Learn
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    cross_val_score, learning_curve
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score, ConfusionMatrixDisplay
)
from sklearn.dummy import DummyClassifier
import joblib

# Visual Settings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.facecolor'] = 'white'

print("✅ All libraries imported successfully.")
"""))

# ============================================================
# CELL 3: Data Loading
# ============================================================
cells.append(new_markdown_cell("""---
## 1. Dataset Audit & Quality Assessment
Load the cleaned clinical dataset, inspect schema, check for missing values, duplicates, and anomalies.
"""))

cells.append(new_code_cell("""# Load dataset
df = pd.read_csv('cleaned_breast_cancer.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"📊 Dataset Shape: {df.shape[0]} patients × {df.shape[1]} features")
print(f"📦 Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"🔍 Missing Values (Total): {df.isnull().sum().sum()}")
print(f"📋 Duplicate Rows: {df.duplicated().sum()}")
print()

# Display dtypes
print("=== Column Data Types ===")
for col in df.columns:
    null_count = df[col].isnull().sum()
    unique_count = df[col].nunique()
    print(f"  {col:40s} | {str(df[col].dtype):10s} | {unique_count:5d} unique | {null_count} nulls")
"""))

cells.append(new_code_cell("""# First 5 rows
df.head()
"""))

cells.append(new_code_cell("""# Statistical Summary of Numeric Features
df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(3)
"""))

# ============================================================
# CELL 4: Target Distribution & Class Imbalance
# ============================================================
cells.append(new_markdown_cell("""---
## 2. Target Variable Analysis & Class Imbalance

The target variable **`Overall Survival Status`** is binary:
- **0 = Deceased** (did not survive the study period)
- **1 = Survived / Living**

Class imbalance can severely impact model performance — especially Precision, Recall, and F1-Score.
"""))

cells.append(new_code_cell("""# Target Distribution
target_counts = df['Overall Survival Status'].value_counts()
target_pct = df['Overall Survival Status'].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie Chart
colors = ['#e74c3c', '#2ecc71']
axes[0].pie(target_counts, labels=['Deceased (0)', 'Survived (1)'],
            autopct='%1.1f%%', startangle=140, colors=colors,
            explode=(0, 0.05), shadow=True, textprops={'fontsize': 12})
axes[0].set_title('Overall Survival Status Distribution', fontsize=14, fontweight='bold')

# Bar Chart with counts
bars = axes[1].bar(['Deceased (0)', 'Survived (1)'], target_counts.values,
                    color=colors, edgecolor='black', alpha=0.85)
axes[1].set_ylabel('Patient Count', fontsize=12)
axes[1].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
for bar, count, pct in zip(bars, target_counts.values, target_pct.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                 f'{count}\\n({pct:.1f}%)', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# Imbalance Ratio
majority = target_counts.max()
minority = target_counts.min()
imbalance_ratio = majority / minority
print(f"\\n📊 Class Distribution:")
print(f"   Deceased (0): {target_counts[0]} patients ({target_pct[0]:.1f}%)")
print(f"   Survived (1): {target_counts[1]} patients ({target_pct[1]:.1f}%)")
print(f"   Imbalance Ratio: {imbalance_ratio:.2f}:1")
print(f"\\n⚠️  The dataset is {'IMBALANCED' if imbalance_ratio > 1.5 else 'relatively balanced'} "
      f"— this will affect model training and evaluation metrics.")
"""))

# ============================================================
# CELL 5: Deep EDA - Distributions
# ============================================================
cells.append(new_markdown_cell("""---
## 3. Deep Exploratory Data Analysis

### 3.1 Numerical Feature Distributions
Examining the distribution of each numerical feature, split by survival outcome.
"""))

cells.append(new_code_cell("""NUMERIC_FEATURES = [
    'Age at Diagnosis', 'Tumor Size', 'Tumor Stage',
    'Lymph nodes examined positive', 'Neoplasm Histologic Grade'
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(NUMERIC_FEATURES):
    ax = axes[i]
    for label, color, name in [(0, '#e74c3c', 'Deceased'), (1, '#2ecc71', 'Survived')]:
        subset = df[df['Overall Survival Status'] == label][col]
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=name, edgecolor='black', linewidth=0.5)
    ax.set_title(f'{col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.legend()

# Remove empty subplot
axes[5].set_visible(False)
plt.suptitle('Numerical Feature Distributions by Survival Status', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

# ============================================================
# CELL 6: Box Plots for Outlier Detection
# ============================================================
cells.append(new_markdown_cell("""### 3.2 Outlier Detection with Box Plots
Identifying extreme values that could impact model training.
"""))

cells.append(new_code_cell("""fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for i, col in enumerate(NUMERIC_FEATURES):
    sns.boxplot(data=df, x='Overall Survival Status', y=col, ax=axes[i],
                hue='Overall Survival Status',
                palette=['#e74c3c', '#2ecc71'], width=0.5, legend=False)
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Survival Status')

plt.suptitle('Box Plots: Feature Distributions by Survival Outcome', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Quantify outliers using IQR method
print("\\n📊 Outlier Analysis (IQR Method):")
for col in NUMERIC_FEATURES:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"   {col:40s} → {len(outliers):4d} outliers ({len(outliers)/len(df)*100:.1f}%)")
"""))

# ============================================================
# CELL 7: Survival Rate Breakdowns
# ============================================================
cells.append(new_markdown_cell("""### 3.3 Survival Rate Across Key Clinical Dimensions
"""))

cells.append(new_code_cell("""CATEGORICAL_FEATURES = [
    'Type of Breast Surgery', 'Chemotherapy', 'Radio Therapy',
    'Hormone Therapy', 'ER Status', 'PR Status', 'HER2 Status',
    'Inferred Menopausal State', 'Primary Tumor Laterality', 'Cellularity'
]

fig, axes = plt.subplots(2, 5, figsize=(24, 10))
axes = axes.flatten()

for i, col in enumerate(CATEGORICAL_FEATURES):
    survival_rate = df.groupby(col)['Overall Survival Status'].mean() * 100
    bars = axes[i].bar(survival_rate.index, survival_rate.values,
                       color=sns.color_palette('viridis', len(survival_rate)),
                       edgecolor='black', alpha=0.85)
    axes[i].set_title(f'{col}', fontsize=10, fontweight='bold')
    axes[i].set_ylabel('Survival %')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_ylim(0, 55)
    for bar in bars:
        yval = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2.0, yval + 0.8,
                     f'{yval:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.suptitle('Survival Rate by Categorical Features', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

# ============================================================
# CELL 8: Correlation Heatmap
# ============================================================
cells.append(new_markdown_cell("""### 3.4 Feature Correlation Matrix
Examining Pearson correlations between numerical features and the target variable.
"""))

cells.append(new_code_cell("""corr_cols = NUMERIC_FEATURES + ['Overall Survival Status']
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            mask=mask, ax=ax, linewidths=0.5, cbar_kws={'shrink': 0.8},
            vmin=-0.5, vmax=0.5, square=True)
ax.set_title('Pearson Correlation Matrix (Lower Triangle)', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()

# Highlight correlations with target
print("\\n📊 Correlations with Overall Survival Status:")
target_corr = corr_matrix['Overall Survival Status'].drop('Overall Survival Status').sort_values()
for feat, corr in target_corr.items():
    indicator = "⬇️" if corr < 0 else "⬆️"
    strength = "Strong" if abs(corr) > 0.2 else "Moderate" if abs(corr) > 0.1 else "Weak"
    print(f"   {indicator} {feat:40s} → r = {corr:+.4f} ({strength})")
"""))

# ============================================================
# CELL 9: Feature Engineering + Model Setup
# ============================================================
cells.append(new_markdown_cell("""---
## 4. Model Training with Proper Train / Validation / Test Split

### Why a 3-way split matters:
- **Training Set (60%)**: Model learns patterns
- **Validation Set (20%)**: Hyperparameter tuning & overfitting detection
- **Test Set (20%)**: Final unbiased evaluation

Comparing **Train vs. Validation vs. Test** scores is the gold standard for detecting overfitting/underfitting.
"""))

cells.append(new_code_cell("""TARGET_COL = 'Overall Survival Status'

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
y = df[TARGET_COL].astype(int)

# Step 1: Split into Train+Val (80%) and Test (20%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Step 2: Split Train+Val into Train (75% of 80% = 60%) and Val (25% of 80% = 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)

print(f"📊 Data Split Summary:")
print(f"   Training Set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"   Validation Set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"   Test Set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"\\n   Train target distribution: {dict(y_train.value_counts())}")
print(f"   Val target distribution:   {dict(y_val.value_counts())}")
print(f"   Test target distribution:  {dict(y_test.value_counts())}")

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CATEGORICAL_FEATURES)
    ]
)
"""))

# ============================================================
# CELL 10: Baseline Model
# ============================================================
cells.append(new_markdown_cell("""### 4.1 Naive Baseline (Majority Class Classifier)
Before training any model, we establish a baseline — what score does a "dumb" model achieve by always predicting the majority class?
"""))

cells.append(new_code_cell("""# Naive Baseline
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)

dummy_train_acc = dummy.score(X_train, y_train)
dummy_test_acc = dummy.score(X_test, y_test)

print(f"🎯 Naive Baseline (Always predicts majority class):")
print(f"   Train Accuracy: {dummy_train_acc:.4f}")
print(f"   Test Accuracy:  {dummy_test_acc:.4f}")
print(f"\\n   Any model must beat {dummy_test_acc:.4f} accuracy to be useful.")
"""))

# ============================================================
# CELL 11: Train All Models
# ============================================================
cells.append(new_markdown_cell("""### 4.2 Model Training & GridSearchCV Hyperparameter Tuning

Training 4 classifiers with exhaustive grid search:
1. **Decision Tree** — Simple, interpretable
2. **Random Forest** — Ensemble of trees, reduces variance
3. **AdaBoost** — Boosting weak learners
4. **XGBoost** — Gradient boosting, often state-of-the-art
"""))

cells.append(new_code_cell("""models_config = {
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
best_estimators = {}

for model_name, config in models_config.items():
    print(f"\\n{'='*60}")
    print(f"  🔧 Training & Tuning: {model_name}")
    print(f"{'='*60}")

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', config['model'])
    ])

    grid = GridSearchCV(pipe, param_grid=config['params'], cv=cv,
                        scoring='f1', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best_pipe = grid.best_estimator_
    best_estimators[model_name] = best_pipe

    # Predictions on Train, Validation, and Test
    y_train_pred = best_pipe.predict(X_train)
    y_val_pred = best_pipe.predict(X_val)
    y_test_pred = best_pipe.predict(X_test)

    y_train_proba = best_pipe.predict_proba(X_train)[:, 1]
    y_val_proba = best_pipe.predict_proba(X_val)[:, 1]
    y_test_proba = best_pipe.predict_proba(X_test)[:, 1]

    # Compute metrics for all 3 sets
    train_metrics = {
        'Accuracy': accuracy_score(y_train, y_train_pred),
        'F1': f1_score(y_train, y_train_pred),
        'ROC-AUC': roc_auc_score(y_train, y_train_proba)
    }
    val_metrics = {
        'Accuracy': accuracy_score(y_val, y_val_pred),
        'F1': f1_score(y_val, y_val_pred),
        'ROC-AUC': roc_auc_score(y_val, y_val_proba)
    }
    test_metrics = {
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'F1': f1_score(y_test, y_test_pred),
        'ROC-AUC': roc_auc_score(y_test, y_test_proba)
    }

    results.append({
        'Algorithm': model_name,
        'Train_Acc': round(train_metrics['Accuracy'], 4),
        'Val_Acc': round(val_metrics['Accuracy'], 4),
        'Test_Acc': round(test_metrics['Accuracy'], 4),
        'Train_F1': round(train_metrics['F1'], 4),
        'Val_F1': round(val_metrics['F1'], 4),
        'Test_F1': round(test_metrics['F1'], 4),
        'Train_AUC': round(train_metrics['ROC-AUC'], 4),
        'Val_AUC': round(val_metrics['ROC-AUC'], 4),
        'Test_AUC': round(test_metrics['ROC-AUC'], 4),
        'Best_CV_F1': round(grid.best_score_, 4),
        'Best_Params': grid.best_params_
    })

    # Print results
    print(f"  Best CV F1: {grid.best_score_:.4f}")
    print(f"  ┌──────────────┬──────────┬──────────┬──────────┐")
    print(f"  │   Metric     │  Train   │   Val    │   Test   │")
    print(f"  ├──────────────┼──────────┼──────────┼──────────┤")
    print(f"  │ Accuracy     │ {train_metrics['Accuracy']:.4f}   │ {val_metrics['Accuracy']:.4f}   │ {test_metrics['Accuracy']:.4f}   │")
    print(f"  │ F1-Score     │ {train_metrics['F1']:.4f}   │ {val_metrics['F1']:.4f}   │ {test_metrics['F1']:.4f}   │")
    print(f"  │ ROC-AUC      │ {train_metrics['ROC-AUC']:.4f}   │ {val_metrics['ROC-AUC']:.4f}   │ {test_metrics['ROC-AUC']:.4f}   │")
    print(f"  └──────────────┴──────────┴──────────┴──────────┘")

results_df = pd.DataFrame(results)
print("\\n\\n✅ All models trained successfully!")
"""))

# ============================================================
# CELL 12: Overfit / Underfit Analysis
# ============================================================
cells.append(new_markdown_cell("""---
## 5. 🔍 Overfitting & Underfitting Detection

### How to Diagnose:
| Condition | Train Score | Val/Test Score | Gap |
|-----------|------------|----------------|-----|
| **Overfitting** | Very High (>0.90) | Significantly Lower | Large (>0.10) |
| **Underfitting** | Low (<0.70) | Low | Small |
| **Good Fit** | Moderate-High | Close to Train | Small (<0.05) |

We compare **Train vs. Validation vs. Test** metrics for each model.
"""))

cells.append(new_code_cell("""# Overfit / Underfit Diagnosis
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
metrics_to_plot = [('Acc', 'Accuracy'), ('F1', 'F1-Score'), ('AUC', 'ROC-AUC')]

for idx, (suffix, title) in enumerate(metrics_to_plot):
    ax = axes[idx]
    x = np.arange(len(results_df))
    width = 0.25

    train_vals = results_df[f'Train_{suffix}'].values
    val_vals = results_df[f'Val_{suffix}'].values
    test_vals = results_df[f'Test_{suffix}'].values

    bars1 = ax.bar(x - width, train_vals, width, label='Train', color='#3498db', alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x, val_vals, width, label='Validation', color='#f39c12', alpha=0.85, edgecolor='black')
    bars3 = ax.bar(x + width, test_vals, width, label='Test', color='#2ecc71', alpha=0.85, edgecolor='black')

    ax.set_xlabel('Model')
    ax.set_ylabel(title)
    ax.set_title(f'{title}: Train vs Val vs Test', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Algorithm'], rotation=20, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.suptitle('🔍 Overfitting / Underfitting Detection: Train vs Validation vs Test',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(new_code_cell("""# Detailed Overfitting / Underfitting Report
print("=" * 80)
print("   📋 OVERFITTING / UNDERFITTING DIAGNOSTIC REPORT")
print("=" * 80)

for _, row in results_df.iterrows():
    model = row['Algorithm']
    train_f1 = row['Train_F1']
    val_f1 = row['Val_F1']
    test_f1 = row['Test_F1']
    gap_train_val = train_f1 - val_f1
    gap_train_test = train_f1 - test_f1

    print(f"\\n{'─'*60}")
    print(f"  🤖 {model}")
    print(f"{'─'*60}")
    print(f"  Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
    print(f"  Train-Val Gap:  {gap_train_val:+.4f}")
    print(f"  Train-Test Gap: {gap_train_test:+.4f}")

    # Diagnosis
    if train_f1 > 0.90 and gap_train_val > 0.10:
        diagnosis = "⚠️  OVERFITTING DETECTED"
        explanation = ("The model memorizes training data (very high train score) "
                      "but fails to generalize (large gap to validation/test).")
        remedy = "Consider: more regularization, pruning, fewer features, or more training data."
    elif train_f1 < 0.65 and val_f1 < 0.65:
        diagnosis = "⚠️  UNDERFITTING DETECTED"
        explanation = ("The model is too simple to capture the underlying patterns. "
                      "Both train and validation scores are low.")
        remedy = "Consider: more complex model, feature engineering, or reducing regularization."
    elif gap_train_val > 0.05:
        diagnosis = "⚡ MILD OVERFITTING (Acceptable)"
        explanation = ("There is some generalization gap, but it is within an acceptable range "
                      "for this dataset size and complexity.")
        remedy = "Monitor closely. Slight regularization tuning may help."
    else:
        diagnosis = "✅ GOOD FIT"
        explanation = ("Train and validation/test scores are reasonably close. "
                      "The model generalizes well.")
        remedy = "No major changes needed."

    print(f"\\n  Diagnosis: {diagnosis}")
    print(f"  Analysis:  {explanation}")
    print(f"  Remedy:    {remedy}")
"""))

# ============================================================
# CELL 13: Learning Curves
# ============================================================
cells.append(new_markdown_cell("""---
## 6. 📈 Learning Curves

Learning curves show how model performance changes with increasing training data size.

- **Overfitting**: Training score stays high while validation score plateaus far below
- **Underfitting**: Both curves converge at a low score
- **Good Fit**: Both curves converge at a high score with small gap
"""))

cells.append(new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (model_name, pipe) in enumerate(best_estimators.items()):
    ax = axes[idx]

    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_trainval, y_trainval,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='f1', n_jobs=-1, random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#3498db')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#e74c3c')
    ax.plot(train_sizes, train_mean, 'o-', color='#3498db', label='Training F1', linewidth=2)
    ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c', label='Validation F1', linewidth=2)

    ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('F1-Score')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Annotate the gap
    final_gap = train_mean[-1] - val_mean[-1]
    ax.annotate(f'Gap: {final_gap:.3f}',
                xy=(train_sizes[-1], (train_mean[-1] + val_mean[-1])/2),
                fontsize=10, fontweight='bold', color='purple',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.suptitle('📈 Learning Curves — Bias/Variance Diagnostic', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(new_markdown_cell("""### 📖 How to Read the Learning Curves:

- **Blue line** = Training F1-Score | **Red line** = Cross-Validation F1-Score
- **Large gap** between blue and red → **Overfitting** (model memorizes training data)
- **Both lines converge low** → **Underfitting** (model too simple)
- **Both lines converge high with small gap** → **Good fit** ✅
- **Shaded area** = ±1 standard deviation across folds
"""))

# ============================================================
# CELL 14: Confusion Matrices
# ============================================================
cells.append(new_markdown_cell("""---
## 7. 📊 Confusion Matrices

The confusion matrix is the most important evaluation tool for classification:

|  | **Predicted Negative** | **Predicted Positive** |
|--|----------------------|----------------------|
| **Actual Negative** | True Negative (TN) | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP) |

- **False Negatives (FN)** = Model predicted SURVIVED but patient actually DECEASED → **Most dangerous in clinical context**
- **False Positives (FP)** = Model predicted DECEASED but patient actually SURVIVED
"""))

cells.append(new_code_cell("""fig, axes = plt.subplots(2, 4, figsize=(22, 10))

for idx, (model_name, pipe) in enumerate(best_estimators.items()):
    # Test Set Confusion Matrix
    y_test_pred = pipe.predict(X_test)
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Normalized Confusion Matrix
    cm_test_norm = confusion_matrix(y_test, y_test_pred, normalize='true')

    # Raw counts
    ax1 = axes[0, idx]
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Deceased (0)', 'Survived (1)'],
                yticklabels=['Deceased (0)', 'Survived (1)'],
                cbar=False, linewidths=1, linecolor='black')
    ax1.set_title(f'{model_name}\\n(Raw Counts)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Actual')
    ax1.set_xlabel('Predicted')

    # Normalized (percentages)
    ax2 = axes[1, idx]
    sns.heatmap(cm_test_norm, annot=True, fmt='.2%', cmap='Oranges', ax=ax2,
                xticklabels=['Deceased (0)', 'Survived (1)'],
                yticklabels=['Deceased (0)', 'Survived (1)'],
                cbar=False, linewidths=1, linecolor='black', vmin=0, vmax=1)
    ax2.set_title(f'{model_name}\\n(Normalized %)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')

plt.suptitle('📊 Confusion Matrices — All Models on Test Set',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(new_code_cell("""# Detailed Classification Reports for all models
print("=" * 90)
print("   📋 DETAILED CLASSIFICATION REPORTS (Test Set)")
print("=" * 90)

for model_name, pipe in best_estimators.items():
    y_test_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)

    tn, fp, fn, tp = cm.ravel()

    print(f"\\n{'━'*70}")
    print(f"  🤖 {model_name}")
    print(f"{'━'*70}")
    print(f"  True Negatives (TN):  {tn:4d}  │  Correctly predicted Deceased")
    print(f"  True Positives (TP):  {tp:4d}  │  Correctly predicted Survived")
    print(f"  False Positives (FP): {fp:4d}  │  Predicted Survived but actually Deceased (Type I)")
    print(f"  False Negatives (FN): {fn:4d}  │  Predicted Deceased but actually Survived (Type II)")
    print(f"\\n  Sensitivity / Recall (TP Rate): {tp/(tp+fn):.4f}")
    print(f"  Specificity (TN Rate):          {tn/(tn+fp):.4f}")
    print(f"  False Positive Rate:            {fp/(fp+tn):.4f}")
    print(f"  False Negative Rate:            {fn/(fn+tp):.4f}")
    print(f"\\n{classification_report(y_test, y_test_pred, target_names=['Deceased (0)', 'Survived (1)'])}")
"""))

# ============================================================
# CELL 15: ROC Curves
# ============================================================
cells.append(new_markdown_cell("""---
## 8. ROC Curves & Precision-Recall Curves

- **ROC Curve**: Plots True Positive Rate vs. False Positive Rate at various thresholds.
  - AUC closer to 1.0 = better model
- **Precision-Recall Curve**: More informative for **imbalanced datasets**
  - Higher area = better at detecting the minority class
"""))

cells.append(new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ROC Curves
ax1 = axes[0]
for model_name, pipe in best_estimators.items():
    y_proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax1.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)

ax1.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC=0.500)', alpha=0.5)
ax1.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Precision-Recall Curves
ax2 = axes[1]
for model_name, pipe in best_estimators.items():
    y_proba = pipe.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    ax2.plot(recall, precision, label=f'{model_name} (AP={ap:.3f})', linewidth=2)

# Baseline
baseline = y_test.mean()
ax2.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline (AP={baseline:.3f})', alpha=0.5)
ax2.set_title('Precision-Recall Curves — All Models', fontsize=13, fontweight='bold')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""))

# ============================================================
# CELL 16: Feature Importance
# ============================================================
cells.append(new_markdown_cell("""---
## 9. Feature Importance Analysis

Understanding **which features drive the model's predictions** is critical for clinical trust and interpretability.
"""))

cells.append(new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for idx, (model_name, pipe) in enumerate(best_estimators.items()):
    ax = axes[idx]
    classifier = pipe.named_steps['classifier']

    if hasattr(classifier, 'feature_importances_'):
        cat_encoder = pipe.named_steps['preprocessor'].named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        all_feature_names = list(NUMERIC_FEATURES) + list(cat_feature_names)

        importances = classifier.feature_importances_
        feat_imp_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True).tail(15)

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(feat_imp_df)))
        ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'],
                color=colors, edgecolor='black', alpha=0.85)
        ax.set_title(f'{model_name} — Top 15 Features', fontsize=12, fontweight='bold')
        ax.set_xlabel('Importance Score')

plt.suptitle('🔬 Feature Importance — All Models', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

# ============================================================
# CELL 17: Model Comparison Summary Table
# ============================================================
cells.append(new_markdown_cell("""---
## 10. Comprehensive Model Comparison Summary

Final comparison table with **Train / Validation / Test** metrics and overfitting gap analysis.
"""))

cells.append(new_code_cell("""# Create comprehensive summary table
summary = results_df[['Algorithm', 'Train_Acc', 'Val_Acc', 'Test_Acc',
                       'Train_F1', 'Val_F1', 'Test_F1',
                       'Train_AUC', 'Val_AUC', 'Test_AUC', 'Best_CV_F1']].copy()

summary['Overfit_Gap_F1'] = (summary['Train_F1'] - summary['Test_F1']).round(4)
summary['Overfit_Gap_AUC'] = (summary['Train_AUC'] - summary['Test_AUC']).round(4)

# Style the table
print("=" * 100)
print("   📊 COMPREHENSIVE MODEL COMPARISON SUMMARY")
print("=" * 100)
display(summary.style.format({
    'Train_Acc': '{:.4f}', 'Val_Acc': '{:.4f}', 'Test_Acc': '{:.4f}',
    'Train_F1': '{:.4f}', 'Val_F1': '{:.4f}', 'Test_F1': '{:.4f}',
    'Train_AUC': '{:.4f}', 'Val_AUC': '{:.4f}', 'Test_AUC': '{:.4f}',
    'Best_CV_F1': '{:.4f}', 'Overfit_Gap_F1': '{:+.4f}', 'Overfit_Gap_AUC': '{:+.4f}'
}).background_gradient(subset=['Overfit_Gap_F1', 'Overfit_Gap_AUC'], cmap='RdYlGn_r')
  .highlight_max(subset=['Test_F1', 'Test_AUC'], color='#90EE90')
  .highlight_min(subset=['Overfit_Gap_F1', 'Overfit_Gap_AUC'], color='#90EE90'))
"""))

# ============================================================
# CELL 18: Cross-Validation Stability
# ============================================================
cells.append(new_markdown_cell("""---
## 11. Cross-Validation Score Stability

Examining variance across CV folds to assess model stability and reliability.
"""))

cells.append(new_code_cell("""fig, ax = plt.subplots(figsize=(12, 6))
cv_data = []

for model_name, pipe in best_estimators.items():
    scores = cross_val_score(pipe, X_trainval, y_trainval, cv=5, scoring='f1', n_jobs=-1)
    cv_data.append(scores)
    print(f"{model_name:20s} → CV F1: {scores.mean():.4f} ± {scores.std():.4f}  |  "
          f"Min: {scores.min():.4f}  Max: {scores.max():.4f}  Range: {scores.max()-scores.min():.4f}")

bp = ax.boxplot(cv_data, labels=list(best_estimators.keys()), patch_artist=True,
                medianprops=dict(color='red', linewidth=2))
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_title('Cross-Validation F1-Score Distribution (5-Fold)', fontsize=14, fontweight='bold')
ax.set_ylabel('F1-Score')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# ============================================================
# CELL 19: Final Verdict
# ============================================================
cells.append(new_markdown_cell("""---
## 12. 🏆 Final Verdict & Recommendations
"""))

cells.append(new_code_cell("""# Determine best model
best_idx = results_df['Test_F1'].idxmax()
best = results_df.loc[best_idx]

print("=" * 80)
print("   🏆 FINAL MODEL EVALUATION VERDICT")
print("=" * 80)

print(f"\\n   Best Performing Model: {best['Algorithm']}")
print(f"   ─────────────────────────────────────────")
print(f"   Test Accuracy:   {best['Test_Acc']:.4f}")
print(f"   Test F1-Score:   {best['Test_F1']:.4f}")
print(f"   Test ROC-AUC:    {best['Test_AUC']:.4f}")
print(f"   CV F1 (5-Fold):  {best['Best_CV_F1']:.4f}")
print(f"   Overfit Gap (F1): {best['Train_F1'] - best['Test_F1']:+.4f}")

gap = best['Train_F1'] - best['Test_F1']

print(f"\\n   📋 OVERFITTING VERDICT:")
if gap > 0.10:
    print(f"   ⚠️  The model shows SIGNIFICANT overfitting (gap: {gap:.4f})")
    print(f"       Recommendation: Increase regularization, reduce model complexity,")
    print(f"       or gather more training data.")
elif gap > 0.05:
    print(f"   ⚡ The model shows MILD overfitting (gap: {gap:.4f})")
    print(f"       This is acceptable for this dataset size but monitor in production.")
else:
    print(f"   ✅ The model shows GOOD generalization (gap: {gap:.4f})")
    print(f"       Train and test scores are close — no overfitting concern.")

print(f"\\n   📋 UNDERFITTING VERDICT:")
if best['Test_F1'] < 0.60:
    print(f"   ⚠️  Test F1 = {best['Test_F1']:.4f} is relatively LOW.")
    print(f"       Consider: feature engineering, more complex models, or")
    print(f"       addressing class imbalance (SMOTE, class weights).")
else:
    print(f"   ✅ Test F1 = {best['Test_F1']:.4f} indicates the model captures")
    print(f"       meaningful patterns in the data.")

print(f"\\n   📋 CLASS IMBALANCE IMPACT:")
print(f"   The dataset has a {y.value_counts()[0]/y.value_counts()[1]:.1f}:1 imbalance ratio.")
print(f"   F1-Score and ROC-AUC (not just accuracy) should be the primary metrics")
print(f"   to evaluate performance on the minority class (Survived = 1).")

print(f"\\n{'='*80}")
print(f"   Developed by Najeeb Ullah | Machine Learning Engineer")
print(f"{'='*80}")
"""))

# ============================================================
# Build and save the notebook
# ============================================================
nb.cells = cells

with open('Deep_Analysis_and_Model_Evaluation.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("[OK] Notebook 'Deep_Analysis_and_Model_Evaluation.ipynb' created successfully!")
print("Run it with: jupyter notebook Deep_Analysis_and_Model_Evaluation.ipynb")
