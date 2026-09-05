"""
================================================================================
Breast Cancer Survival Prognosis: Machine Learning Pipeline & Model Training
Developed by Najeeb Ullah | Machine Learning Engineer
================================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# 1. Load Data
DATA_PATH = os.path.join(os.path.dirname(__file__), 'cleaned_breast_cancer.csv')
print(f"[*] Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Clean column names if necessary
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 2. Layman Feature Selection
# Strictly clinical & laymen-accessible features; exclude target leakage and complex genomic clusters
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

# Verify features exist
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
y = df[TARGET_COL].astype(int)

print(f"[*] Features selected: {len(NUMERIC_FEATURES)} Numerical, {len(CATEGORICAL_FEATURES)} Categorical")
print(f"[*] Dataset shape: {X.shape}, Target distribution: \n{y.value_counts(normalize=True).to_dict()}")

# 3. Train-Test Split (Stratified 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"[*] Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# 4. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CATEGORICAL_FEATURES)
    ]
)

# 5. Model Definitions & Hyperparameter Grids
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

# 6. Model Training & GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
best_estimators = {}

print("\n" + "="*80)
print("EXECUTING GRID SEARCH CV HYPERPARAMETER TUNING")
print("="*80)

for model_name, config in models_config.items():
    print(f"\n>>> Tuning {model_name}...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', config['model'])
    ])
    
    grid = GridSearchCV(
        pipeline,
        param_grid=config['params'],
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)
    
    best_pipe = grid.best_estimator_
    best_estimators[model_name] = best_pipe
    
    # Evaluate on holdout test set
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"    Best CV Score (F1): {grid.best_score_:.4f}")
    print(f"    Best Parameters: {grid.best_params_}")
    print(f"    Test Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    
    results.append({
        'Algorithm': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4),
        'Best_CV_F1': round(grid.best_score_, 4),
        'Best_Params': str(grid.best_params_)
    })

# 7. Comparison Summary
results_df = pd.DataFrame(results).sort_values(by=['F1-Score', 'ROC-AUC', 'Accuracy'], ascending=False)
print("\n" + "="*80)
print("COMPREHENSIVE MODEL EVALUATION BENCHMARK")
print("="*80)
print(results_df[['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].to_string(index=False))

# 8. Save Best Performing Model
winning_model_name = results_df.iloc[0]['Algorithm']
winning_pipeline = best_estimators[winning_model_name]
model_export_path = os.path.join(os.path.dirname(__file__), 'best_breast_cancer_model.pkl')
joblib.dump(winning_pipeline, model_export_path)

# Save metadata json for Streamlit app
metadata = {
    'winning_model': winning_model_name,
    'numeric_features': NUMERIC_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES,
    'target_col': TARGET_COL,
    'results': results_df.to_dict(orient='records'),
    'developer': 'Najeeb Ullah | Machine Learning Engineer'
}
metadata_path = os.path.join(os.path.dirname(__file__), 'model_metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4)

print(f"\n[OK] Best Performing Model: '{winning_model_name}' successfully serialized to: {model_export_path}")
print(f"[OK] Metadata and benchmark results saved to: {metadata_path}")
print(f"[OK] Pipeline ready for deployment in Streamlit application.")
