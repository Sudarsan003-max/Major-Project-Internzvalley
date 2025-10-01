
# 🫀 Heart Disease Classification using Random Forest

Predict heart disease presence from clinical and demographic features
using a Random Forest Classifier built, tuned, and explained in
Google Colab.

---

## 📌 Overview

This project implements an end-to-end machine learning workflow in
Google Colab: - Data upload and preprocessing - Feature selection via
model importances - Model training (default vs tuned Random Forests) -
Evaluation via confusion matrix and classification report -
Explainability using SHAP - Visual insights and clinical interpretation

> Environment: 100% Google Colab (no local setup required)

---

## 🎯 Objectives

- Understand Random Forests: intuition, pros, and cons - Perform
feature selection using Random Forest feature importances - Compare
default vs tuned models - Evaluate model performance with standard
metrics - Explain model decisions using SHAP - Derive clinically
meaningful insights

---

## 🧰 Tech Stack

- Python (Colab)
- pandas, numpy
- scikit-learn
- seaborn, matplotlib
- shap

---

## 📊 Dataset

- Heart Disease dataset: 303 records, 14 clinical features
- File:
`heart-disease.csv`
- Target column: typically `target` (0 = no
disease, 1 = disease). Adjust if different in your dataset.

Note: This project assumes a standard heart disease tabular dataset. If
you're using a variant, update column names accordingly.

---

## 🚀 Quick Start (Google Colab)

1) Open Colab  - https://colab.research.google.com

2) Create a new notebook

3) Upload the dataset 
- Run the upload cell and select
`heart-disease.csv`

4) Run all cells sequentially
5) All outputs (plots and metrics) will
display inline

---

## 🔧 Setup & Imports

```# Core libraries (available in Colab) import pandas as pd
import numpy as np import matplotlib.pyplot as plt import seaborn as sns

# Modeling & evaluation from sklearn.model_selection import
train_test_split, GridSearchCV from sklearn.ensemble import
RandomForestClassifier from sklearn.metrics import confusion_matrix,
classification_report, accuracy_score

# Explainability import shap

# Colab upload from google.colab import files

# Optional: in case shap isn\'t available in your environment
# !pip
install shap -q
```
---

## 📥 Data Load

```# Upload file in Colab uploaded = files.upload()
# choose
heart-disease.csv

# Read data df = pd.read_csv(\'heart-disease.csv\')

# Peek df.head() df.info() df.describe()
```

If your target column isn't named `target`, update the code below
accordingly.

---

## 🔬 Train/Test Split

```# Split features/target X =
df.drop(columns=\[\'target\'\]) \# change if your target column has a
different name y = df\[\'target\'\]

\# Train/test split (stratified) X_train, X_test, y_train, y_test =
train_test_split( X, y, test_size=0.2, stratify=y, random_state=42 )
```

---

## 🌲 Baseline Random Forest

```rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

y_pred_default = rf_default.predict(X_test) acc_default =
accuracy_score(y_test, y_pred_default)

print(f\'Baseline RF Accuracy: {acc_default:.2%}\')
print(classification_report(y_test, y_pred_default))
```

---

## 🎛️ Hyperparameter Tuning (Grid Search)

```
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

print('Best params:', grid.best_params_)

y_pred_tuned = best_rf.predict(X_test)
acc_tuned = accuracy_score(y_test, y_pred_tuned)

print(f'Tuned RF Accuracy: {acc_tuned:.2%}')
print(classification_report(y_test, y_pred_tuned))
```

---

## 📉 Confusion Matrix

```
def plot_confusion(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(title)
    plt.show()

plot_confusion(y_test, y_pred_default, title='Default RF - Confusion Matrix')
plot_confusion(y_test, y_pred_tuned,   title='Tuned RF - Confusion Matrix')
```

---

## ⭐ Feature Importance (Model-Based Selection)

```
importances = pd.Series(rf_default.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=importances_sorted.values, y=importances_sorted.index, color='teal')
plt.title('Random Forest Feature Importances (Default Model)')
plt.xlabel('Importance'); plt.ylabel('Feature')
plt.show()
```

Tip: You can perform feature selection by keeping features above a
certain importance threshold and retraining.

---

## 🧠 SHAP Explainability

```
# SHAP values for the tuned model (works for tree-based models)
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)

# Summary plot (for binary classification, index 1 is typically the positive class)
shap.summary_plot(shap_values[1], X_test, show=True)  # shows global feature impact
```

If you see a display issue in Colab, run: 
```
shap.initjs()
```
---

## 📈 Results (from Colab Execution)

- Default Random Forest: 86% accuracy
- Tuned Random Forest: 84% accuracy

Key features identified:
- oldpeak (ST depression induced by exercise)
- thalach (Maximum heart rate achieved)
- cp (Chest pain type)
- thal (Thalassemia)
- trestbps (Resting blood pressure)

Note: It's possible for a tuned model to underperform on the test split
due to overfitting or hyperparameter choices. Consider cross-validation,
different grids, or randomized search.

---

## 🧩 Why Random Forest?

- Intuition:
- Ensemble of decision trees built on bootstrapped
samples
- Random feature subsets reduce correlation between trees
- Final prediction via majority vote (classification)

- Advantages:
- Handles non-linearities and feature interactions
- Robust to noise and outliers
- Built-in feature importance
- Typically strong baseline performance with minimal tuning

- Disadvantages:
- Less interpretable than single trees (mitigated via SHAP)
- Can be slower for very large datasets
- Tuning may not always
improve generalization

---

## 🩺 Clinical Insights (Interpretation Guide)

\- Higher oldpeak often associates with increased risk - Higher thalach
(max heart rate) may be protective in some cohorts - Certain chest pain
types (cp) are predictive - thal and trestbps can provide additional
discriminative power

Caution: Insights are dataset-dependent and should not be used for
clinical decisions without validation. Always consult medical
professionals and perform rigorous evaluation.

---

## 💡 Colab Advantages Used

- Runs fully in browser (no local setup)
- Pre-installed scientific
Python stack - Inline visualizations
- Easy data upload and sharing
- Optional GPU/TPU (not required for Random Forest)

---

## 🔄 Typical Workflow in Colab

1) Data Upload → Colab's file upload widget
2) Data Analysis → pandas/numpy
3) Visualization → seaborn/matplotlib
4)  Modeling → scikit-learn RandomForestClassifier
5) Evaluation → confusion matrix, classification report
6) Explainability → SHAP plots
7) Save/Share → Notebook and outputs via Google Drive

---

## 📁 Project Files

- Heart_Disease_Random_Forest.ipynb --- main Colab notebook -
heart-disease.csv --- dataset file (upload when prompted)

---

## 🧪 Reproducibility

- Set `random_state=42` in train/test split andRandomForestClassifier
-  Use stratified splitting for balanced classes
- Consider cross-validation for stable metrics

---

## 🧯 Troubleshooting

- SHAP import error:
-  Run \`!pip install shap -q\` and re-import
-  Target column not found:
- Ensure your dataset has a target column
(e.g., `target`) and update code
- Class imbalance:
- Try`class_weight=\'balanced\'` in RandomForestClassifier

---

## 🎓 Learning Outcomes

- Hands-on ML development in Google Colab
- Practical Random Forest modeling
- Hyperparameter tuning with GridSearchCV
- Feature importance and SHAP-based explainability
- Translating model outputs into domain insights

---

## 👤 Author

- Sudarsananarayanan U R
- Data Science - May Batch
- Date: 17/07/2024
- Platform: Google Colab

---

## 🔗 Useful Links

- Google Colab: https://colab.research.google.com
- Colab Docs: https://research.google.com/colaboratory/faq.html
- scikit-learn: https://scikit-learn.org
- SHAP: https://shap.readthedocs.io

---

## 🏷️ Tags

google-colab, machine-learning, random-forest, healthcare, classification, data-science, heart-disease, python, colab-notebook, medical-ai, cloud-computing

---

## ⚖️ License


