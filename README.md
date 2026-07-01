# Wine Quality Comparative Study

Comparison of **Logistic Regression** vs **Random Forest** classifiers for predicting
wine `quality` on the UCI Wine Quality datasets (red and white variants), based on
physicochemical properties (acidity, sugar, sulphates, alcohol, etc.).

## Contents

- `Wine_Quality_Comparative_Study_Mani_PGDSAI3.ipynb` — full notebook (Tasks 1–7), executed with real outputs
- `winequality-red.csv` — red wine dataset (1,599 samples)
- `winequality-white.csv` — white wine dataset (4,898 samples)

## Method

- 80/20 stratified train/test split (`random_state=42`)
- Logistic Regression: `StandardScaler` + `LogisticRegression(max_iter=5000)`
- Random Forest: `RandomForestClassifier(n_estimators=300, random_state=42)`
- Metrics: Accuracy, Macro F1, full classification report, confusion matrices

## Results

| Dataset | Model               | Accuracy | Macro F1 |
|---------|---------------------|----------|----------|
| Red     | Logistic Regression | 0.5906   | 0.2776   |
| Red     | Random Forest       | 0.6813   | 0.4094   |
| White   | Logistic Regression | 0.5490   | 0.2367   |
| White   | Random Forest       | 0.6724   | 0.4222   |

## Conclusion

Random Forest outperforms Logistic Regression on both datasets, on both accuracy
and macro F1. The extreme quality classes (3, 4, 8, 9) are the hardest to predict
for both models due to very low sample counts and overlap with the dominant
middle classes (5 and 6). A natural next step is using `class_weight="balanced"`
or resampling techniques (e.g. SMOTE) to improve minority-class performance.
