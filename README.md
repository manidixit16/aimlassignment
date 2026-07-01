# Wine Quality Comparative Study

Multi-class classification comparative study — **Logistic Regression vs Random Forest**
on red vs white wine quality prediction, per the assignment brief (Section 1 + Tasks 1–7).

## Contents

- `Wine_Quality_Comparative_Study_Mani_PGDSAI3.ipynb` — full notebook: Section 1 + Tasks 1–7, executed with all outputs shown
- `winequality-red.csv` — red wine dataset (1,599 samples)
- `winequality-white.csv` — white wine dataset (4,898 samples)

## Method (fixed across both datasets for a fair comparison)

- 80/20 stratified train/test split: `test_size=0.2, random_state=42, stratify=y`
- **Model A:** `StandardScaler()` + `LogisticRegression(max_iter=5000)`
- **Model B:** `RandomForestClassifier(n_estimators=300, random_state=42)`
- Metrics used consistently: Accuracy, Confusion Matrix, Classification Report, **macro F1-score**

## Results (Task 6 comparison table)

| Dataset | Model               | Accuracy | F1-score | Key Observation |
|---------|---------------------|----------|----------|------------------|
| Red     | Logistic Regression | 0.5906   | 0.2776   | 0.00 recall on rare classes 3/4/8 |
| Red     | Random Forest       | 0.6813   | 0.4094   | Best red-wine result |
| White   | Logistic Regression | 0.5490   | 0.2367   | Weakest result; strong majority-class bias |
| White   | Random Forest       | 0.6724   | 0.4222   | Best macro F1 overall; some recall recovered on class 8 |

## Conclusion

Random Forest outperforms Logistic Regression on both datasets by capturing non-linear
interactions between chemical properties that a linear model cannot. Red wine is
marginally easier under Logistic Regression (fewer classes, less extreme imbalance), but
white wine's larger sample size lets Random Forest edge ahead on macro F1. The extreme
quality classes (3, 4, 8, 9) are the hardest to predict in both datasets due to very low
sample counts and heavy feature overlap with the dominant classes 5 and 6. Proposed next
step: `class_weight="balanced"` on both models to directly target minority-class recall.
