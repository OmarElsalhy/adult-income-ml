# 📊 Adult Income Classification — ML Models from Scratch vs Built-in

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Sklearn](https://img.shields.io/badge/scikit--learn-1.x-green?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A comprehensive machine learning project that implements **6 classification models** on the Adult Census Income dataset — each model built **twice**: once using scikit-learn's built-in library and once **from scratch using NumPy only**, with full evaluation and comparison.

---

## 🎯 Project Objective

Predict whether a person's annual income exceeds **$50K** based on demographic and work-related features.

This is a **binary classification** problem:
- `0` → Income **≤ 50K**
- `1` → Income **> 50K**

---

## 📁 Project Structure

```
adult-income-ml/
│
├── adult_income_ml.ipynb   # Main notebook (all models + analysis)
├── README.md               # This file
└── .gitignore
```

> ⚠️ The dataset (`adult.csv`) is **not included** in this repo.  
> 📥 Download it from Kaggle: [Adult Census Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income)  
> Then place `adult.csv` in the same folder as the notebook.

---

## 🧠 Models Implemented

| # | Model | Built-in | From Scratch | Preprocessing |
|---|-------|----------|--------------|---------------|
| 1 | **Decision Tree** | ✅ sklearn | ✅ Gini impurity + recursive splitting | Label Encoding |
| 2 | **Random Forest** | ✅ sklearn | ✅ Bagging + random feature subsets | Label Encoding |
| 3 | **Linear Regression** | ✅ sklearn | ✅ Normal Equation | OHE + StandardScaler |
| 4 | **Logistic Regression** | ✅ sklearn | ✅ Gradient Descent + Sigmoid | OHE + StandardScaler |
| 5 | **K-Nearest Neighbors** | ✅ sklearn | ✅ Euclidean distance + majority vote | OHE + MinMaxScaler |
| 6 | **Naïve Bayes** | ✅ sklearn | ✅ Gaussian likelihood + log-posteriors | Label Encoding |

---

## 🔄 Notebook Walkthrough

| Step | Description |
|------|-------------|
| **Step 0** | Import libraries |
| **Step 1** | Exploratory Data Analysis (EDA) + Visualizations |
| **Step 2** | Global Preprocessing — handle missing values, drop redundant columns, encode target |
| **Step 3** | Model-specific preprocessing (Encoding + Scaling per model) |
| **Step 4** | Unified evaluation function (Accuracy, Precision, Recall, F1, AUC, Confusion Matrix) |
| **Steps 5–10** | Each model: Built-in → From Scratch → Side-by-side comparison |
| **Step 11** | Full comparison: bar charts, heatmap, ROC curves, radar chart |
| **Step 12** | Best model analysis + final ranking table |

---

## 🧹 Preprocessing Summary

The dataset contains several issues that required careful handling:

- **Hidden missing values** — `?` used instead of `NaN` in `workclass`, `occupation`, `native-country` → replaced with **mode**
- **Redundant column** — `education` is fully captured by `educational-num` (ordinal) → **dropped**
- **Irrelevant column** — `fnlwgt` is a census sampling weight, not a predictive feature → **dropped**
- **Class imbalance** — 76% `≤50K` vs 24% `>50K` → reflected in evaluation metrics

Different preprocessing pipelines were applied per model type:

| Pipeline | Encoding | Scaling | Used For |
|----------|----------|---------|----------|
| A | Label Encoding | None | Decision Tree, Random Forest, Naïve Bayes |
| B | One-Hot Encoding | StandardScaler | Logistic Regression, Linear Regression |
| C | One-Hot Encoding | MinMaxScaler | KNN |

---

## 📈 Results Summary

> Results on 20% held-out test set (stratified split, `random_state=42`)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Random Forest | **~0.86** | **~0.76** | **~0.63** | **~0.69** | **~0.92** |
| Decision Tree | ~0.84 | ~0.72 | ~0.58 | ~0.64 | ~0.85 |
| Logistic Regression | ~0.85 | ~0.74 | ~0.58 | ~0.65 | ~0.90 |
| KNN | ~0.83 | ~0.70 | ~0.56 | ~0.62 | ~0.88 |
| Linear Regression | ~0.83 | ~0.69 | ~0.59 | ~0.64 | ~0.89 |
| Naïve Bayes | ~0.80 | ~0.63 | ~0.55 | ~0.58 | ~0.85 |

> 📌 Exact values will appear after running the notebook — the table above shows approximate expected ranges.

---

## 🏆 Best Model: Random Forest

**Why Random Forest wins:**

1. **Ensemble learning** — combines hundreds of Decision Trees, reducing variance and overfitting
2. **Bagging** — each tree trains on a different bootstrap sample → better generalization
3. **Random feature selection** — each tree sees a random subset of features → reduces correlation between trees
4. **No scaling needed** — works well with mixed numerical and label-encoded categorical data
5. **Robust to outliers** — splits on thresholds, not raw values

**Why Naïve Bayes ranks last:**

The independence assumption (all features are conditionally independent given the class) is clearly violated here — `marital-status` and `relationship` are strongly correlated, and so are `occupation` and `workclass`. This assumption causes systematic probability miscalibration.

---

## ⚙️ How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/adult-income-ml.git
cd adult-income-ml

# 2. Download the dataset from Kaggle and place adult.csv in the folder

# 3. Launch Jupyter
jupyter notebook adult_income_ml.ipynb
```

---

## 📦 Dependencies

| Library | Version |
|---------|---------|
| Python | 3.10+ |
| pandas | 1.5+ |
| numpy | 1.23+ |
| scikit-learn | 1.2+ |
| matplotlib | 3.6+ |
| seaborn | 0.12+ |
| jupyter | latest |

---

## 👤 Author

**Omar** — Computer Science Student  
Modern Academy for Engineering & Technology, Egypt  
📧 Connect on [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)  
🐙 [GitHub](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is licensed under the MIT License.

---

*Built as part of a Machine Learning course assignment — implementing models from mathematical foundations to production-ready code.*
