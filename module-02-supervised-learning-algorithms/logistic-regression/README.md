# 🧪 Logistic Regression – Supervised Learning

## 📑 Table of Contents

1. [Objective](#1-objective)
2. [Library Used](#2-library-used)
3. [Dataset Used](#3-dataset-used)
4. [Steps Performed](#4-steps-performed)
5. [How to Run the System](#5-how-to-run-the-system)
6. [Why Logistic Regression Is Appropriate Here](#6-why-logistic-regression-is-appropriate-here)
7. [Key Lessons Learned and Conclusions](#7-key-lessons-learned-and-conclusions)

---

## 1. Objective 🎯 
The objective of this project is to implement a complete supervised classification pipeline using **Logistic Regression** in a healthcare context.

Specifically, we aim to:
* Classify breast tumors as malignant or benign
* Understand how logistic regression works from a programming perspective
* Build a clean and modular pipeline (loading, splitting, scaling, training, evaluation, deployment)
* Implement threshold tuning for clinical decision-making
* Simulate prediction for a new patient using raw input features

The goal is not just to train a model, but to understand the implications of using this model in real medical decision-making.

---

## 2. Library Used
We used the following Python libraries:

* **Scikit-learn (sklearn)**: 
    * `LogisticRegression` for modeling.
    * `train_test_split` for data partitioning.
* **pandas**: For data structure handling.
* **numpy**: For numerical operations.

**Why Scikit-learn?**
* Provides a robust and standardized implementation of Logistic Regression
* Includes preprocessing tools (StandardScaler)
* Includes evaluation metrics (confusion matrix, precision, recall, ROC-AUC)
* Widely used in both academic and applied machine learning

---

## 3. Dataset Used 📊 
We used the Breast Cancer Wisconsin (Diagnostic) dataset, available directly from scikit-learn.

Characteristics:
* 569 patients
* 30 clinical features
* Binary classification target:
    * `0 → malignant`
    * `1 → benign`

🔗 Official documentation + description:
[https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)

**Why this dataset?**

The features are numeric measurements extracted from digitized images of breast mass cell nuclei (e.g., mean radius, mean texture, mean perimeter, etc.).

This dataset is appropriate because:
* It represents a real healthcare classification task.
* It contains raw numeric features (not pre-scaled).
* It allows proper demonstration of scaling and deployment workflows.
* It has moderate class imbalance (~63% benign, ~37% malignant).

---

## 4. Steps Performed

### ✅ Step 1 – Load Dataset

Loaded dataset using scikit-learn.

Converted to pandas DataFrame for feature name preservation.

We loaded the data using `load_breast_cancer()` to explore:
* Dataset shape
* Diagnosis categories
* Printed the first 3 feature names

### ✅ Step 2 – Exploratory Data Analysis (EDA)
We examined:
* Data set overview: number of patients, number of clinical features
* Class distribution: presence of benign vs. malignant in the dataset
* Feature scale snapshot

The analysis helped to identify that scaling is needed to center all features around 0 and have a variance of 1. Logistic Regression is sensitive to feature scale.

### ✅ Step 3 – Train/Test Split
We split the data into **80% training** and **20% testing**. 

This prevents data leakage and evaluates how the model generalizes to unseen patients.

**Note**: The criteria for determining the proportion of the dataset to include in the train split vs. the test split in this exercise are arbitrary. Usually depends on variables like the size of the dataset. 

We applied `stratify=diagnosis_labels` to ensure class proportions were preserved. This prevents evaluation bias.

### ✅ Step 4 – Feature Standardization

Used `StandardScaler` to center all features around 0, as explained in Step 2, and print the first patient, first 5 features to compare:
* the original value, against 
* the scaled value

### ✅ Step 5 – Train Logistic Regression Model

### ✅ Step 6 – Evaluate Model
Evaluated using:
* Accuracy
* Precision (Malignant detection)
* Recall (Malignant detection / Sensitivity)
* Recall (Benign detection)
* F1 Score
* ROC-AUC
* Confusion Matrix using default threshold (0.5)

Special attention was given to false negatives (malignant predicted as benign)

### ✅ Step 7 – Threshold Tuning
We tried tuning the Threshold sensitivity to prevent having a false negative. Even though the result of the performance metrics were good, in medicine, even one false negative is bad.

We analyze how changing the threshold affected the performance metrics.

### ✅ Step 8 – New Patient Prediction
* Created a synthetic patient using realistic feature values
* Applied trained scaler
* Predicted malignancy probability

Printed:
* Probability of malignancy
* Risk category
* Model sensitivity
* Model specificity

This simulates real-world deployment logic.

---

## 5. How to Run the System ▶️

Before executing:

Make sure you have pandas
```bash
python3 -m pip install pandas
```

Install the library to access the data set
```bash
python3 -m pip install scikit-learn
```

From the module directory:

```bash
python3 main.py
```

This will execute all steps (1-8)

---

## 6. Why Logistic Regression Is Appropriate Here
Logistic Regression is suitable because:

* The problem is binary classification.
* Outputs are probabilistic (critical in healthcare).
* It is interpretable (coefficients relate features to risk).
* It performs well on linearly separable medical datasets.
* It provides stable, calibrated probabilities.

---

## 7. Key Lessons Learned and Conclusions

**1. Threshold Defines Clinical Behavior**

The model does not decide alone.

The threshold determines:
* Sensitivity
* Specificity
* False negative rate
* False positive rate
* Changing threshold changes clinical risk tolerance.

**2. Accuracy Alone Is Not Enough**

In medical systems:
* Missing malignant cases is more dangerous than false alarms.
* Sensitivity is often more important than overall accuracy.

**3. Models Do Not Provide Certainty**

They provide probabilities.

Clinical systems must:
* Report risk
* Report sensitivity and specificity
* Be interpreted alongside medical expertise

**4. Logistic Regression Is Powerful but Not Perfect**

Even with ROC-AUC ≈ 0.99:

* There will still be false negatives.
* There will still be uncertainty.
* Biological data is not perfectly separable.

Machine learning in healthcare is about risk management, not perfection.

---