# 🧪 Support Vector Machines – Supervised Learning

## 📑 Table of Contents

1. [Objective](#1-objective)
2. [Library Used](#2-library-used)
3. [Dataset Used](#3-dataset-used)
4. [Steps Performed](#4-steps-performed)
5. [How to Run the System](#5-how-to-run-the-system)
6. [Why Support Vector Machines Is Appropriate Here](#6-why-support-vector-machines-is-appropriate-here)
7. [Key Lessons Learned and Conclusions](#7-key-lessons-learned-and-conclusions)

---

## 1. Objective 🎯 
The objective of this exercise is to understand how **Support Vector Machines (SVM)** work in a healthcare classification setting.

We apply:
* Linear SVM
* Linear SVM with hyperparameter tuning (C)
* Cost-sensitive Linear SVM (class weighting)
* RBF (Radial Basis Function) SVM

The goal is not only to train models, but to understand:
* When SVM makes sense
* How regularization affects false negatives
* How cost-sensitive learning improves clinical performance
* Whether nonlinear decision boundaries are necessary

---

## 2. Library Used
We used the following Python libraries:

* **Scikit-learn (sklearn)**: 
    * `SVC - C-Support Vector Classification.` for modeling.
    * `train_test_split` for data partitioning.
* **pandas**: For data structure handling.
* **numpy**: For numerical operations.

**Why Scikit-learn?**
* Includes preprocessing tools (StandardScaler)
* Includes evaluation metrics (confusion matrix, precision, recall, ROC-AUC)
* Widely used in both academic and applied machine learning

**SVC (C-Support Vector Classification) because:**
* It supports both linear and RBF kernels.
* It allows control over regularization (C).
* It allows cost-sensitive learning via class_weight.

---

## 3. Dataset Used 📊 
We used the Breast Cancer Wisconsin (Diagnostic) dataset, available directly from scikit-learn.

Characteristics:
* 569 patients
* 30 clinical features
* Binary classification target - we switched :
    * `1 → malignant`
    * `0 → benign`

🔗 Official documentation + description:
[https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)

**Target Encoding Adjustment**

In the original sklearn dataset:
* `0 → malignant`
* `1 → benign`

For this exercise, we intentionally re-labeled the target so that:
* `1 → malignant`
* `0 → benign`

This adjustment was made for clinical and evaluation consistency.

In scikit-learn, many classification metrics such as:

* Recall
* Precision
* F1-score

assume that the positive class is labeled as 1.

Since our primary clinical concern is detecting malignant tumors (cancer), we defined malignant as the positive class.

This ensures that:

* Recall directly measures cancer detection rate.
* False negatives correspond to missed malignant cases.
* Evaluation metrics reflect real clinical risk.

This relabeling does not change the data distribution or model behavior. It only ensures correct metric interpretation.

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

We loaded the data using `load_breast_cancer()`.

### ✅ Step 2 – Exploratory Data Analysis (EDA)
We examined:
* Dataset shape
* Class distribution: presence of benign vs. malignant in the dataset
* Missing values
* Feature scale snapshot

The analysis helped to identify that scaling is needed to center all features around 0 and have a variance of 1. SVM is sensitive to feature scale.

### ✅ Step 3 – Train/Test Split
We split the data into **80% training** and **20% testing**. 

This prevents data leakage and evaluates how the model generalizes to unseen patients.

**Note**: The criteria for determining the proportion of the dataset to include in the train split vs. the test split in this exercise are arbitrary. Usually depends on variables like the size of the dataset. 

We applied `stratify=diagnosis_labels` to ensure class proportions were preserved. This prevents evaluation bias.

### ✅ Step 4 – Feature Standardization

Used `StandardScaler` to center all features around 0, as explained in Step 2, and print the first patient, first 5 features to compare:
* the original value, against 
* the scaled value

### ✅ Step 5 – Train SVM Model

### ✅ Step 6 – Evaluate Model
Evaluated using:
* Accuracy
* Precision (Malignant detection)
* Recall (Malignant detection / Sensitivity)
* Recall (Benign detection)
* F1 Score
* Confusion Matrix using default Class-weighted: 1.0 (Default C = 1.0)

Special attention was given to false negatives (malignant predicted as benign)

### ✅ Step 7 – Linear SVM hyperparameter tuning & Cost-sensitive Linear SVM 
We tried hyperparameter tuning to prevent having a false negative. Even though the result of the performance metrics were good, in medicine, even one false negative is bad.

We analyze how changing the parameters also affected the performance metrics.

**Linear SVM hyperparameter tuning**
* Tested C = 10 and C = 100
* Observed overfitting behavior at high C

**Cost-sensitive Linear SVM**
* Used `class_weight = 2.0` to penalize missed cancers
* Improved malignant detection

### ✅ Step 8 – Compare previous results against SVM RBF
The RBF (Radial Basis Function) SVM allows the model to learn a non-linear decision boundary in the feature space.

If the relationship between clinical features and tumor diagnosis is not linearly separable, a nonlinear kernel may better capture complex patterns and improve malignant detection.

In this step we:
* Tested an RBF (Gaussian) kernel
* Compared its performance to linear models (Logistic Regression and Linear SVM)
* Evaluated whether allowing non-linear boundaries reduces false negatives

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

## 6. Why Support Vector Machines Is Appropriate Here
Support Vector Machines are suitable in this scenario because:

* The dataset has multiple numerical features.
* SVM performs well in high-dimensional spaces.
* Margin maximization provides robust decision boundaries.
* Class-weighting allows incorporating clinical cost asymmetry.
* RBF kernel allows modeling nonlinear relationships if necessary.

However, SVM does not naturally provide calibrated probabilities and may not always outperform logistic regression in linearly separable medical datasets. In fact, if you rung the previous exercise, that uses the same dataset, you will see that the performance of logistic Regression is better, considering the Confusion Matrix.

---

## 7. Key Lessons Learned and Conclusions

1. Linear SVM performs well but may miss more cancers than Logistic Regression.
2. Increasing C does not guarantee better malignant detection and may cause overfitting.
3. Cost-sensitive learning (class_weight) improves clinical relevance.
4. RBF SVM did not significantly outperform linear SVM in this dataset.
5. The dataset appears largely linearly separable.
6. Logistic Regression may be more appropriate when:
     * Probability outputs are required
     * Threshold tuning is important
     * Interpretability is preferred

Final insight:
Model choice in healthcare should prioritize clinical impact (false negatives) rather than overall accuracy.

---