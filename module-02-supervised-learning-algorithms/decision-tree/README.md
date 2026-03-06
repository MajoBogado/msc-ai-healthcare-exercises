# 🧪 Decision Tree – Supervised Learning

## 📑 Table of Contents

1. [Objective](#1-objective)
2. [Library Used](#2-library-used)
3. [Dataset Used](#3-dataset-used)
4. [Steps Performed](#4-steps-performed)
5. [How to Run the System](#5-how-to-run-the-system)
6. [Why Decision Tree Is Appropriate Here](#6-why-decision-tree-is-appropriate-here)
7. [Key Lessons Learned and Conclusions](#7-key-lessons-learned-and-conclusions)

---

## 1. Objective 🎯 
The objective of this exercise is to explore **Decision Tree** classification in a healthcare context using a real-world clinical dataset (Stroke Prediction Dataset).

The goal is not only to train a model, but to:

* Understand how Decision Trees behave under class imbalance
* Observe overfitting and underfitting in practice
* Explore hyperparameter tuning using Grid Search
* Analyze feature importance
* Understand how Decision Trees estimate probabilities
* Perform end-to-end prediction on new patients

This exercise emphasizes model behavior, interpretation, and limitations, rather than optimizing for a single “best” model.

---

## 2. Library Used
This implementation uses:

**Scikit-learn (sklearn)**:

  * `DecisionTreeClassifier`

  * `GridSearchCV`

  * `Pipeline`

`pandas` for data manipulation

`numpy` for numerical operations

All preprocessing is handled inside a reusable Pipeline, ensuring:
* No data leakage
* Consistent transformation of training, testing, and new patient data

---

## 3. Dataset Used 📊 
We use the Stroke Prediction Dataset (clinical tabular dataset).

🔗 Official documentation + description:

[https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download)

Target variable:
`stroke`
* 0 → No stroke
* 1 → Stroke occurred

Important dataset characteristic: **highly imbalanced**.
* ~95% no stroke
* ~5% stroke

This imbalance strongly influences model behavior and evaluation metrics.

---

## 4. Steps Performed

### ✅ Step 1 – Load Dataset

Load Stroke Prediction Dataset via local CSV.

### ✅ Step 2 – Exploratory Data Analysis (EDA)
We examine:
* Dataset size
* Target distribution
* Missing values
* Features
* Data types

Key finding:
* Stroke cases are rare (~5%), creating a significant class imbalance problem.

### ✅ Step 3 – Train/Test Split
We split the data into **80% training** and **20% testing**. 

This prevents data leakage and evaluates how the model generalizes to unseen patients.

**Note**: The criteria for determining the proportion of the dataset to include in the train split vs. the test split in this exercise are arbitrary. Usually depends on variables like the size of the dataset. 

The dataset is split using stratification, ensuring:
* The same stroke/no-stroke proportion
* Both training and testing sets reflect real distribution

This is critical in imbalanced classification problems.

### ✅ Step 4 – Preprocessing - Imputation, Encoding categorical variables, Reusable for new patients

We construct a reusable preprocessing pipeline that:
* Imputes missing BMI values (median learned from training data)
* One-hot encodes categorical variables
* Preserves column alignment
* Prevents data leakage

The pipeline is reused for:
1. Training data
2. Testing data
3. New patient prediction

### ✅ Step 5 – Train and Evaluate baseline Decision Tree - max_depth=None, min_samples_leaf=1, class_weight=None

We train a default Decision Tree:
* No depth limit
* No class weighting

Result:
* Training metrics: perfect (accuracy = 1.0)
* Testing recall: very low

This demonstrates overfitting:
* The model memorizes training data but generalizes poorly.

### ✅ Step 6 – Constrained Decision Tree (max_depth=5, class_weight=None)

We restrict tree depth.

Result:
* Model predicts mostly the majority class (no stroke)
* Recall collapses to zero

This demonstrates underfitting and the dominance of class imbalance.

### ✅ Step 7 –  Constrained + Balanced Decision Tree (max_depth=5, class_weight=balanced)

We adjust class weights to compensate for imbalance.

Result:
* Recall increases significantly
* False positives increase
* Accuracy decreases

This demonstrates the precision–recall tradeoff.

In screening contexts, higher recall may be preferred.

### ✅ Step 8 –  Grid Search (Optimize F1 on TRAIN using Cross-Validation)

We perform systematic hyperparameter tuning using GridSearchCV.

**Note that Grid Search will take longer** because it’s evaluating many parameter combinations. That’s expected.

Optimized metric:
* F1-score (balance between precision and recall)

This step explores: `max_depth`, `min_samples_leaf`, `min_samples_split`, `class_weight`

The goal is not to find a “perfect” model, but to observe how hyperparameters affect:

* Generalization
* Recall
* Model complexity

### ✅ Step 9 –  Evaluate Grid-Search Best Model on TRAIN and TEST


### ✅ Step 10 –  Feature Importance Comparison (Baseline vs Grid Best)

We extract feature importances from:

* Baseline tree
* Grid-search optimized tree

Key observation:
* Baseline model distributes importance across many features.
* Regularized model concentrates importance (e.g., age becomes dominant).

Important clarification:
Decision Tree feature importance reflects impurity reduction,
not causality or clinical effect size.

### ✅ Step 11 –  New patient prediction (end-to-end) - High risk vs. Low risk

We create synthetic patients:
* High-risk profile
* Lower-risk profile

The full pipeline (preprocessing + model) predicts:
* Classification (0 / 1)
* Model-estimated stroke probability

This confirms:
* The system is reusable
* Raw patients can be safely passed into the trained pipeline

### ✅ Step 12 –  Probability-smoother Decision Tree (min_samples_leaf=50, balanced)

**A critical learning outcome, based on analyzing the outcome in Step 11**

Decision Tree probabilities are:
* The proportion of stroke cases in the training samples located in the same leaf node.

They are:
* Not calibrated
* Not smooth
* Often extreme (0.0 or 1.0 with small leaves)

This explains why some patients may receive:
* 0.0000 probability

This does NOT mean true clinical risk is zero.

To mitigate this, in this step, we explored:
* Increasing min_samples_leaf
* Forcing larger leaf sample sizes

Producing smoother probability estimates.

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

This will execute all steps (1-12)

---

## 6. Why Decision Tree is Appropriate Here
Decision Trees are appropriate because:
* They handle mixed numeric and categorical data.
* They are interpretable.
* They require minimal preprocessing.
* They allow inspection of feature importance.
* They demonstrate overfitting/underfitting clearly.

However, limitations include:
* Sensitivity to class imbalance
* Unstable splits with small datasets
* Non-calibrated probability outputs
* Piecewise constant decision boundaries

This makes them excellent for learning — but not always ideal for production without refinement.

---

## 7. Key Lessons Learned and Conclusions

This exercise demonstrates several important machine learning concepts:

**1. Class Imbalance Matters**
Accuracy alone is misleading in imbalanced datasets.

**2. Overfitting vs Underfitting**
* Unlimited depth → memorization
* Too constrained → majority-class prediction

**3. Hyperparameter Tuning Is Structured Exploration**
Grid search reveals how model complexity affects recall and precision.

**4. Feature Importance Is Model-Dependent**
Regularization changes which features dominate decisions..

**5. Decision Tree Probabilities Are Leaf Frequencies**
They are not calibrated clinical risks.

**6. Model Output ≠ Clinical Truth**
A model’s 0% probability does not mean true risk is zero.
It reflects training data structure and model assumptions.

This exercise reinforces that:
1. Machine learning performance is constrained by:
  * Data distribution
  * Feature informativeness
  * Dataset size
  * Model choice

2. There is no single “correct” configuration. The purpose of this exploration is to understand:
  * Model mechanics
  * Trade-offs
  * Limitations
  * Responsible interpretation in healthcare settings

---