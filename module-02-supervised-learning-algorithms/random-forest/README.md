# 🧪 Random Forest – Supervised Learning

## 📑 Table of Contents

1. [Objective](#1-objective)
2. [Library Used](#2-library-used)
3. [Dataset Used](#3-dataset-used)
4. [Steps Performed](#4-steps-performed)
5. [How to Run the System](#5-how-to-run-the-system)
6. [Why Random Forest Is Appropriate Here](#6-why-random-forest-is-appropriate-here)
7. [Key Lessons Learned and Conclusions](#7-key-lessons-learned-and-conclusions)

---

## 1. Objective 🎯 
The objective of this exercise is to explore Random Forest classification in a healthcare context using a real-world clinical dataset (Stroke Prediction Dataset).

The goal is not only to train a model, but to:

* Understand how ensemble methods improve over single Decision Trees
* Explore the impact of class imbalance in healthcare datasets
* Observe how hyperparameters influence model behavior
* Analyze Out-of-Bag (OOB) error estimation
* Study threshold tuning for clinical decision making
* Compare Random Forest performance with Decision Trees

This exercise focuses on understanding model behavior and trade-offs, rather than simply maximizing a performance metric.

---

## 2. Library Used
This implementation uses:

**Scikit-learn (sklearn)**:

  * `RandomForestClassifier`

  * `ColumnTransformer`

  * `Pipeline`

Other libraries:

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

This imbalance strongly affects:
* model training
* evaluation metrics
* threshold selection

For this reason, several experiments in this exercise explore class weighting and threshold tuning.

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

### ✅ Step 5 – Train and Evaluate Baseline Random Forest (Pipeline + Train/Test evaluation)  - max_depth=None, min_samples_leaf=1, class_weight=None

A baseline Random Forest model is trained using default-like parameters:

Example configuration:

`number_of_trees = 200`

`max_depth = None`

`min_samples_leaf = 1`

`class_weight = None`

Observation:

The model achieves perfect training performance, which indicates overfitting.

On the test set, the model shows high accuracy but very poor recall for the stroke class due to class imbalance.

### ✅ Step 6 – Random Forest (Pipeline + Train/Test evaluation) — max_depth=4, min_samples_leaf=10, class_weight=None

To reduce overfitting, the model complexity is constrained.

Example configuration:

`max_depth = 4`

`min_samples_leaf = 10`

This forces trees to be simpler and improves generalization stability.

However, the model still struggles to detect stroke cases because the dataset is highly imbalanced.

### ✅ Step 7 –  Random Forest with class_weight='balanced' (Baseline flexibility)

To address imbalance, the model is retrained using:

`class_weight = "balanced"`

This increases the importance of the minority class during training.

Effect:

* recall for stroke detection increases
* false positives increase

This demonstrates the classic precision-recall trade-off.

### ✅ Step 8 –  Random Forest with class_weight='balanced' (Constrained forest)

The best performing configuration combines:

* constrained trees
* balanced class weights

This configuration improves stroke detection while maintaining reasonable stability.

### ✅ Step 9 –  Threshold tuning (using Balanced Constrained Random Forest probabilities)

Instead of relying on the default classification threshold (0.5), different probability thresholds are evaluated.

Example thresholds tested:

`0.90, 0.80, 0.70, 0.60, 0.55, 0.50, ...`

This step demonstrates that the model outputs probabilities, not decisions.

The final classification depends on the threshold chosen by the practitioner.

For example:

Threshold	Recall	Precision
0.50	higher recall	lower precision
0.60	lower recall	higher precision

This trade-off is particularly important in clinical risk prediction systems.


### ✅ Extra: Out-of-Bag (OOB) Error Estimation - added in steps 5 and 6

Random Forest provides an internal validation mechanism called Out-of-Bag (OOB) error estimation.

Random Forest trains each tree using bootstrap sampling. This means that for every tree:
* A random sample of the training dataset is drawn with replacement.
* Some observations are selected multiple times.
* Some observations are not selected at all for that tree.
* Those unused observations are called Out-of-Bag samples.

Example:

If the training dataset contains 1000 patients, a bootstrap sample may select:

`~630 unique patients`

`~370 patients left out`

Those **370 patients** become the **Out-of-Bag validation set for that tree**.

The process works as follows:

1. Each tree is trained on its bootstrap sample.

2. For each training observation, predictions are made using only the trees where that observation was not included in the bootstrap sample.

3. These predictions are aggregated to compute the OOB score.

This allows Random Forest to estimate model performance without using the external test set.

Important clarification:

Although the OOB estimate uses the training dataset, it is not evaluating the model on the same data used to train each tree. Each observation is evaluated using only the subset of trees that did not see that observation during training.

In this sense, OOB estimation behaves similarly to cross-validation, but it is built directly into the Random Forest training process.

Advantages of OOB estimation:
* Provides a quick internal estimate of model performance
* Requires no additional validation dataset
* Helps detect overfitting early during training

However, **OOB estimation should still be complemented with evaluation on an independent test set**, as done in this exercise.

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

This will execute all steps (1-9)

---

## 6. Why Random Forest is Appropriate Here

**A. Handles nonlinear relationships**

* Clinical variables often interact in complex ways that linear models cannot capture.
* Random Forest can model these nonlinear interactions.

**B. Robust to noisy variables**

* Medical datasets frequently contain noisy or partially informative features.
* Random Forest averages many trees, reducing variance and improving robustness.

**C. Works well with tabular clinical data**
* Random Forest is one of the most reliable baseline algorithms for structured healthcare datasets.

**D. Provides feature importance**
* The model can estimate which clinical variables contribute most to predictions, supporting interpretability.

---

## 7. Key Lessons Learned and Conclusions

This exercise demonstrates several important machine learning concepts:

**1. Accuracy is misleading for imbalanced datasets**

Because stroke cases are rare (~5%), a model that predicts no stroke for everyone would still achieve high accuracy.

Therefore, metrics such as recall, precision, and F1-score are more informative.

**2. Ensemble models improve over single Decision Trees**

Random Forest combines many trees to reduce variance.
Compared to a single Decision Tree, Random Forest typically provides:
* better recall
* more stable predictions
* better generalization

**3. Handling class imbalance is essential**

Without class weighting, the model largely ignores the minority class.

Using:

`class_weight = "balanced"`

encourages the model to detect stroke cases.

**4. Threshold selection determines clinical behavior**
The model outputs probabilities, not final decisions.

Adjusting the classification threshold allows practitioners to choose between:
* fewer missed strokes (higher recall)
* fewer false alarms (higher precision)

This decision depends on the clinical context and available resources.

**5. Machine learning models reflect data limitations**
Even with advanced models, stroke prediction remains difficult.

The model shows moderate separation ability, indicating that:

* available features contain useful signal
* but stroke risk is influenced by many factors not captured in the dataset


**Overall**

Random Forest improves predictive performance compared to a single Decision Tree, but real-world healthcare prediction still requires:
* careful metric selection
* threshold tuning
* clinical interpretation of model outputs.

---