# 🧪 Linear Regression – Supervised Learning

## 📑 Table of Contents

1. [Objective](#1-objective)
2. [Library Used](#2-library-used)
3. [Dataset Used](#3-dataset-used)
4. [Steps Performed](#4-steps-performed)
5. [How to Run the System](#5-how-to-run-the-system)
6. [Why Linear Regression Is Appropriate Here](#6-why-linear-regression-is-appropriate-here)
7. [There was a step 9: Why Step 9 (Raw Patient -> Model) Could Not Be Executed](#7-there-was-a-step-9-why-step-9-raw-patient---model-could-not-be-executed)
8. [How To Implement Step 9 Properly (Future Work)](#8-how-to-implement-step-9-properly-future-work)
9. [Key Lessons Learned and Conclusions](#9-key-lessons-learned-and-conclusions)

---

## 1. Objective 🎯 
The goal of this exercise was to understand how to implement and interpret a Linear Regression model in a healthcare context using the **Scikit-learn (sklearn)** library in Python.

Specifically, we aimed to:
* Learn how to load and explore a dataset.
* Split data into training and test sets.
* Train a Linear Regression model.
* Evaluate its performance.
* Interpret the model at both global (feature sensitivity) and local (patient-level) levels.
* Understand the limitations of the dataset for real-world deployment.

---

## 2. Library Used
We used the following Python libraries:

* **Scikit-learn (sklearn)**: 
    * `LinearRegression` for modeling.
    * `train_test_split` for data partitioning.
    * `mean_squared_error` and `r2_score` for evaluation.
* **pandas**: For data structure handling.
* **numpy**: For numerical operations.
* **matplotlib**: For data visualization.

**Why Scikit-learn?**

It is the standard ML library in Python, providing reliable, production-grade implementations and enforcing a clean separation between training and inference.

---

## 3. Dataset Used 📊 
We used the built-in Scikit-learn diabetes dataset, commonly used for linear regression:

```python
from sklearn.datasets import load_diabetes
```

🔗 Official documentation + description:
[https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)

**Why this dataset?**
* It is healthcare-related.
* The target variable is continuous (disease progression).
* It is a classic benchmarking dataset for regression.

**Important Note:**

The dataset features are: 
* already mean-centered and 
* already scaled (not in original clinical units). 

Therefore:
* Raw units (e.g., age in years, BMI in $kg/m^2) are not available.
* The dataset is intended for algorithm benchmarking, not direct clinical deployment.

---

## 4. Steps Performed

### ✅ Step 1 – Load Dataset

Access structured, clean healthcare data to begin the supervised regression workflow.

We loaded the data using `load_diabetes()`.

### ✅ Step 2 – Exploratory Data Analysis (EDA)
We examined patient counts, feature names, and statistical summaries (`describe()`). 

We identify that features were already standardized and confirm data integrity.

### ✅ Step 3 – Train/Test Split
We split the data into **80% training** and **20% testing**. 

This prevents data leakage and evaluates how the model generalizes to unseen patients.

**Note**: The criteria for determining the proportion of the dataset to include in the train split vs. the test split in this exercise are arbitrary. Usually depends on variables like the size of the dataset. 

### ✅ Step 4 – Train Linear Regression Model
We trained the `LinearRegression()` model using the training split of the data defined in the previous step to learn the linear relationships between features and disease progression.

### ✅ Step 5 – Generate Predictions

We apply learned relationships to unseen data (using the test split of the data defined in step 3) to simulate real-world inference.

We used `model.predict(patient_features_test)`.

### ✅ Step 6 – Evaluate Model

Metrics used:

* MSE (Mean Squared Error)

* RMSE (Root Mean Squared Error)

* R² (Coefficient of Determination)

Why:

* MSE → mathematical loss metric

* RMSE → interpretable error in outcome units

* R² → proportion of variance explained

**Result:** R² approx 0.45 indicated that the model explains ~45% of the variability in disease progression.

As part of the same step, we evaluated the sensitivity of the features and defined **sensitivity thresholds** (based on abs(coefficients))
* LOW  <= 183.06
* HIGH >= 536.34

### ✅ Step 7 – Model Interpretation - Interpret Model Coefficients
We evaluated feature sensitivity grouping to identify, based on the sensitivity thresholds defined in the previous step, which features have higger sensitivy than others.

### ✅ Step 8 – Patient-level interpretation (clinical interpretation)
We implemented feature sensitivity grouping and patient-level “what-if” analysis.
Raw coefficients are not intuitive for clinicians. We focus on how changing one feature affects the specific patient's prediction, based on the sensitivity thresholds defined in the previous steps.

* We changed one feature with high sensitivity and evaluated the prediction for that patient based on the original value versus the prediction with the modified value.
* We changed one feature with low sensitivity and evaluated the prediction for that patient based on the original value versus the prediction with the modified value.

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

## 6. Why Linear Regression Is Appropriate Here
Linear regression is suitable because:

* The target variable (disease progression score) is continuous

* Relationships are assumed to be approximately linear

* The model is transparent and interpretable

* It serves as a strong baseline model

In healthcare, linear regression is valuable because it is explainable (it is statistically grounded) and therefore, it's easy to audit.

We assume approximate linearity as a first-order model and evaluate performance. If performance is insufficient or residual diagnostics suggest nonlinear patterns, more flexible models should be considered.

---

## 7. There was a step 9: Why Step 9 (Raw Patient -> Model) Could Not Be Executed
We initially attempted to input raw clinical values (e.g., Age = 23 years) with the aim of adding a new patient into the model, considering creating a function to go from raw to scaled parameters. 

This was not possible because:
* The dataset features are **already preprocessed** *-AND-*
* The original scaling parameters (means and standard deviations) are not available in the library *-AND-*
* The transformation used in the original dataset cannot be reversed mathematically without those parameters.

Therefore, we couldn't create the function we originally planned to.

---

## 8. How To Implement Step 9 Properly (Future Work)
To correctly perform **Raw Patient → Standardized Features → Prediction**, one must follow this architecture:


🔵 Step A – Use a Dataset with Raw Clinical Units

The dataset must include:

* Age in years
* BMI in kg/m²
* BP in mmHg
* Lab values in mg/dL

🔵 Step B – Fit a Scaler on Training Data

Using:
 
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
```

🔵 Step C – Transform Training Data

Using:

```python
X_train_scaled = scaler.transform(X_train)
```

🔵 Step D – Train Model on Scaled Data

```python
model.fit(X_train_scaled, y_train)
```

🔵 Step E – Inference on New Raw Patient

```python
new_patient_scaled = scaler.transform(new_patient_raw)
prediction = model.predict(new_patient_scaled)
```

This ensures:

* No data leakage
* Correct scaling
* Proper deployment pipeline

---

## 9. Key Lessons Learned and Conclusions
This exercise successfully demonstrated the end-to-end supervised learning workflow. The model provides a population-based severity estimate for diabetes progression. While effective for benchmarking, we concluded that clinical deployment requires a custom preprocessing pipeline (Scaling) that the built-in dataset does not provide.

* **Data Readiness:** Not all datasets are deployment-ready; preprocessed data often lacks the metadata (scaling parameters) needed for real-world inference.
* **Contextual Interpretation:** Model interpretation requires clinical context and patient-level analysis, not just raw mathematical coefficients.
* **Validation Integrity:** Train/test separation is non-negotiable for ensuring reliable performance in healthcare AI.

---