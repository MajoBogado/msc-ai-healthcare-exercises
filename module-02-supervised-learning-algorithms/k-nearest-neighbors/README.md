# 🧪 K Nearest Neighbors – Supervised Learning

## 📑 Table of Contents

1. [Objective](#1-objective)
2. [Library Used](#2-library-used)
3. [Dataset Used](#3-dataset-used)
4. [Steps Performed](#4-steps-performed)
5. [How to Run the System](#5-how-to-run-the-system)
6. [Why KNN Is Appropriate Here](#6-why-KNN-is-appropriate-here)
7. [Key Lessons Learned and Conclusions](#7-key-lessons-learned-and-conclusions)

---

## 1. Objective 🎯 
The objective of this exercise is to understand how **K-Nearest Neighbors (KNN)** works in a healthcare classification setting, using a Heart Disease dataset.

The goal is not only to train a model, but to understand:
* What K means in KNN and how it affects predictions
* Why feature scaling is critical for distance-based algorithms
* Why feature encoding (numeric vs one-hot) changes the geometry of “closeness”
* How tuning changes model behavior (bias/variance tradeoff)
* What weights means (uniform vs distance) and why it matters clinically

Your module describes k-NN as a distance-based method that classifies a new observation using its k closest neighbors, making distance and the choice of k central to performance.

---

## 2. Library Used
We used the following Python libraries:

* **Scikit-learn (sklearn)**:

* `fetch_openml` to load the dataset
* `train_test_split` for partitioning
* `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `Pipeline`, `ColumnTransformer` for preprocessing
* `KNeighborsClassifier` for KNN classification
* confusion_matrix, precision, recall, f1, accuracy for evaluation

* **pandas**: For data structure handling. `DataFrames` for readable feature handling and column-name preservation

Why scikit-learn?
It provides a consistent and well-tested ML workflow: preprocessing → training → evaluation.

---

## 3. Dataset Used 📊 
We used the Heart Disease (Cleveland) dataset (loaded via OpenML).

🔗 Official documentation + description:

[https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

🔗 OpenML access: 

[https://www.openml.org/search?type=data&sort=runs&id=194&status=active](https://www.openml.org/search?type=data&sort=runs&id=194&status=active)

Target column in the dataset:

num is typically in {0,1,2,3,4} where:
* 0 = no heart disease
* 1–4 = heart disease present

For this exercise, we convert it into a binary target:
* target_heart_disease = 1 → disease present
* target_heart_disease = 0 → no disease

This makes evaluation easier and clinically meaningful: recall becomes “disease detection rate”.

---

## 4. Steps Performed

### ✅ Step 1 – Load Dataset

Load Heart Disease (Cleveland) data via OpenML
Convert all columns to numeric (? becomes NaN)
Convert target num into binary target_heart_disease (1 = disease, 0 = no disease)

### ✅ Step 2 – Exploratory Data Analysis (EDA)
We display:
* Dataset shape
* Target distribution
* First rows of features (clean printing, no pandas dtype noise)

This helps us recognize:
* Mixed feature types (continuous + categorical encoded as integers)
* Potential missing values (NaNs after coercion)

### ✅ Step 3 – Train/Test Split
We split the data into **80% training** and **20% testing**. 

This prevents data leakage and evaluates how the model generalizes to unseen patients.

**Note**: The criteria for determining the proportion of the dataset to include in the train split vs. the test split in this exercise are arbitrary. Usually depends on variables like the size of the dataset. 

We applied stratification:
* Preserves the class distribution (disease vs no disease) across train and test.
* Stabilizes evaluation and reduces bias in the test set.

### ✅ Step 4 – Train and Evaluate numeric Pipeline (baseline k=5)

Pipeline A: Numeric strategy
* Impute missing values (median)
* Standardize all features
* Train KNN with baseline k=5 and weights="uniform"
* Evaluate using confusion matrix + metrics

Why scaling?
KNN uses distances; without scaling, large-range variables dominate “closeness”.

### ✅ Step 5 – Train and Evaluate One-Hot Pipeline (baseline k=5)

Pipeline B: One-hot strategy
* Split features into:
  * continuous (scaled)
  * categorical (one-hot encoded)
* Impute:
  * continuous → median
  * categorical → most frequent
* One-hot encode categorical variables (handle_unknown="ignore")
* Train KNN with baseline k=5 and weights="uniform"
* Evaluate using confusion matrix + metrics

Why do this?
KNN is distance-based; treating categories as numeric can add artificial ordering (e.g., cp=4 “farther” than cp=1). One-hot encoding changes the geometry of distances.

### ✅ Step 6 – Hyperparameter Tuning: Modify k from 1..15
This is the key KNN hyperparameter.

**What is K?**
**K** is the number of neighbors used to vote for a prediction:
* Small K → more sensitive to noise (lower bias, higher variance)
* Large K → smoother boundary (higher bias, lower variance)

Your module highlights that choosing k can significantly affect performance, so tuning is common.

We tune:
* k = 1..15 for both pipelines
and track:
* accuracy, precision, recall, f1

We observed:
* Numeric pipeline achieved recall = 1.0 for some k values (including k=9).
* One-hot pipeline achieved best overall balance (best F1) at a higher k.

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

This will execute all steps (1-6)

---

## 6. Why KNN Is Appropriate Here
KNN makes sense for this exercise because:
* The dataset is small-to-medium sized and tabular (typical clinical ML shape).
* The model is easy to reason about: predictions come from “similar patients”.
* It forces correct ML engineering decisions:
  * scaling
  * missing value handling
  * encoding choices
  * hyperparameter tuning (k)

This aligns with the conceptual description of k-NN in the module: it’s a non-parametric method that classifies based on neighborhood distances, and the value of k can strongly affect results.

Limitations to remember:
KNN can degrade with higher dimensionality and larger datasets (distance becomes less informative).

---

## 7. Key Lessons Learned and Conclusions

**1. K is a hyperparameter, not learned**
Changing K changes bias/variance and can meaningfully change recall.

**2. Scaling is not optional for KNN**
KNN is distance-based; unscaled features distort “closeness”.

**3. Encoding changes the geometry**
Numeric encoding vs one-hot encoding can materially change results.

**4. Different clinical priorities lead to different “best” models**
* If the priority is avoiding missed disease cases (low FN), prefer settings that maximize recall.
* If the priority is overall balance, choose higher F1 / precision tradeoffs.

**5. Distance weighting**
Distance Weighting Was Tested but Removed from Final Configuration
We evaluated both:

`weights="uniform"`

`weights="distance"`

Using the best tuned k values:
* Numeric pipeline → k = 9
* One-Hot pipeline → k = 11

However, results were identical across all evaluation metrics:
* Accuracy
* Precision
* Recall
* F1-score

This indicates that:
For this dataset and this train/test split, the neighborhood class composition was already stable, distance weighting did not alter any final predictions.

In KNN, distance weighting only changes predictions when:
* neighbor votes are close (e.g., 5 vs 4)
* And closer neighbors belong to the minority class strongly enough to flip the decision.

In our case, this scenario did not occur.

Therefore:
Distance weighting was experimentally validated but excluded from the final model configuration, as it provided no measurable benefit in this setting.

**🔬 Future Work**

Although distance weighting did not impact performance in this experiment, it remains a theoretically meaningful hyperparameter in KNN.

Future work will include:
* Testing this parameter on different healthcare datasets
* Evaluating scenarios with higher class imbalance
* Exploring cases where local neighborhood votes are less stable

The objective of future experiments is to better understand under which data distributions and geometric conditions distance weighting meaningfully affects model performance.

---