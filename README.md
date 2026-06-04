# Classification with Logistic Regression

## Objective

Build a binary classification model using Logistic Regression and evaluate its performance using various classification metrics.

## Tools & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Dataset

The project uses the **Healthcare Dataset** (`healthcare_dataset.csv`).

### Target Variable

The `Test Results` column is converted into a binary target:

* Normal → 0
* Abnormal → 1

Rows containing `Inconclusive` results are removed to create a binary classification problem.

## Project Workflow

### 1. Data Preprocessing

* Load dataset using Pandas.
* Remove unnecessary columns.
* Handle categorical features using Label Encoding.
* Create a binary target variable.

### 2. Train-Test Split

* Split dataset into training and testing sets.
* Use an 80:20 ratio.

### 3. Feature Standardization

* Apply StandardScaler to normalize feature values.
* Improve Logistic Regression performance.

### 4. Model Training

* Train a Logistic Regression model using Scikit-learn.

### 5. Model Evaluation

The model is evaluated using:

* Confusion Matrix
* Precision Score
* Recall Score
* Classification Report
* ROC Curve
* ROC-AUC Score

### 6. Threshold Tuning

* Default threshold: 0.5
* Custom threshold: 0.3
* Compare Precision and Recall after threshold adjustment.

### 7. Sigmoid Function

Logistic Regression uses the Sigmoid Function to convert predictions into probabilities.

Formula:

P(y=1) = 1 / (1 + e^(-z))

The sigmoid function maps any real-valued number into a value between 0 and 1.

## Results

This project demonstrates:

* Binary Classification using Logistic Regression
* Feature Scaling
* Model Evaluation using Classification Metrics
* ROC-AUC Analysis
* Decision Threshold Tuning
* Sigmoid Function Visualization

## How to Run

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the project

```bash
python logistic_regression.py
```

## Output

The project generates:

* Confusion Matrix
* ROC Curve
* Classification Report
* Precision Score
* Recall Score
* ROC-AUC Score
* Sigmoid Function Plot

## Author

Shreya Sinha