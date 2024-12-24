# 📊 Classical Machine Learning Assistant

This repository contains Python scripts for processing data, generating machine learning models, and optimizing hyperparameters for classification, regression, and clustering tasks. The framework includes utilities for evaluation and exporting results for documentation purposes.

---

## 🗂️ Overview

1. **`data_processor.py`** 🧹 - Utilities for cleaning and transforming datasets.
2. **`evaluator.py`** 🧮 - Common evaluation metrics for models.
3. **`classifierGenerator.py`** 🤖 - Classification model generation and tuning.
4. **`cluster_generator.py`** 🧩 - Clustering model generation and tuning.
5. **`regressor_generator.py`** 📈 - Regression model generation and tuning.
6. **`master_generator.py`** 🏗️ - High-level generator for all tasks.

---

## 📜 Scripts Explanation

### 1. **`data_processor.py`** 🧹
Provides methods for handling missing data, dimensionality reduction, and feature selection.
- **Highlights**:
  - Fill missing values using modes or regression.
  - Perform dimensionality reduction (PCA, SVD, t-SNE).
  - Filter features with statistical tests (Chi-Square).

**Usage Example**:
```python
from data_processor import Data_processor
processed_df = Data_processor.fill_na_with_mode(df, column_name="column_name")
```

---

### 2. **`evaluator.py`** 🧮
Contains methods to evaluate models with metrics for regression, classification, and clustering tasks.
- **Highlights**:
  - Regression: MAE, MAPE, RMSE, Adjusted R².
  - Classification: Accuracy, Precision, Recall, F1, ROC AUC.
  - Clustering: Silhouette Score, Davies-Bouldin Score.

**Usage Example**:
```python
from evaluator import Evaluator
Evaluator.eval_regression(y_pred, y_true, bins=5)
```

---

### 3. **`classifierGenerator.py`** 🤖
Generates and tunes classification models using decision trees, random forests, and SVMs.
- **Highlights**:
  - Supports hyperparameter optimization with Optuna.
  - GPU acceleration with RAPIDS cuML (if available).

**Usage Example**:
```python
from classifierGenerator import ClassifierGenerator
clf_gen = ClassifierGenerator(dataset=df, target_column="target", use_cuml=True)
clf_gen.generate(n_trials=50)
clf_gen.save("classification_results")
```

---

### 4. **`cluster_generator.py`** 🧩
Generates and tunes clustering models using k-means, DBSCAN, and Gaussian Mixture Models.
- **Highlights**:
  - Optimizes parameters like the number of clusters or distance metrics.
  - Evaluates clustering with silhouette and homogeneity scores.

**Usage Example**:
```python
from cluster_generator import ClusterGenerator
cluster_gen = ClusterGenerator(dataset=df, use_cuml=False)
cluster_gen.generate(n_trials=30, ground_truth=ground_truth_labels)
cluster_gen.save("clustering_results")
```

---

### 5. **`regressor_generator.py`** 📈
Generates and tunes regression models using decision trees, random forests, SVMs, and linear regression.
- **Highlights**:
  - Optimizes hyperparameters with Optuna.
  - Evaluates models with detailed regression metrics.

**Usage Example**:
```python
from regressor_generator import RegressionGenerator
reg_gen = RegressionGenerator(X=df_features, y=df_target, use_cuml=False)
reg_gen.generate(n_trials=50)
reg_gen.save("regression_results")
```

---

### 6. **`master_generator.py`** 🏗️
A comprehensive generator for all tasks: classification, regression, and clustering.
- **Highlights**:
  - Combines all generators for an end-to-end ML pipeline.
  - Saves results in structured directories for documentation.

**Usage Example**:
```python
from master_generator import MasterGenerator
master_gen = MasterGenerator(X=df_features, y_categ=categ_target, y_numeric=num_target, name="experiment_name", n_tries=50)
master_gen.generate()
```

---

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

1. Import the necessary modules from the scripts.
2. Use `data_processor` to clean and preprocess your dataset.
3. Select a generator (`ClassifierGenerator`, `RegressionGenerator`, etc.) to create models.
4. Evaluate your results with `evaluator`.
5. Save outputs for documentation.