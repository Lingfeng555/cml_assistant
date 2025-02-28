from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    mean_absolute_error,
    normalized_mutual_info_score, 
    r2_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_auc_score,
    silhouette_score,
    v_measure_score
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class Evaluator:
    """
    A utility class for evaluating predictions from regression, classification, and ordinal classification models.

    Description:
        - This class provides static methods to evaluate model predictions using common performance metrics.
        - It is designed to handle 1-dimensional `np.array` inputs for both predictions and ground truth (`X` and `y`).
        - Includes functionality to save evaluation summaries into `.tex` files for reporting purposes.

    Main Methods:
        1. `eval_regression`: Evaluates regression models using metrics like MAE and R².
        2. `eval_classification`: Evaluates classification models using metrics like accuracy, precision, recall, F1-score, and confusion matrix.
        3. `eval_ordinal_classification`: Evaluates ordinal classification models with a focus on specialized metrics for ordinal data.

    Additional Features:
        - If `regressor_name` or `classifier_name` is provided as a parameter, the evaluation is registered internally for further processing or saving.
        - The `save()` method allows exporting all evaluations into a LaTeX `.tex` file for inclusion in scientific reports or papers.

    Requirements:
        1. **Python Libraries**:
            - `sklearn`: For evaluation metrics like `mean_absolute_error`, `r2_score`, `accuracy_score`, etc.
            - `numpy`: For handling arrays and numerical computations.
            - `pandas`: For managing evaluation results in tabular format.
            - `matplotlib`: For visualizing metrics or results if needed.
            - `os`: For file management when saving `.tex` files.

        2. **Inputs**:
            - All `X` (predicted values) and `y` (true values) must be 1-dimensional `np.array`.

        3. **Outputs**:
            - Methods return evaluation metrics as dictionaries or DataFrames for easy integration into other workflows.

    Intended Usage:
        - Use the main evaluation methods (`eval_regression`, `eval_classification`, and `eval_ordinal_classification`) to compute relevant metrics.
        - Store results with optional model names to track performance.
        - Export evaluation summaries for documentation or reporting using the `save()` method.
    """

    regression_register = {"Algorithm": [], 
                           "mae": [], 
                           "mse": [], 
                           "mape": [] ,
                           "r2": [], 
                           "error_mean": [], 
                           "error_std_dev": [], 
                           "adjuste_r2": []}
    classification_register = {"Algorithm": [], 
                               "accuracy": [], 
                               "precision": [], 
                               "recall": [] ,
                               "f1": [], 
                               "roc_auc": []}
    clustering_register = {"Algorithm": [], 
                           "silhouette_score": [], 
                           "calinski_harabasz_score": [], 
                           "davies_bouldin_score": [], 
                           "adjusted_rand_score": [],
                           "normalized_mutual_info_score": [], 
                           "homogeneity_score": [],
                           "completeness_score": [],
                           "v_measure_score": []}

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def mean_squared_error(y_true, y_pred, squared=True):
        mse = np.mean((y_true - y_pred) ** 2)
        if not squared:
            return np.sqrt(mse)
        return mse

    @staticmethod
    def calculate_statistics(arr):
        mean = np.mean(arr)
        variance = np.var(arr)
        std_dev = np.std(arr)
        return mean, variance, std_dev
    
    @staticmethod
    def equal_depth_binning(arr: np.array):
        values, counts = np.unique(arr, return_counts=True)
        counts_dict = dict(zip(values, counts))
        return counts_dict
    
    @staticmethod
    def adjustedr2(y_true, y_pred, n_features):
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
        return adjusted_r2
    
    @staticmethod
    def regression_error_distribution(y_pred: np.array, y_true: np.array,  bins: int, plot:bool, precision: int = 4):
        lower_limit = min(y_true)
        upper_limit = max(y_true)

        data = pd.DataFrame({"pred": y_pred, "true": y_true, "diff": np.abs(y_pred-y_true)})
        
        bin_depth = (upper_limit - lower_limit)/bins

        result = {"bin_label":[] ,"mean": [], "variance": [], "std_dev": [], "max_error": [], "min_error": [], "n_sample":[]}

        for i in range(1, bins):
            errors = data["diff"][(data["true"] <= bin_depth * i) & (data["true"] > bin_depth * (i - 1))].to_numpy()

            if len(errors) == 0:
                continue
            
            mean, variance, std_dev = Evaluator.calculate_statistics(errors)
            result["mean"].append(round(mean, precision))
            result["variance"].append(round(variance, precision))
            result["std_dev"].append(round(std_dev, precision))
            
            result["max_error"].append(round(np.max(errors), precision))
            result["min_error"].append(round(np.min(errors), precision))

            result["bin_label"].append(f"({round(bin_depth*(i-1), precision)}, {round(bin_depth*i, precision)}]")
            result["n_sample"].append(np.size(errors))

        result = pd.DataFrame(result)
        
        if plot : 
            colnames = result.columns.to_list()
            colnames.remove("bin_label")

            for col in colnames:
                Evaluator.plot_bar_chart_key_value(keys = result["bin_label"], values=result[col], title=col, xlabel=col, ylabel="bin_label")

        return result

    @staticmethod
    def plot_bar_chart_key_value(keys: list, values: list, title="Bar Chart", xlabel="Keys", ylabel="Values"):
        plt.figure(figsize=(10, 5))  # Set the figure size
        plt.bar(keys, values, color='black')  # Create a bar chart
        plt.xlabel(xlabel)  # Set the label for the x-axis
        plt.ylabel(ylabel)  # Set the label for the y-axis
        plt.title(title)  # Set the title of the chart
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
        plt.tight_layout()  # Adjust layout to not cut off elements
        plt.show()

    @staticmethod
    def plot_bar_chart(data: dict, title="Bar Chart", xlabel="Keys", ylabel="Values"):
        keys = list(data.keys())
        values = list(data.values())

        plt.figure(figsize=(10, 5))  # Set the figure size
        plt.bar(keys, values, color='blue')  # Create a bar chart
        plt.xlabel(xlabel)  # Set the label for the x-axis
        plt.ylabel(ylabel)  # Set the label for the y-axis
        plt.title(title)  # Set the title of the chart
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
        plt.tight_layout()  # Adjust layout to not cut off elements
        plt.show()

    @staticmethod
    def eval_regression(y_pred:np.array, y_true:np.array, plot: bool = True, bins: int = 5, n_features:int = None, regressor_name: str = None, precision: int = 4):
        """
        Evaluates regression model performance using common metrics.

        Parameters:
            y_pred (np.array): Predicted values.
            y_true (np.array): True values.
            plot (bool): If True, plots the regression error distribution. Default is True.
            bins (int): Number of bins for the error distribution plot. Default is 5.
            n_features (int): Number of features in the model (for adjusted R²). Optional.
            regressor_name (str): Name of the regressor for logging results. Optional.
            precision (int): Number of decimal places for rounding. Default is 4.

        Metrics:
            - MAE, MSE, RMSE, R², Adjusted R² (if `n_features` is provided), MAPE.
            - Error statistics: Mean, Variance, Standard Deviation.

        Returns:
            Plot or distribution of regression errors if `plot=True`.

        Notes:
            - Registers metrics if `regressor_name` is provided.
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = Evaluator.mean_squared_error(y_true, y_pred)
        rmse = Evaluator.mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        mape = Evaluator.mean_absolute_percentage_error(y_true, y_pred)
        print("MAE:", mae, "\n", "MSE:", mse, "\n",  "RMSE:", rmse, "\n", "R2:", r2, "\n", "MAPE:", mape)

        diff = np.abs( y_pred - y_true )
        mean, variance, std_dev = Evaluator.calculate_statistics(diff)
        print("Mean:", mean)
        print("Variance:", variance)
        print("Standard Deviation:", std_dev)

        r2_adjusted = None
        if (n_features != None): 
            r2_adjusted = Evaluator.adjustedr2(y_pred=y_pred, y_true=y_true, n_features=n_features)
            print("r2_adjusted:", r2_adjusted)

        if (regressor_name != None):
            Evaluator.register_regression(regressor_name, mae, mse, r2, mape, mean, std_dev, r2_adjusted)
        
        return Evaluator.regression_error_distribution(y_pred, y_true,  bins = bins, plot = plot, precision = precision)

    @staticmethod
    def register_regression(regressor_name, mae, mse, r2, mape, mean, std_dev, r2_adjusted):
        Evaluator.regression_register["Algorithm"].append(regressor_name)
        Evaluator.regression_register["mae"].append(round(mae, 4))
        Evaluator.regression_register["mse"].append(round(mse, 4))
        Evaluator.regression_register["mape"].append(round(mape, 4))
        Evaluator.regression_register["r2"].append(round(r2, 4))
        Evaluator.regression_register["error_mean"].append(round(mean, 4))
        Evaluator.regression_register["error_std_dev"].append(round(std_dev, 4))
        Evaluator.regression_register["adjuste_r2"].append(round(r2_adjusted, 4) if r2_adjusted is not None else None)
            
    @staticmethod
    def eval_classification(
        y_pred: np.array, 
        y_true: np.array, 
        binary_classification: bool, 
        average: str = 'weighted', 
        classifier_name: str = None
    ):
        """
        Evaluates classification model performance with key metrics.

        Parameters:
            y_pred (np.array): Predicted labels.
            y_true (np.array): True labels.
            binary_classification (bool): If True, computes ROC AUC.
            average (str): Averaging method for multi-class metrics (default: 'weighted').
            classifier_name (str): Optional, registers evaluation results if provided.

        Metrics:
            - Accuracy, Precision, Recall, F1 Score.
            - Confusion Matrix.
            - ROC AUC (for binary classification).

        Prints metrics and optionally registers them.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)
        conf_matrix = confusion_matrix(y_true, y_pred)
    
        print("Accuracy:", accuracy, "\n", "Precision:", precision, "\n", "Recall:", recall, "\n", "F1 Score:", f1)
        print("Confusion Matrix:\n", conf_matrix)

        roc_auc = None
        if binary_classification : 
            roc_auc = roc_auc_score(y_true, y_pred)
            print("ROC AUC:", roc_auc)

        if classifier_name != None:
            Evaluator.register_classification(classifier_name, accuracy, precision, recall, f1, roc_auc)

    @staticmethod
    def register_classification(classifier_name, accuracy, precision, recall, f1, roc_auc):
        Evaluator.classification_register["Algorithm"].append(classifier_name)
        Evaluator.classification_register["accuracy"].append(round(accuracy, 4))
        Evaluator.classification_register["precision"].append(round(precision, 4))
        Evaluator.classification_register["recall"].append(round(recall, 4))
        Evaluator.classification_register["f1"].append(round(f1, 4))
        Evaluator.classification_register["roc_auc"].append(round(roc_auc, 4) if roc_auc is not None else None)

    @staticmethod
    def eval_ordinal_classification(diff:np.array, plot = True):
        """
        Evaluates regression model performance using common metrics.

        Parameters:
            y_pred (np.array): Predicted values.
            y_true (np.array): True values.
            plot (bool): If True, plots the regression error distribution. Default is True.
            bins (int): Number of bins for the error distribution plot. Default is 5.
            n_features (int): Number of features in the model (for adjusted R²). Optional.
            regressor_name (str): Name of the regressor for logging results. Optional.

        Metrics:
            - MAE, MSE, RMSE, R², Adjusted R² (if `n_features` is provided), MAPE.
            - Error statistics: Mean, Variance, Standard Deviation.

        Returns:
            Plot or distribution of regression errors if `plot=True`.

        Notes:
            - Registers metrics if `regressor_name` is provided.
        """
        errors = Evaluator.equal_depth_binning(diff[diff > 0])

        if plot : Evaluator.plot_bar_chart(data=errors, title="Error frequeancies", xlabel="Range", ylabel="Frequency")
        print(errors)
        print("Error mean:", np.mean(diff[diff > 0]))
        print("Error rate:", len(diff[diff > 0])/len(diff)*100, "%")
        print("Overall mean:", np.mean(diff))
    
    @staticmethod
    def external_evaluation(labels, ground_truth):
        return {
            "adjusted_rand_score": adjusted_rand_score(ground_truth, labels),
            "normalized_mutual_info_score": normalized_mutual_info_score(ground_truth, labels),
            "homogeneity_score": homogeneity_score(ground_truth, labels),
            "completeness_score": completeness_score(ground_truth, labels),
            "v_measure_score": v_measure_score(ground_truth, labels)
        }

    @staticmethod
    def internal_evaluation(data, labels):
        return {
            "silhouette_score": silhouette_score(data, labels),
            "calinski_harabasz_score": calinski_harabasz_score(data, labels),
            "davies_bouldin_score": davies_bouldin_score(data, labels)
        }

    @staticmethod
    def eval_clustering(data, labels, ground_truth, algorithm_name):
        Evaluator.clustering_register["Algorithm"].append(algorithm_name)
        Evaluator.clustering_register["silhouette_score"].append(silhouette_score(data, labels) if len(set(labels)) > 1 else None)
        Evaluator.clustering_register["calinski_harabasz_score"].append(calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else None)
        Evaluator.clustering_register["davies_bouldin_score"].append(davies_bouldin_score(data, labels) if len(set(labels)) > 1 else None)
        Evaluator.clustering_register["adjusted_rand_score"].append(adjusted_rand_score(ground_truth, labels) if ground_truth is not None else None)
        Evaluator.clustering_register["normalized_mutual_info_score"].append(normalized_mutual_info_score(ground_truth, labels) if ground_truth is not None else None)
        Evaluator.clustering_register["homogeneity_score"].append(homogeneity_score(ground_truth, labels) if ground_truth is not None else None)
        Evaluator.clustering_register["completeness_score"].append(completeness_score(ground_truth, labels) if ground_truth is not None else None)
        Evaluator.clustering_register["v_measure_score"].append(v_measure_score(ground_truth, labels) if ground_truth is not None else None)

    @staticmethod
    def save(name: str):

        if not os.path.exists(name):
            os.makedirs(name)
        float_format = "{:.4e}".format
        Evaluator.save_regression(name, float_format)
        Evaluator.save_classification(name, float_format)
        Evaluator.save_clustering(name, float_format)

    @staticmethod
    def save_clustering(file_path, float_format = "{:.4e}".format):
        clustering_df = pd.DataFrame(Evaluator.clustering_register)
        clustering_latex_path = f"{file_path}_clustering.tex"
        with open(clustering_latex_path, "w") as f:
            f.write(clustering_df.to_latex(index=False, float_format=float_format))
        print(f"Clustering results saved to: {clustering_latex_path}")

        Evaluator.clustering_register = {"Algorithm": [], 
                            "silhouette_score": [], 
                            "calinski_harabasz_score": [], 
                            "davies_bouldin_score": [], 
                            "adjusted_rand_score": [],
                            "normalized_mutual_info_score": [], 
                            "homogeneity_score": [],
                            "completeness_score": [],
                            "v_measure_score": []}

    @staticmethod
    def save_classification(file_path, float_format = "{:.4e}".format):
        classification_df = pd.DataFrame(Evaluator.classification_register)
        classification_latex_path = f"{file_path}_classification.tex"
        with open(classification_latex_path, "w") as f:
            f.write(classification_df.to_latex(index=False, float_format=float_format))
        print(f"Classification results saved to: {classification_latex_path}")

        Evaluator.classification_register = {"Algorithm": [], 
                                "accuracy": [], 
                                "precision": [], 
                                "recall": [] ,
                                "f1": [], 
                                "roc_auc": []}

    @staticmethod
    def save_regression(file_path, float_format = "{:.4e}".format):
        regression_df = pd.DataFrame(Evaluator.regression_register)
        regression_latex_path = f"{file_path}_regression.tex"
        with open(regression_latex_path, "w") as f:
            f.write(regression_df.to_latex(index=False, float_format=float_format))
        print(f"Regression results saved to: {regression_latex_path}")
        Evaluator.regression_register = {"Algorithm": [], 
                           "mae": [], 
                           "mse": [], 
                           "mape": [] ,
                           "r2": [], 
                           "error_mean": [], 
                           "error_std_dev": [], 
                           "adjuste_r2": []}
