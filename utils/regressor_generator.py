import os
import pandas as pd
import numpy as np
import optuna
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression
from .evaluator import Evaluator

try:
    from cuml.svm import SVR as cuSVR, LinearSVR as cuLinealSVR
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

class RegressionGenerator:
    """
    The RegressionGenerator class provides functionality to generate, tune, and evaluate regression models
    using various algorithms. It supports hyperparameter optimization with Optuna and allows exporting results
    into LaTeX files for further documentation.

    Attributes:
    ------------
    X : pd.DataFrame
        The feature matrix for regression.

    y : np.ndarray
        The target variable for regression.

    use_cuml : bool
        Whether to use cuML for GPU-accelerated computations if available.

    Methods:
    --------
    find_best_model(algorithm, n_trials):
        Optimizes the regression algorithm and finds the best hyperparameters.

    generate(algorithms, n_trials):
        Generates regression results for a list of algorithms.

    save(name):
        Saves internal and external evaluation metrics, as well as the best parameters, to .tex files.
    """
        
    def __init__(self, X: pd.DataFrame, y: np.ndarray, use_cuml=False):
        """
        Initialize the RegressionOptimizer class.

        Parameters:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Target variable.
        use_cuml (bool): Whether to use cuML for acceleration if available.
        """
        self.X = X.astype({col: 'int' for col in X.select_dtypes('bool').columns}).astype('float32')
        self.y = y
        self.use_cuml = use_cuml and CUML_AVAILABLE

    def _objective(self, trial, algorithm):
        """
        Objective function for Optuna hyperparameter optimization.

        Parameters:
        trial (optuna.trial.Trial): The trial object for Optuna.
        algorithm (str): The regression algorithm to optimize.

        Returns:
        float: The negative mean squared error for the trial.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        if algorithm == "cart":
            # Hiperparámetros adicionales para DecisionTreeRegressor
            max_depth = trial.suggest_int("max_depth", 2, 30)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            max_features = trial.suggest_categorical("max_features", [None, "sqrt", "log2"])
            criterion = trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"])

            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                criterion=criterion,
                random_state=42
            )

        elif algorithm == "random_forest":
            # Hiperparámetros adicionales para RandomForestRegressor
            n_estimators = trial.suggest_int("n_estimators", 20, 50, step=50)
            max_depth = trial.suggest_int("max_depth", 20, 50)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            bootstrap = trial.suggest_categorical("bootstrap", [True, False])

            if self.use_cuml:
                model = cuRandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    n_streams=1,
                    random_state=42
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=bootstrap,
                    random_state=42
                )

        elif algorithm == "svr":
            # Hiperparámetros adicionales para SVR
            kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
            C = trial.suggest_float("C", 0.01, 100.0, log=True)
            epsilon = trial.suggest_float("epsilon", 0.001, 1.0, log=True)
            degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else None
            coef0 = trial.suggest_float("coef0", 0.0, 1.0) if kernel in ["poly", "sigmoid"] else None
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            max_iter = trial.suggest_int("max_iter", 1000, 10000, step=1000)
            
            if self.use_cuml:
                    if kernel == "linear":
                        # Usar cuLinealSVR para kernel lineal con cuML
                        model = cuLinealSVR(
                            C=C,
                            epsilon=epsilon,
                            max_iter=max_iter
                        )
                    else:
                        # Usar cuSVR para otros kernels
                        degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else None
                        coef0 = trial.suggest_float("coef0", 0.0, 1.0) if kernel in ["poly", "sigmoid"] else None
                        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
                        
                        model = cuSVR(
                            kernel=kernel,
                            C=C,
                            epsilon=epsilon,
                            degree=degree if degree else 3,
                            coef0=coef0 if coef0 else 0.0,
                            gamma=gamma
                        )
            else:
                if kernel == "linear":
                    # Usar LinearSVR de sklearn para kernel lineal
                    model = LinearSVR(
                        C=C,
                        epsilon=epsilon,
                        max_iter=max_iter,
                        random_state=42
                    )
                else:
                    # Usar SVR de sklearn para otros kernels
                    degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else None
                    coef0 = trial.suggest_float("coef0", 0.0, 1.0) if kernel in ["poly", "sigmoid"] else None
                    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

                    model = SVR(
                        kernel=kernel,
                        C=C,
                        epsilon=epsilon,
                        degree=degree if degree else 3,
                        coef0=coef0 if coef0 else 0.0,
                        gamma=gamma
                    )

        elif algorithm == "linear_regression":
            model = LinearRegression()

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return Evaluator.mean_absolute_percentage_error(y_test, y_pred)

    def find_best_model(self, algorithm, n_trials=50):
        """
        Optimize the regression algorithm and find the best hyperparameters.

        Parameters:
        algorithm (str): The regression algorithm to optimize (e.g., 'cart', 'random_forest').
        n_trials (int): The number of optimization trials.

        Returns:
        dict: The best parameters and corresponding metrics.
        """
        study = optuna.create_study(direction="minimize")
        n_jobs=-1
        with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            study.optimize(lambda trial: self._objective(trial, algorithm), n_trials=n_trials, n_jobs=n_jobs)
        best_params = study.best_params

        # Train the final model with the best parameters
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        if algorithm == "cart":
            model = DecisionTreeRegressor(**best_params, random_state=42)

        elif algorithm == "random_forest":
            if self.use_cuml:
                model = cuRandomForestRegressor(**best_params, random_state=42)
            else:
                model = RandomForestRegressor(**best_params, random_state=42)

        elif algorithm == "svr":
            if self.use_cuml:
                model = cuSVR(**best_params)
            else:
                model = SVR(**best_params)

        elif algorithm == "linear_regression":
            model = LinearRegression()

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "MAPE": Evaluator.mean_absolute_percentage_error(y_test, y_pred),
        }

        return {"best_params": best_params, "metrics": metrics, "model": model}

    def generate(self, algorithms=None, n_trials=50):
        """
        Generate regression results for a list of algorithms.

        Parameters:
        algorithms (list): List of algorithms to optimize (e.g., ['cart', 'random_forest']).
        n_trials (int): Number of optimization trials for each algorithm.

        Returns:
        dict: A dictionary with algorithms as keys and their results as values.
        """
        if algorithms is None:
            algorithms = ["cart", "random_forest", "svr", "linear_regression"]

        self.best_params = {}

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        for algorithm in algorithms:
            try:
                print(f"Optimizing algorithm: {algorithm}")
                result = self.find_best_model(algorithm, n_trials=n_trials)
                self.best_params[algorithm] = result["best_params"]

                y_pred = result["model"].predict(X_test)
                Evaluator.eval_regression(y_pred = y_pred, bins=5, y_true=y_test, plot=False, n_features=X_train.shape[1], regressor_name=algorithm)

            except Exception as e:
                print(f"Error optimizing {algorithm}: {e}")
                self.best_params[algorithm] = None
        return self.best_params

    def save(self, name: str):
        """
        Save the results to a CSV file.

        Parameters:
        results (dict): The results dictionary returned by the generate method.
        name (str): Name of the file to save the results.
        """
        directory_path = "evaluation"
        dir = f"{directory_path}/{name}/regression"

        # Create directory if it doesn't exist
        if not os.path.exists(dir):
            os.makedirs(dir)


        if hasattr(self, 'best_params'):
            for method, params in self.best_params.items():
                params_df = pd.DataFrame([params])
                params_df.to_latex(f"{dir}/regression_best_param_{method}.tex", index=False)

        Evaluator.save(dir)