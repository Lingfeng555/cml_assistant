import pandas as pd
from .evaluator import Evaluator
import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
import optuna

try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans, DBSCAN as cuDBSCAN
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

class ClusterGenerator:
    """
    The ClusterGenerator class provides functionality to generate, tune, and evaluate clustering models
    using various algorithms. It supports hyperparameter optimization with Optuna and allows exporting
    results into LaTeX files for further documentation.

    Attributes:
    ------------
    dataset : pd.DataFrame
        The input dataset for clustering.

    use_cuml : bool
        Whether to use cuML for GPU-accelerated computations if available.

    Methods:
    --------
    find_best_clustering(method, n_trials):
        Optimizes the clustering algorithm and finds the best parameters.

    custom_clustering(method, **params):
        Generates a clustering model using specific parameters provided by the user.

    generate(ground_truth, n_trials):
        Optimizes and evaluates multiple clustering methods and records results.

    save(name):
        Saves internal and external evaluation metrics, as well as the best parameters, to .tex files.
    """
        
    def __init__(self, dataset, use_cuml=False):
        """
        Initialize the ClusterGenerator class with a dataset.

        Parameters:
        dataset (pd.DataFrame): The input dataset for clustering.
        use_cuml (bool): Whether to use cuML for acceleration if available.
        """
        self.dataset = dataset
        self.use_cuml = use_cuml and CUML_AVAILABLE

    def _objective(self, trial, method):
        """
        Objective function for hyperparameter optimization with Optuna.

        Parameters:
        trial (optuna.trial.Trial): The trial object for Optuna.
        method (str): The clustering method to optimize.

        Returns:
        float: The silhouette score for the trial.
        """
        if method == "kmeans":
            n_clusters = trial.suggest_int("n_clusters", 2, 15)
            init = trial.suggest_categorical("init", ["k-means++", "random"])
            n_init = trial.suggest_int("n_init", 10, 50)
            max_iter = trial.suggest_int("max_iter", 100, 500)
            model = cuKMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, random_state=42) if self.use_cuml else KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, random_state=42)

        elif method == "agglomerative":
            n_clusters = trial.suggest_int("n_clusters", 2, 15)
            linkage = trial.suggest_categorical("linkage", ["ward", "complete", "average", "single"])
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

        elif method == "dbscan":
            eps = trial.suggest_float("eps", 0.1, 5.0, log=True)
            min_samples = trial.suggest_int("min_samples", 3, 20)
            model = cuDBSCAN(eps=eps, min_samples=min_samples) if self.use_cuml else DBSCAN(eps=eps, min_samples=min_samples)

        elif method == "birch":
            threshold = trial.suggest_float("threshold", 0.1, 1.0)
            n_clusters = trial.suggest_int("n_clusters", 2, 15)
            branching_factor = trial.suggest_int("branching_factor", 10, 100)
            model = Birch(threshold=threshold, n_clusters=n_clusters, branching_factor=branching_factor)

        elif method == "optics":
            min_samples = trial.suggest_int("min_samples", 3, 20)
            max_eps = trial.suggest_float("max_eps", 0.1, 5.0, log=True)
            metric = trial.suggest_categorical("metric", ["minkowski", "euclidean", "manhattan", "cosine"])
            cluster_method = trial.suggest_categorical("cluster_method", ["xi", "dbscan"])
            model = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric, cluster_method=cluster_method)

        elif method == "gmm":
            n_components = trial.suggest_int("n_components", 2, 15)
            covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])
            tol = trial.suggest_float("tol", 1e-4, 1e-2, log=True)
            reg_covar = trial.suggest_float("reg_covar", 1e-6, 1e-2, log=True)
            max_iter = trial.suggest_int("max_iter", 100, 200)
            model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, tol=tol, reg_covar=reg_covar, max_iter=max_iter, random_state=42)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        labels = model.fit_predict(self.dataset) if hasattr(model, 'fit_predict') else model.fit(self.dataset).predict(self.dataset)

        if len(set(labels)) > 1:  # Ensure there is more than one cluster
            return silhouette_score(self.dataset, labels)
        else:
            return -1.0  # Penalize single-cluster solutions

    def find_best_clustering(self, method, n_trials=50):
        """
        Optimize the clustering algorithm and find the best parameters.

        Parameters:
        method (str): The clustering method to optimize (e.g., 'kmeans', 'agglomerative').
        n_trials (int): The number of optimization trials.

        Returns:
        dict: The best parameters and corresponding metrics.
        """
        study = optuna.create_study(direction="maximize")
        n_jobs=-1
        with ThreadPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            study.optimize(lambda trial: self._objective(trial, method), n_trials=n_trials, n_jobs=n_jobs)
        best_params = study.best_params

        # Train the final model with the best parameters
        if method == "kmeans":
            model = cuKMeans(**best_params, random_state=42) if self.use_cuml else KMeans(**best_params, random_state=42)

        elif method == "agglomerative":
            model = AgglomerativeClustering(**best_params)

        elif method == "dbscan":
            model = cuDBSCAN(**best_params) if self.use_cuml else DBSCAN(**best_params)

        elif method == "birch":
            model = Birch(**best_params)

        elif method == "optics":
            model = OPTICS(**best_params)

        elif method == "gmm":
            model = GaussianMixture(**best_params, random_state=42)

        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        labels = model.fit_predict(self.dataset) if hasattr(model, 'fit_predict') else model.fit(self.dataset).predict(self.dataset)

        # Calculate clustering metrics

        return {"best_params": best_params, "labels": labels, "model": model}
        
    def custom_clustering(self, method, **params):
        """
        Generate a clustering model using specific parameters provided by the user.

        Parameters:
        method (str): Clustering method (e.g., 'kmeans', 'agglomerative', 'dbscan', 'birch', 'optics', 'gmm').
        params (dict): Specific parameters for the clustering method.

        Returns:
        dict: Contains the generated labels, internal metrics, and the trained model.
        """
        if method == "kmeans":
            model = cuKMeans(**params, random_state=42) if self.use_cuml else KMeans(**params, random_state=42)

        elif method == "agglomerative":
            model = AgglomerativeClustering(**params)

        elif method == "dbscan":
            model = cuDBSCAN(**params) if self.use_cuml else DBSCAN(**params)

        elif method == "birch":
            model = Birch(**params)

        elif method == "optics":
            model = OPTICS(**params)

        elif method == "gmm":
            model = GaussianMixture(**params, random_state=42)

        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        # Entrenar y predecir etiquetas
        labels = model.fit_predict(self.dataset) if hasattr(model, 'fit_predict') else model.fit(self.dataset).predict(self.dataset)
        return {"labels": labels, "model": model}

    def generate(self, ground_truth=None, n_trials=50):
        """
        Optimizes and evaluates multiple clustering methods, recording results for each.

        Parameters:
        -----------
        ground_truth : np.array, optional
            The ground truth labels for clustering evaluation.

        n_trials : int, optional
            The number of optimization trials. Default is 50.
        """
        methods = ["kmeans", "agglomerative", "dbscan", "birch", "optics", "gmm"]
        self.best_params = {}
        
        for method in methods:
            print(f"Optimizing method: {method}")
            clustering_result = self.find_best_clustering(method, n_trials=n_trials)

            # Record best parameters
            self.best_params[method] = clustering_result["best_params"]

            # Use evaluator to evaluate the clustering
            Evaluator.eval_clustering(data=self.dataset, labels=clustering_result["labels"], ground_truth=ground_truth, algorithm_name=method)

    def save(self, name: str):
        """
        Save internal and external evaluations, as well as best parameters, to .tex files.

        Parameters:
        name (str): Name of the directory to save the .tex files.
        """
        if self.best_params is None:
            raise ValueError("No results to save. Please run generate() first.")

        directory_path = "evaluation"
        dir = f"{directory_path}/{name}/clustering"

        # Create directory if it doesn't exist
        if not os.path.exists(dir):
            os.makedirs(dir)

        if hasattr(self, 'best_params'):
            for method, params in self.best_params.items():
                params_df = pd.DataFrame([params])
                params_df.to_latex(f"{dir}/regression_best_param_{method}.tex", index=False)

        Evaluator.save_clustering(dir)
