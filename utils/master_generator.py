import pandas as pd
import numpy as np
from .classifierGenerator import ClassifierGenerator
from .cluster_generator import ClusterGenerator
from .regressor_generator import RegressionGenerator

class MasterGenerator:
    """
    The MasterGenerator class is responsible for generating and tuning classical machine learning algorithms
    for regression, classification, and clustering tasks. It automates the process of training models, tuning hyperparameters,
    and exporting the results into LaTeX files for documentation purposes.

    Attributes:
    ------------
    X : pd.DataFrame
        The feature set for the machine learning models.

    y_categ : np.array
        The categorical target values for classification tasks.

    y_numeric : np.array
        The numerical target values for regression tasks.

    name : str
        The name of the experiment or project used for saving results.

    n_tries : int
        The number of trials for hyperparameter optimization.

    CUML : bool, optional
        Whether to use RAPIDS cuML for GPU-accelerated computations. Default is False.

    Methods:
    --------
    generate():
        Executes the generation and tuning process for regression, classification, and clustering tasks.

    """

    def __init__(self, X: pd.DataFrame, y_categ: np.array, y_numeric: np.array, name: str, n_tries: int, CUML: bool = False):
        """
        Initializes the MasterGenerator with the provided dataset and configuration.

        Parameters:
        -----------
        X : pd.DataFrame
            The feature set for the models.

        y_categ : np.array
            The target values for classification tasks.

        y_numeric : np.array
            The target values for regression tasks.

        name : str
            The name of the project/experiment.

        n_tries : int
            The number of trials for hyperparameter tuning.

        CUML : bool, optional
            Whether to use RAPIDS cuML for GPU acceleration. Default is False.
        """
        self.X = X
        self.y_categ = y_categ
        self.y_numeric = y_numeric
        self.name = name
        self.n_tries = n_tries
        self.CUML = CUML

    def _regression_generate(self):
        """
        Generates regression models using the RegressionGenerator class.

        This method trains and tunes regression models, then saves the results
        to LaTeX files with the provided experiment name.
        """
        regression_generator = RegressionGenerator(X=self.X, y=self.y_numeric, use_cuml=self.CUML)
        regression_generator.generate(n_trials=self.n_tries)
        regression_generator.save(self.name)

    def _classification_generate(self):
        """
        Generates classification models using the ClassifierGenerator class.

        This method trains and tunes classification models, then saves the results
        to LaTeX files with the provided experiment name.
        """
        classifier_generator = ClassifierGenerator(dataset=self.X, target_column=self.y_categ.values.ravel(), use_cuml=self.CUML)
        classifier_generator.generate(n_trials=self.n_tries)
        classifier_generator.save(self.name)

    def _clustering_generate(self):
        """
        Generates clustering models using the ClusterGenerator class.

        This method trains and tunes clustering models, then saves the results
        to LaTeX files with the provided experiment name.
        """
        cluster_generator = ClusterGenerator(dataset=self.X, use_cuml=self.CUML)
        cluster_generator.generate(n_trials=self.n_tries, ground_truth=self.y_categ)
        cluster_generator.save(self.name)

    def generate(self) -> None:
        """
        Executes the generation process for regression, classification, and clustering models.

        This method sequentially calls the private methods to generate models for each task type,
        tuning hyperparameters and saving results for each.
        """
        self._regression_generate()
        self._classification_generate()
        self._clustering_generate()
