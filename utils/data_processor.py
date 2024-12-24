import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA,TruncatedSVD, NMF, FastICA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import prince
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class Data_processor:
    """
    A utility class for processing datasets by handling missing data and filtering features.

    Requirements:
    - Python libraries:
      - pandas: for data manipulation and analysis.
      - seaborn, matplotlib: for visualization.
      - sklearn (preprocessing, decomposition, linear_model, feature_selection): for feature engineering and statistical analysis.
      - prince: for multiple correspondence analysis (MCA).
      - numpy: for numerical operations.

    Methods:
    This class is intended to include static or instance methods for:
      - Handling missing data: Methods for filling, removing, or analyzing missing values.
      - Feature filtering: Tools for selecting features based on statistical or model-driven approaches.
      - Dimensionality reduction: Techniques like PCA and TruncatedSVD.
      - Label encoding and other preprocessing tasks.
      - Visualization of data or features in 2D or 3D using matplotlib and seaborn.
      
    Intended Usage:
    - Use this class as a preprocessing step in data pipelines for machine learning projects.
    - Extend this class by adding more methods tailored to specific use cases.
    """

    @staticmethod
    def fill_na_with_mode(df: pd.DataFrame, column_name: list, inplace=False) -> pd.DataFrame:
        """
        Fills missing values in the specified column of a DataFrame with the column's mode.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            column_name (str): The name of the column to process.
            inplace (bool): Whether to modify the DataFrame in place. Default is False.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled, or the same DataFrame if inplace=True.

        Notes:
            - If the column does not exist, a message is printed and no changes are made.
            - The mode is the most frequently occurring value in the column.
        """
        if column_name in df.columns:
            mode_value = df[column_name].mode().iloc[0]
            df[column_name] = df[column_name].fillna(mode_value, inplace=inplace)
        else:
            print(f"La columna '{column_name}' no existe en el DataFrame.")
        return df

    @staticmethod
    def impute_with_linear_regression(data: pd.DataFrame, x_columns: list, y_column: str) -> pd.DataFrame:
        """
        Imputes missing values in the target column using linear regression based on other columns.

        Parameters:
            data (pd.DataFrame): The input DataFrame.
            x_columns (list): List of column names used as predictors (independent variables).
            y_column (str): The target column (dependent variable) to be imputed.

        Returns:
            pd.DataFrame: The DataFrame with missing values in the target column imputed.

        Process:
            1. Splits the data into:
            - Rows with non-missing target values for training.
            - Rows with missing target values for prediction.
            2. Trains a linear regression model using the provided predictors.
            3. Calculates the Mean Absolute Percentage Error (MAPE) on training data and prints it.
            4. Predicts and imputes missing values in the target column.

        Notes:
            - If there are no missing values in `y_column`, no imputation is performed.
            - Prints the regression MAPE to evaluate model accuracy.
            - Use only the most correlated columns
        """
        df_with_target = data.dropna(subset=[y_column])
        df_without_target = data[data[y_column].isna()]
        
        X_train = df_with_target[x_columns]
        y_train = df_with_target[y_column]
        X_test = df_without_target[x_columns]
        
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        epsilon = np.finfo(np.float64).eps
        mape = np.mean(np.abs((y_train - y_pred) / (y_train + epsilon))) * 100
        print(f"Regression mape {x_columns} -> {y_column}: {mape}%")
        if not X_test.empty:
            predicted_values = model.predict(X_test)
            data.loc[data[y_column].isna(), y_column] = predicted_values
        return data

    @staticmethod
    def remove_redundand_columns(df:pd.DataFrame )->pd.DataFrame:
        """
        Removes columns from a DataFrame that contain only a single unique non-null value.

        Parameters:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with redundant columns removed.

        Process:
            - Iterates through each column in the DataFrame.
            - Checks if the column has only one unique non-null value.
            - Drops the column if it is redundant.

        Notes:
            - Columns with all null values are not considered redundant.
        """
        for column in df.columns: #Remove redundant columns
            unique_values = df[column].dropna().unique() 
            if len(unique_values) == 1:
                df = df.drop(column, axis=1)
        return df

    @staticmethod
    def __impute_categorical_mode(df: pd.DataFrame, X: str, Y: str)->pd.DataFrame:
        """
        Imputes missing values in a categorical column based on the mode grouped by another column.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            X (str): The column used to group the data.
            Y (str): The target column in which missing values are imputed.

        Returns:
            pd.DataFrame: The DataFrame with missing values in the target column imputed.

        Process:
            - Groups the DataFrame by the column `X` and calculates the mode of column `Y` for each group.
            - Joins the calculated modes back to the DataFrame.
            - Fills missing values in `Y` with the corresponding mode from the grouped data.
            - Removes the temporary mode column after imputation.

        Notes:
            - If a group in `X` has no non-null values in `Y`, the mode for that group will be `None`.
        """
        modes = df.groupby(X, observed=False)[Y].agg(
            lambda x: x.dropna().mode()[0] if not x.dropna().empty else None
        )
        modes.name = 'Mode'
        df = df.join(modes, on=X, how='left')
        df[Y] = df.apply(lambda row: row['Mode'] if pd.isna(row[Y]) else row[Y], axis=1)
        df.drop('Mode', axis=1, inplace=True)      
        return df

    @staticmethod
    def impute_categorical_mode(df: pd.DataFrame, X: list, Y: str)->pd.DataFrame:
        """
        Iteratively imputes missing values in a categorical column based on the mode grouped by multiple columns.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            X (list): A list of columns to use iteratively for grouping.
            Y (str): The target column in which missing values are imputed.

        Returns:
            pd.DataFrame: The DataFrame with missing values in the target column imputed.

        Process:
            - Iterates through the list `X` to group the data and impute missing values in `Y` using modes.
            - After each iteration, the last column in `X` is removed, and the process continues with the remaining columns.
            - Remaining missing values in `Y` after all iterations are filled with the string "unknown".

        Notes:
            - Ensures that any unhandled missing values in `Y` are filled with "unknown".
        """
        for i in range(len(X)):
            df = Data_processor.__impute_categorical_mode(df, X, Y)
            X.pop(len(X)-1)
            missing_count = df[Y].isna().sum()
        print(f"Number of missing values in '{Y}' before filling with 'unknown': {missing_count}")
        df[Y] = df[Y].fillna("unknown")
        return df

    @staticmethod
    def CA (categorical_columns: pd.DataFrame, col_x: str, col_y: str)->pd.DataFrame:
        """
        Performs Correspondence Analysis (CA) on two categorical columns and plots the results.

        Parameters:
            categorical_columns (pd.DataFrame): The DataFrame containing the categorical columns.
            col_x (str): The name of the first categorical column.
            col_y (str): The name of the second categorical column.

        Process:
            1. Creates a contingency table of the two categorical columns.
            2. Calculates the probability matrix (P) from the contingency table.
            3. Derives row and column mass matrices (D_r and D_c).
            4. Normalizes the probability matrix and applies Singular Value Decomposition (SVD) to obtain:
            - Row coordinates: Position of row categories in reduced space.
            - Column coordinates: Position of column categories in reduced space.
            5. Calls `CAplot` to visualize the results.

        Notes:
            - Uses TruncatedSVD for dimensionality reduction to two components.
            - Requires a method `CAplot` to handle the visualization of row and column coordinates.

        Returns:
            None. The function plots the results using `CAplot`.

        Dependencies:
            - numpy for matrix calculations.
            - pandas for creating contingency tables.
            - sklearn's TruncatedSVD for dimensionality reduction.
        """
            
        contingency_table = pd.crosstab(categorical_columns[col_x], categorical_columns[col_y])
        P = contingency_table / contingency_table.values.sum()
        D_r = np.diag(1 / P.sum(axis=1))
        D_c = np.diag(1 / P.sum(axis=0))
        S = np.sqrt(D_r).dot(P).dot(np.sqrt(D_c))
        svd = TruncatedSVD(n_components=2)
        svd.fit(S)
        row_coordinates = svd.transform(S) 
        col_coordinates = svd.components_.T  
        Data_processor.CAplot(contingency_table, row_coordinates, col_coordinates)

    @staticmethod
    def CAplot(contingency_table, row_coordinates, col_coordinates):
        plt.figure(figsize=(8, 8))
        for i, label in enumerate(contingency_table.index):
            plt.scatter(row_coordinates[i, 0], row_coordinates[i, 1], color='blue')
            plt.text(row_coordinates[i, 0], row_coordinates[i, 1], f'{label}', color='blue', ha='right', va='bottom')
        for i, label in enumerate(contingency_table.columns):
            plt.scatter(col_coordinates[i, 0], col_coordinates[i, 1], color='red', marker='^')
            plt.text(col_coordinates[i, 0], col_coordinates[i, 1], f'{label}', color='red', ha='left', va='top')

        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        plt.title('Gráfico de Análisis de Correspondencias')
        plt.grid(True)
        plt.show()

    @staticmethod
    def chi_square_test(categorical_columns: pd.DataFrame, column_y: str) -> pd.DataFrame:
        """
        Performs a Chi-Square test for independence between categorical features and a target column.

        Parameters:
            categorical_columns (pd.DataFrame): The DataFrame containing categorical columns.
            column_y (str): The target column for the Chi-Square test.

        Returns:
            pd.DataFrame: A DataFrame containing:
                - Feature: The feature names.
                - Chi2 Stat: The Chi-Square statistic for each feature.
                - p-value: The p-value for each feature.

        Process:
            1. Encodes categorical columns into numerical form using LabelEncoder.
            2. Splits the DataFrame into predictors (X) and the target (y).
            3. Computes the Chi-Square statistic and p-values using `chi2`.
            4. Stores the results in a DataFrame and plots the results using `BarChart`.

        Notes:
            - This method assumes all columns except `column_y` are predictors.
            - A lower p-value indicates a stronger association between the feature and the target column.

        Dependencies:
            - sklearn's LabelEncoder for encoding categorical data.
            - sklearn's chi2 for Chi-Square computation.
            - pandas for result storage and manipulation.
            - `BarChart` method for visualization of the results.
        """
        encoded_df = categorical_columns.copy()
        label_encoder = LabelEncoder()
        for col in categorical_columns.columns:
            encoded_df[col] = label_encoder.fit_transform(encoded_df[col])

        X = encoded_df.drop(columns=[column_y])
        y = encoded_df[column_y]
        chi2_stat, p_values = chi2(X, y)

        results = pd.DataFrame({
            'Feature': X.columns,
            'Chi2 Stat': chi2_stat,
            'p-value': p_values
        })

        Data_processor.BarChart(results)
        return results

    @staticmethod
    def BarChart(results):
        results.sort_values('p-value', inplace=True)

        plt.figure(figsize=(10, 6))
        plt.barh(results['Feature'], results['p-value'], color='skyblue')
        plt.xlabel('p-value')
        plt.ylabel('Features')
        plt.title('Chi-Square Test Results')
        plt.gca().invert_yaxis()  # Invertir el eje y para que la característica con menor p-value esté arriba
        plt.show()
    
    @staticmethod
    def chi_square_filter(categorical_columns: pd.DataFrame, column_y: str, p_value_filter: float) -> pd.DataFrame:
        """
        Filters categorical features based on the Chi-Square test and a p-value threshold.

        Parameters:
            categorical_columns (pd.DataFrame): The DataFrame containing categorical features.
            column_y (str): The target column for the Chi-Square test.
            p_value_filter (float): The threshold for the p-value to select features.

        Returns:
            pd.DataFrame: A DataFrame containing only the features with p-values below or equal to the threshold, 
                        along with the target column ("price_categ").

        Process:
            1. Performs the Chi-Square test on all categorical features against the target column.
            2. Filters features with p-values less than or equal to the specified threshold.
            3. Retains the target column "price_categ" regardless of the test results.

        Notes:
            - Assumes the target column "price_categ" is always retained in the filtered DataFrame.
            - Utilizes the `chi_square_test` method to compute Chi-Square statistics and p-values.
        """
        result = Data_processor.chi_square_test(categorical_columns=categorical_columns, column_y=column_y)
        columns = list(result[ result["p-value"] <= p_value_filter ]["Feature"])
        columns.append("price_categ")
        return categorical_columns[ columns ]
    
    @staticmethod
    def reduce_dimensionality(dataframe: pd.DataFrame, n_components: int = 2, method: str = "svd"):
        """
        Reduces the dimensionality of numerical and dummy columns in a DataFrame using various methods.

        Parameters:
        - dataframe (pd.DataFrame): Input DataFrame with numerical and dummy columns.
        - n_components (int): Number of components for dimensionality reduction. Default is 2.
        - method (str): Dimensionality reduction method, options include "svd", "pca", "nmf", "ica", or "tsne". Default is "svd".

        Returns:
        - pd.DataFrame: DataFrame with reduced dimensions.
        """
        # Validate inputs
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not all(dataframe.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
            raise ValueError("All columns in the DataFrame must be numerical (including dummy columns).")

        # Select method
        if method == "svd":
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == "nmf":
            reducer = NMF(n_components=n_components, random_state=42, init='random')
        elif method == "ica":
            reducer = FastICA(n_components=n_components, random_state=42)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42)
        else:
            raise ValueError("Method must be one of 'svd', 'pca', 'nmf', 'ica', or 'tsne'.")

        # Scaling (skip for t-SNE as it works better with raw data)
        if method != "tsne":
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(dataframe)
        else:
            scaled_data = dataframe.values

        # Apply the reducer
        reduced_data = reducer.fit_transform(scaled_data)

        # Print explained variance ratio if available
        if hasattr(reducer, 'explained_variance_ratio_'):
            explained_variance = reducer.explained_variance_ratio_
            total_explained_variance = explained_variance.sum() * 100
            print(f"Explained variance by each component: {explained_variance * 100}")
            print(f"Total explained variance: {total_explained_variance}%")
        else:
            print(f"Dimensionality reduction with {method} does not provide explained variance.")

        # Convert to DataFrame
        reduced_df = pd.DataFrame(
            reduced_data,
            columns=[f"component_{i+1}" for i in range(n_components)]
        )

        return reduced_df