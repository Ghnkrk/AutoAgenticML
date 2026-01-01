import numpy as np 
import pandas as pd
import scipy.stats as stats
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from Dataset.Registry import dataset_registry


class Analysis():

        def __init__(self, state):
            """
            Initializes the AnalysisNode with the current application state.

            Args:
                state (dict): The current state of the application, expected to contain 
                             'descriptive_results' from previous steps.
            """
            self.state = state

        def __cardinality(self, df: pd.DataFrame, cat_columns: list[str]):
            """
            Calculates the cardinality (number of unique values) for categorical columns.

            Args:
                df (pd.DataFrame): The input dataset.
                cat_columns (list[str]): List of categorical column names.

            Returns:
                dict: A dictionary mapping column names to their unique value counts.
            """
            valid_cols = [c for c in cat_columns if c in df.columns]
            return {col: df[col].nunique() for col in valid_cols}

        def __missing_values(self, df: pd.DataFrame):
            """
            Identifies missing values and calculates the percentage of missing data per column.

            Args:
                df (pd.DataFrame): The input dataset.

            Returns:
                dict: A dictionary where each key is a column name and the value is 
                      another dictionary with 'missing_count' and 'missing_percentage'.
            """
            return {col : 
                    {"missing_count": int(df[col].isnull().sum()),
                    "missing_percentage": float(df[col].isnull().mean() * 100)} for col in df.columns}

        def __task_type(self, df: pd.DataFrame, target_column: str):
            """
            Infuses the ML task type based on the target column's characteristics.

            Args:
                df (pd.DataFrame): The input dataset.
                target_column (str): The name of the target variable.

            Returns:
                str: One of 'binary', 'categorical', or 'regression'.
            """
            k = 10
            n_unique = df[target_column].nunique()
            target_dtype = df[target_column].dtype

            if n_unique == 2:
                return "binary"
            elif n_unique <= k and target_dtype in ("object", "int"):
                return "categorical"
            else:
                return "regression"

        def __feature_target_corr(self, df: pd.DataFrame, target_column: str, num_columns: list[str], task_type: str):
            """
            Computes correlation/association between numerical features and the target.
            Uses Pearson for regression, Point-Biserial for binary, and ANOVA (F-score) for categorical.

            Args:
                df (pd.DataFrame): The input dataset.
                target_column (str): The name of the target variable.
                num_columns (list[str]): List of numerical feature names.
                task_type (str): The type of ML task.

            Returns:
                dict: A dictionary mapping features to their statistical scores and p-values.
            """
            df_sub = df[num_columns + [target_column]].dropna(axis=0)
            all_coeff = {}

            if task_type == "regression":
                for col in num_columns:
                    corr, p_val = stats.pearsonr(df_sub[col], df_sub[target_column])
                    all_coeff[col] = {"correlation": float(corr), "p-value": float(p_val)}
                return all_coeff
            
            if task_type == "binary":
                for col in num_columns:
                    corr, p_val = stats.pointbiserialr(df_sub[col], df_sub[target_column])
                    all_coeff[col] = {"correlation": float(corr), "p-value": float(p_val)}
                return all_coeff

            if task_type == "categorical":
                for col in num_columns:
                    # Group by target categories and perform ANOVA
                    groups = [group[col].values for name, group in df_sub.groupby(target_column)]
                    f, p_val = stats.f_oneway(*groups)
                    all_coeff[col] = {"f-score": float(f), "p-value": float(p_val)}
                return all_coeff

        def __feature_to_feature_corr(self, df: pd.DataFrame, num_columns: list[str], method: str = "pearson"):
            """
            Identifies highly correlated pairs of numerical features.

            Args:
                df (pd.DataFrame): The input dataset.
                num_columns (list[str]): List of numerical feature names.
                method (str): Correlation method (e.g., 'pearson', 'spearman').

            Returns:
                list: A list of tuples (feature1, feature2, correlation_value) for pairs > 0.8.
            """

            df_sub = df[num_columns].dropna(axis=0)
            high_corr_pairs = []
            cols = df_sub.columns
            corr_matrix = df_sub.corr(method)

            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    val = corr_matrix.iloc[i, j]
                    if abs(val) > 0.8:
                        high_corr_pairs.append((cols[i], cols[j], float(val)))

            return high_corr_pairs
            
        def __multicollinearity(self, df: pd.DataFrame,target_column: str, num_columns: list[str]):
            """
            Calculates Variance Inflation Factor (VIF) to detect multicollinearity.

            Args:
                df (pd.DataFrame): The input dataset.
                num_columns (list[str]): List of numerical feature names.

            Returns:
                dict: A dictionary mapping feature names to their VIF values.
            """
            df_sub = df[num_columns].dropna(axis=0)
            all_vif = {}

            x_const = add_constant(df_sub)

            for i, col in enumerate(x_const.columns):
                if col == "const":
                    continue
                vif = variance_inflation_factor(x_const.values, i)
                all_vif[col] = float(vif)

            return all_vif


        def forward(self):
            """
            Executes the full structural analysis of the dataset.
            This includes task identification, cardinality checks, missing values analysis,
            and correlation/multicollinearity studies.

            Returns:
                dict: The updated state containing 'analysis_results'.
            """
            state = self.state

            eval_response = state["descEvaluator_response"]
            runs = eval_response.get("run", [])
            skip = eval_response.get("skip", [])

            dataset_id = state["dataset_id"] 
            num_columns = state["num_columns"]  # Changed from num_cols
            cat_columns = state["cat_columns"]  # Changed from cat_cols
            target_column = state["target_column"]

            df = dataset_registry.get(dataset_id)
            # Filter skip list to avoid KeyErrors
            actual_skip = [c for c in skip if c in df.columns]
            df = df.drop(actual_skip, axis=1) 
            
            num_columns = [col for col in num_columns if col not in skip]
            cat_columns = [col for col in cat_columns if col not in skip]

            cardinality = {}
            missing_values = {}
            feature_target_corr = {}
            feature_to_feature_corr = []
            multicollinearity = {}
            task_type = state.get("task_type")


            if task_type is None and any(
                step in runs for step in ("task_type", "feature_target_correlation")
            ):
                task_type = self.__task_type(df, target_column)

            for step in runs:
                if step == "cardinality":
                    cardinality = self.__cardinality(df, cat_columns)
                elif step == "missingness":
                    missing_values = self.__missing_values(df)
                elif step == "feature_target_correlation":
                    feature_target_corr = self.__feature_target_corr(df, target_column, num_columns, task_type)
                elif step == "feature_to_feature_correlation":
                    feature_to_feature_corr = self.__feature_to_feature_corr(df, num_columns)
                elif step == "multicollinearity":
                    multicollinearity = self.__multicollinearity(df, target_column, num_columns)
                else:
                    print(f"Warning: Unknown analysis step '{step}'") if step != "task_type" else None
             
            

            return {
                "analysis_results": {
                    "task_type": task_type or state.get("task_type"),
                    "cardinality": cardinality,
                    "missing_values": missing_values,
                    "feature_target_corr": feature_target_corr,
                    "feature_to_feature_corr": feature_to_feature_corr,
                    "multicollinearity": multicollinearity
                },
                "num_columns": num_columns,
                "cat_columns": cat_columns
            }
            
            
   

        
    
    