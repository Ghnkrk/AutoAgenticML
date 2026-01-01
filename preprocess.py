import pandas as pd
import numpy as np
from Dataset.Registry import dataset_registry
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.impute import SimpleImputer

class Preprocessor:
    def __init__(self, state, mode: str = "train"):
        """
        Initialize the Preprocessor.
        
        Args:
            state: The application state
            mode: Either "train" (fit+transform) or "test" (transform only using saved artifacts)
        """
        self.state = state
        self.mode = mode
        
        if mode not in ["train", "test"]:
            raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")

    def drop_columns(
        self,
        df: pd.DataFrame,
        columns: list[str]
    ) -> pd.DataFrame:
        """
        Drop specified columns from the DataFrame.
        
        Args:
            df: Input DataFrame
            columns: List of column names to drop
            
        Returns:
            DataFrame with specified columns removed
        """
        return df.drop(columns=columns)

    def feature_split(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        task_type: str
    ):
        """
        Split features and target into train/test sets.
        
        Uses stratified split for classification tasks to maintain class distribution.
        
        Args:
            x: Feature DataFrame
            y: Target Series
            task_type: Type of ML task ('binary', 'multiclass', or 'regression')
            
        Returns:
            Tuple of (x_train, x_test, y_train, y_test)
        """
        if task_type in ("binary", "multiclass"):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        return x_train, x_test, y_train, y_test

    def num_impute(
        self,
        df: pd.DataFrame,
        columns: list[str],
        strategy: str,
        imputer: SimpleImputer | None = None,
        fit: bool = True
    ):
        """
        Impute missing values in numerical columns.
        
        Args:
            df: Input DataFrame
            columns: List of numerical column names to impute
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
            imputer: Pre-fitted imputer (for test mode), or None to create new
            fit: If True, fit the imputer; if False, use existing imputer
            
        Returns:
            Tuple of (imputed DataFrame, fitted imputer)
        """
        if not columns:
            return df, imputer

        if imputer is None:
            # Map common LLM outputs to valid sklearn strategies
            if strategy in ["mode", "most frequent"]:
                strategy = "most_frequent"
            elif strategy in ["avg", "average"]:
                strategy = "mean"
            imputer = SimpleImputer(strategy=strategy)

        values = imputer.fit_transform(df[columns]) if fit else imputer.transform(df[columns])

        df = df.copy()
        df[columns] = pd.DataFrame(values, columns=columns, index=df.index)
        return df, imputer


    def cat_impute(
        self,
        df: pd.DataFrame,
        columns: list[str],
        strategy: str,
        imputer: SimpleImputer | None = None,
        fit: bool = True
    ):
        """
        Impute missing values in categorical columns.
        
        Args:
            df: Input DataFrame
            columns: List of categorical column names to impute
            strategy: Imputation strategy ('most_frequent', 'constant')
            imputer: Pre-fitted imputer (for test mode), or None to create new
            fit: If True, fit the imputer; if False, use existing imputer
            
        Returns:
            Tuple of (imputed DataFrame, fitted imputer)
        """
        if not columns:
            return df, imputer

        if imputer is None:
            # Map common LLM outputs to valid sklearn strategies
            if strategy in ["mode", "most frequent"]:
                strategy = "most_frequent"
            imputer = SimpleImputer(strategy=strategy)

        values = imputer.fit_transform(df[columns]) if fit else imputer.transform(df[columns])

        df = df.copy()
        df[columns] = pd.DataFrame(values, columns=columns, index=df.index)
        return df, imputer

    def encode(
        self,
        df: pd.DataFrame,
        encoding_config: dict,
        encoders: dict | None = None,
        target: pd.Series | None = None,
        fit: bool = True
    ):
        """
        Apply categorical encoding transformations.
        
        Supports three encoding strategies:
        - One-hot encoding: Creates binary columns for each category
        - Ordinal encoding: Maps categories to integers
        - Target encoding: Maps categories to target mean (requires target)
        
        Args:
            df: Input DataFrame with features
            encoding_config: Dict with keys 'one_hot', 'ordinal', 'target' containing column lists
            encoders: Pre-fitted encoders dict (for test mode), or None to create new
            target: Target Series (required for target encoding)
            fit: If True, fit encoders; if False, use existing encoders
            
        Returns:
            Tuple of (encoded DataFrame, encoders dict)
        """

        df = df.copy()
        if encoders is None:
            encoders = {}

        # ---------- ONE HOT ----------
        oh_cols = encoding_config.get("one_hot", [])
        # Filter to only columns that exist in dataframe
        oh_cols = [col for col in oh_cols if col in df.columns]
        
        if oh_cols:
            if fit:
                oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded = oh.fit_transform(df[oh_cols])
                encoders["one_hot"] = oh
            else:
                oh = encoders.get("one_hot")
                encoded = oh.transform(df[oh_cols])

            encoded_df = pd.DataFrame(
                encoded,
                columns=oh.get_feature_names_out(oh_cols),
                index=df.index
            )

            df.drop(columns=oh_cols, inplace=True)
            df = pd.concat([df, encoded_df], axis=1)

        # ---------- ORDINAL ----------
        ord_cols = encoding_config.get("ordinal", [])
        # Filter to only columns that exist in dataframe
        ord_cols = [col for col in ord_cols if col in df.columns]
        
        if ord_cols:
            if fit:
                ord_enc = OrdinalEncoder()
                df[ord_cols] = ord_enc.fit_transform(df[ord_cols])
                encoders["ordinal"] = ord_enc
            else:
                ord_enc = encoders.get("ordinal")
                df[ord_cols] = ord_enc.transform(df[ord_cols])

        # ---------- TARGET ----------
        tgt_cols = encoding_config.get("target", [])
        # Filter to only columns that exist in dataframe
        tgt_cols = [col for col in tgt_cols if col in df.columns]
        
        # ROBUSTNESS: Filter out high-cardinality columns (>50 unique values)
        # Target encoding is only suitable for low-to-medium cardinality
        if tgt_cols:
            valid_tgt_cols = []
            for col in tgt_cols:
                n_unique = df[col].nunique()
                if n_unique > 50:
                    print(f"  ⚠️  Skipping target encoding for '{col}' (cardinality={n_unique} > 50, too high)")
                else:
                    valid_tgt_cols.append(col)
            tgt_cols = valid_tgt_cols
        
        if tgt_cols:
            if target is None:
                # Test mode: skip target encoding (can't fit without target)
                print(f"  ⚠️  Skipping target encoding for {tgt_cols} (no target column in test data)")
                # Fall back to one-hot encoding for test mode
                print(f"  ℹ️  Falling back to one-hot encoding for {tgt_cols}")
                df = pd.get_dummies(df, columns=tgt_cols, drop_first=False)
            elif fit:
                try:
                    tgt_enc = TargetEncoder()
                    encoded = tgt_enc.fit_transform(df[tgt_cols], target)
                    # Convert to DataFrame and assign back
                    df[tgt_cols] = pd.DataFrame(encoded, columns=tgt_cols, index=df.index)
                    encoders["target"] = tgt_enc
                except Exception as e:
                    print(f"  ⚠️  Target encoding failed: {e}")
                    print(f"  ℹ️  Falling back to one-hot encoding for {tgt_cols}")
                    # Fall back to one-hot encoding
                    df = pd.get_dummies(df, columns=tgt_cols, drop_first=False)
            else:
                tgt_enc = encoders.get("target")
                if tgt_enc:
                    try:
                        encoded = tgt_enc.transform(df[tgt_cols])
                        # Convert to DataFrame and assign back
                        df[tgt_cols] = pd.DataFrame(encoded, columns=tgt_cols, index=df.index)
                    except Exception as e:
                        print(f"  ⚠️  Target encoding transform failed: {e}")
                        print(f"  ℹ️  Falling back to one-hot encoding for {tgt_cols}")
                        # Fall back to one-hot encoding
                        df = pd.get_dummies(df, columns=tgt_cols, drop_first=False)
                else:
                    # No encoder available, fall back to one-hot
                    print(f"  ℹ️  No target encoder available, using one-hot encoding for {tgt_cols}")
                    df = pd.get_dummies(df, columns=tgt_cols, drop_first=False)

        return df, encoders

    def scale(
        self,
        df: pd.DataFrame,
        columns: list[str],
        method: str,
        scaler=None,
        fit: bool = True
    ):
        """
        Scale numerical features using specified method.
        
        Supports three scaling strategies:
        - 'standard': Standardize features by removing mean and scaling to unit variance
        - 'minmax': Scale features to a given range (default 0-1)
        - 'robust': Scale using statistics robust to outliers (median and IQR)
        
        Args:
            df: Input DataFrame
            columns: List of numerical column names to scale
            method: Scaling method ('standard', 'minmax', or 'robust')
            scaler: Pre-fitted scaler (for test mode), or None to create new
            fit: If True, fit the scaler; if False, use existing scaler
            
        Returns:
            Tuple of (scaled DataFrame, fitted scaler)
        """
        if not columns:
            return df, scaler

        if scaler is None:
            # Normalize to lowercase for case-insensitive matching
            method = method.lower()
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

        values = scaler.fit_transform(df[columns]) if fit else scaler.transform(df[columns])

        df = df.copy()
        df[columns] = pd.DataFrame(values, columns=columns, index=df.index)
        return df, scaler
    def pca(
        self,
        df: pd.DataFrame,
        columns: list[str],
        variance_threshold: float,
        pca=None,
        fit: bool = True
    ):
        """
        Apply PCA dimensionality reduction.
        
        Args:
            df: DataFrame to transform
            columns: Columns to apply PCA on
            variance_threshold: Float between 0 and 1 representing the amount of 
                              variance to retain (e.g., 0.95 = 95%)
            pca: Existing PCA object (for transform-only)
            fit: Whether to fit the PCA or just transform
        
        Returns:
            Transformed DataFrame and PCA object
        """
        if not columns:
            return df, pca

        if pca is None:
            # When n_components is a float (0.0-1.0), sklearn selects the number of
            # components such that the variance explained is >= the specified percentage
            pca = PCA(n_components=variance_threshold)

        values = pca.fit_transform(df[columns]) if fit else pca.transform(df[columns])

        # Generate new column names for PCA components
        n_components = values.shape[1]
        pca_cols = [f"PC{i+1}" for i in range(n_components)]

        df = df.copy()
        # Drop original columns and add PCA components
        df = df.drop(columns=columns)
        pca_df = pd.DataFrame(values, columns=pca_cols, index=df.index)
        df = pd.concat([df, pca_df], axis=1)
        
        return df, pca

    def forward(self):
        """
        Main preprocessing pipeline orchestrator.
        Executes the preprocessing steps based on the analysisEvaluator_response.
        
        In 'train' mode: Fits transformers and returns artifacts
        In 'test' mode: Uses saved artifacts from state to transform
        """
        state = self.state
        mode = self.mode
        is_train = (mode == "train")

        # Get preprocessing plan from the analysis evaluator
        eval_response = state["analysisEvaluator_response"]
        
        # Get dataset and target from top-level state
        dataset_id = state["dataset_id"]
        target_column = state["target_column"]
        df = dataset_registry.get(dataset_id)

        # Get column lists (already filtered by analysis node)
        num_columns = state["num_columns"].copy()
        cat_columns = state["cat_columns"].copy()
        
        # Load existing artifacts if in test mode
        if not is_train:
            artifacts = state.get("preprocessing_artifacts", {})
            if not artifacts:
                raise ValueError("Test mode requires preprocessing_artifacts in state")
        else:
            artifacts = {}
        
        # 1. Drop features and update column lists based on evaluator's keep/drop decisions
        features_config = eval_response.get("features", {})
        keep_cols = features_config.get("keep", [])
        drop_cols = features_config.get("drop", [])
        
        # CRITICAL: Remove target column from keep_cols (it shouldn't be in features)
        target_column = state["target_column"]
        keep_cols = [c for c in keep_cols if c != target_column]
        
        # Drop ALL columns that aren't in keep_cols or target_column
        # This ensures columns like 'Name', 'PassengerId', 'Ticket' are removed
        columns_to_keep = keep_cols + [target_column]
        columns_to_drop = [c for c in df.columns if c not in columns_to_keep]
        
        if columns_to_drop:
            df = self.drop_columns(df, columns_to_drop)
        
        # Update column lists to only include kept features
        # Categorize kept columns into numerical and categorical
        num_columns = [c for c in keep_cols if c in num_columns]
        cat_columns = [c for c in keep_cols if c in cat_columns]
        
        # 2. Impute missing values
        missing_config = eval_response.get("missing_values", {})
        impute_config = missing_config.get("impute", {})
        
        num_imputer = artifacts.get("num_imputer") if not is_train else None
        cat_imputer = artifacts.get("cat_imputer") if not is_train else None
        
        if impute_config.get("numerical"):
            num_strategy = impute_config["numerical"]
            df, num_imputer = self.num_impute(
                df, num_columns, num_strategy, 
                imputer=num_imputer, 
                fit=is_train
            )
        
        if impute_config.get("categorical"):
            cat_strategy = impute_config["categorical"]
            df, cat_imputer = self.cat_impute(
                df, cat_columns, cat_strategy, 
                imputer=cat_imputer, 
                fit=is_train
            )

        # 3. Encode categorical features
        encoding_config = eval_response.get("encoding", {})
        
        # Check if target column exists (for evaluation) or not (for pure inference)
        has_target = target_column in df.columns
        
        if has_target:
            target_series = df[target_column]
            df_features = df.drop(columns=[target_column])
        else:
            # Pure inference mode - no ground truth available
            target_series = None
            df_features = df.copy()
        
        encoders = artifacts.get("encoders") if not is_train else None
        df_features, encoders = self.encode(
            df_features,
            encoding_config,
            encoders=encoders,
            target=target_series,  # May be None in inference mode
            fit=is_train
        )

        # 4. Scale numerical features
        scaling_config = eval_response.get("scaling", {})
        scale_method = scaling_config.get("method", "standard")
        
        # After encoding, identify which columns are still numerical
        remaining_num_cols = [c for c in num_columns if c in df_features.columns]
        
        scaler = artifacts.get("scaler") if not is_train else None
        if remaining_num_cols:
            df_features, scaler = self.scale(
                df_features,
                remaining_num_cols,
                scale_method,
                scaler=scaler,
                fit=is_train
            )
        
        # 5. Apply PCA if requested
        pca_config = eval_response.get("dimensionality_reduction", {})
        pca_obj = artifacts.get("pca_object") if not is_train else None
        
        if pca_config.get("use_pca", False):
            variance_threshold = pca_config.get("variance_threshold", 0.95)
            df_features, pca_obj = self.pca(
                df_features, 
                remaining_num_cols, 
                variance_threshold, 
                pca=pca_obj,
                fit=is_train
            )

        # 6. Split into train/test (only in train mode)
        if is_train:
            task_type = state["analysis_results"]["task_type"]
            x_train, x_test, y_train, y_test = self.feature_split(
                df_features,
                target_series,
                task_type
            )
            train_id = dataset_registry.register(x_train)
            val_id   = dataset_registry.register(x_test)
            y_train_id = dataset_registry.register(y_train)
            y_val_id   = dataset_registry.register(y_test)
            
            # Return preprocessing results for training
            return {
                "x_train_id": train_id,
                "x_val_id": val_id,
                "y_train_id": y_train_id,
                "y_val_id": y_val_id,
                "preprocessing_artifacts": {
                    "num_imputer": num_imputer,
                    "cat_imputer": cat_imputer,
                    "encoders": encoders,
                    "scaler": scaler,
                    "pca_object": pca_obj
                },
                "num_columns": num_columns,
                "cat_columns": cat_columns
            }
        else:
            # Return preprocessed features for testing/inference
            # If target exists (evaluation mode): y will contain ground truth
            # If target doesn't exist (inference mode): y will be None
            x_id = dataset_registry.register(df_features)
            if has_target:
                y_id = dataset_registry.register(target_series)
            else:
                y_id = None
            return {
                "x_id": x_id,
                "y_id": y_id,
                "has_target": has_target,  # Flag to indicate if ground truth is available
                "num_columns": num_columns,
                "cat_columns": cat_columns
            }

