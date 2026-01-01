




import json
import pandas as pd
from pathlib import Path
from langchain_groq import ChatGroq
from Dataset.Registry import dataset_registry
from descriptive import DescriptiveAnalysis
from analysis import Analysis
from preprocess import Preprocessor
from trainer import ModelTrainer
from evaluator import ModelEvaluator
from promptTemplate import (
    DESCRIPTIVE_PROMPT,
    ANALYSIS_PROMPT,
    MODEL_DESIGN_PROMPT,
    EVALUATOR_AGENT_PROMPT,
    SUMMARIZER_PROMPT
)
from ModelResponseSchema import (
    DescriptiveEvaluatorResponse,
    AnalysisEvaluatorResponse,
    ModelDesignResponse,
    EvaluatorAgentResponse,
    parse_llm_response
)


def datasetRegistry(state):
        """
        This node registers the dataset in the registry.
        Handles both train and test datasets.
        
        output: state["dataset_id"]
        """
        # Check if this is test data loading for inference
        return_to_inference = state.get("return_to_inference", False)
        test_path = state.get("test_dataset_path")
        
        if return_to_inference and test_path:
            # Test data mode - to_stage should be test_registry
            assert state["to_stage"] in ["test_registry", "registry"], f"Invalid stage for test data: {state['to_stage']}"
            dataset_path = test_path
            from_stage = "test_registry"
            log_msg = f"Test dataset registered: {test_path}"
        else:
            # Train data mode
            assert state["from_stage"] == "user", "Invalid stage"
            assert state["to_stage"] == "registry", "Invalid stage"
            dataset_path = state["dataset_path"]
            from_stage = "registry"
            log_msg = f"Dataset registered in registry"

        dataset_id = dataset_registry.register(pd.read_csv(dataset_path))
        
        return {
            "dataset_id": dataset_id,
            "from_stage": from_stage,
            "logs": [f"{log_msg} with dataset id : {dataset_id}"]
        }

    
def descriptiveNode(state):
        """
        This node performs descriptive analysis on the dataset.
        It returns the descriptive results in the state.

        output: state["descriptive_results"]
        """
        assert state["to_stage"] == "descriptive", f"Invalid stage: {state.get('to_stage')}" #for debugging cases
        new_state = DescriptiveAnalysis(state)
        
        return {
            "from_stage": "descriptive",
            "descriptive_results": new_state["descriptive_results"],
            "num_columns": new_state["num_columns"],  # Top level for preprocessing access
            "cat_columns": new_state["cat_columns"],  # Top level for preprocessing access
            "logs": [f"Descriptive analysis done on dataset {new_state['descriptive_results']}"]
        }


def analysisNode(state):
        """
        This node performs analysis on the dataset.
        It returns the analysis results in the state.

        output: state["analysis_results"]
        """

        assert state["from_stage"] == "descriptive_evaluator", "Invalid stage" #for debugging cases 
        assert state["to_stage"] == "analysis", "Invalid stage" #for debugging cases 


        analysis_obj = Analysis(state)
        new_state = analysis_obj.forward()    
        
        return {
            "from_stage": "analysis",
            "num_columns": new_state["num_columns"],  # Top level for preprocessing access
            "cat_columns": new_state["cat_columns"],  # Top level for preprocessing access
            "analysis_results": new_state["analysis_results"],
            "logs": [f"Analysis done on dataset {new_state['analysis_results']}"]
        }

def DAEvaluator(state):
        """
        Evaluator with structured outputs and reasoning.
        Uses different schemas for descriptive vs analysis evaluation.
        """
        from_stage = state["from_stage"]
        
        if from_stage == "descriptive":
            # Descriptive Evaluator
            model = ChatGroq(
                model="openai/gpt-oss-20b",
                reasoning_effort="medium",
                model_kwargs={
                    "include_reasoning": True,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "descriptive_evaluator_response",
                            "strict": True,
                            "schema": DescriptiveEvaluatorResponse.model_json_schema()
                        }
                    }
                }
            )
            
            prompt = DESCRIPTIVE_PROMPT
            input_msgs = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(state["descriptive_results"])}
            ]
            
            response = model.invoke(input_msgs)
            
            # Parse with Pydantic
            evaluator_response = parse_llm_response(response, DescriptiveEvaluatorResponse)
            
            # Log reasoning if available
            reasoning = getattr(response, 'reasoning', None)
            if reasoning:
                print(f"[Descriptive Evaluator Reasoning]\n{reasoning}\n")
            
            return {
                "descEvaluator_response": evaluator_response.model_dump(),
                "logs": [f"Descriptive Evaluator response : {evaluator_response.model_dump()}"],
                "from_stage": "descriptive_evaluator"
            }

        elif from_stage == "analysis":
            # Analysis Evaluator
            model = ChatGroq(
                model="openai/gpt-oss-20b",
                reasoning_effort="medium",
                model_kwargs={
                    "include_reasoning": True,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "analysis_evaluator_response",
                            "strict": False,
                            "schema": AnalysisEvaluatorResponse.model_json_schema()
                        }
                    }
                }
            )
            
            prompt = ANALYSIS_PROMPT
            
            # Include available columns so evaluator knows what's actually there
            evaluator_input = {
                "analysis_results": state["analysis_results"],
                "available_num_columns": state.get("num_columns", []),
                "available_cat_columns": state.get("cat_columns", [])
            }
            
            input_msgs = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(evaluator_input)}
            ]
            
            response = model.invoke(input_msgs)
            
            # Parse with Pydantic
            evaluator_response = parse_llm_response(response, AnalysisEvaluatorResponse)
            
            # Log reasoning if available
            reasoning = getattr(response, 'reasoning', None)
            if reasoning:
                print(f"[Analysis Evaluator Reasoning]\n{reasoning}\n")
            
            return {
                "analysisEvaluator_response": evaluator_response.model_dump(),
                "num_columns": state["num_columns"],  # Preserve from analysis node
                "cat_columns": state["cat_columns"],  # Preserve from analysis node
                "logs": [f"Analysis Evaluator response : {evaluator_response.model_dump()}"],
                "from_stage": "analysis_evaluator"
            }
        
        else:
            raise ValueError("Invalid stage") #routing debugging


def preprocessNode(state):
        """
        This node performs preprocessing on the dataset.
        It returns the preprocessing results in the state.
        
        output: x_train_id, x_val_id, y_train_id, y_val_id (train mode)
                OR x_id, y_id, has_target (test mode)
        """
        assert state["to_stage"] == "preprocess", "Invalid stage" #for debugging cases
        
        # Hybrid mode detection: use orchestrator's mode or auto-detect
        mode = state.get("preprocess_mode")
        
        if not mode or mode is None:
            # Auto-detect based on artifacts presence
            has_artifacts = state.get("preprocessing_artifacts") is not None
            mode = "test" if has_artifacts else "train"
            print(f"[PreprocessNode] Auto-detected mode: {mode}")
        else:
            print(f"[PreprocessNode] Using orchestrator mode: {mode}")
        
        preprocess_obj = Preprocessor(state, mode)
        result = preprocess_obj.forward()
        
        # Build return state with from_stage and logs
        return_state = {
            "from_stage": "preprocess",
            "logs": [f"Preprocessing completed in {mode} mode : {result}"]
        }
        
        # Merge preprocessing results
        return_state.update(result)
        
        return return_state

def metaDataForModelDesign(state):
        """
        This node creates metadata for the model design.
        It returns the metadata in the state.
        """
        assert state["to_stage"] == "model_design_metadata", "Invalid stage" #for debugging cases
        
        X_train = dataset_registry.get(state["x_train_id"])
        y_train = dataset_registry.get(state["y_train_id"])

        # Simplify encoding info to avoid token limits
        encoding_config = state["analysisEvaluator_response"]["encoding"]
        encoding_summary = {
            "one_hot_count": len(encoding_config.get("one_hot", [])),
            "ordinal_count": len(encoding_config.get("ordinal", [])),
            "target_count": len(encoding_config.get("target", []))
        }
        
        metadata = {
            "task_type": state["task_type"],
            "n_samples": X_train.shape[0],
            "n_features": X_train.shape[1],
            "class_balance": (
                y_train.value_counts(normalize=True).to_dict()
                if state["task_type"] != "regression"
                else None
            ),
            "preprocessing_summary": {
                "encoding_summary": encoding_summary,
                "scaling": state["analysisEvaluator_response"]["scaling"]["method"],
                "pca": state["analysisEvaluator_response"]["dimensionality_reduction"]["use_pca"]
            },
            "constraints": {
                "interpretability": "medium",
                "training_speed": "medium"
            }
        }

        return {
            "model_metadata": metadata,
            "from_stage": "model_design_metadata" 
        }

def modelDesignNode(state):
        """
        This node is an agent that designs the model training process.
        It returns the model design in the state.
        """
        assert state["to_stage"] == "model_design", "Invalid stage" #for debugging cases

        designAgent = ChatGroq(
                model="openai/gpt-oss-20b",
                reasoning_effort="medium",
                model_kwargs={
                    "include_reasoning": True,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "model_design_response",
                            "strict": True,
                            "schema": ModelDesignResponse.model_json_schema()
                        }
                    }
                }
            )
        model_design_prompt = MODEL_DESIGN_PROMPT
            

        # Truncate metadata to avoid token limit issues
        metadata = state["model_metadata"].copy()
        feature_count = metadata.get("feature_count", 0)
        
        # If too many features, just send summary instead of full list
        if feature_count > 50:
            metadata["features_note"] = f"{feature_count} features after preprocessing (truncated for brevity)"
            # Remove detailed feature lists if present
            metadata.pop("feature_names", None)
        
        model_design_input = {
                "model_metadata": metadata
        }
        
        # Make task type EXPLICIT in user message
        task_type = metadata.get("task_type", "unknown")
        if task_type == "regression":
            task_instruction = "\n\nCRITICAL: This is a REGRESSION task. You MUST use regression models (linear_regression, ridge_regression, random_forest, gradient_boosting) and regression metrics (r2, mse, rmse, mae). DO NOT use logistic_regression or classification metrics!"
        else:
            task_instruction = "\n\nCRITICAL: This is a CLASSIFICATION task. You MUST use classification models (logistic_regression, random_forest, gradient_boosting, linear_svm) and classification metrics (accuracy, f1, roc_auc). DO NOT use linear_regression or ridge_regression!"
            
        input_msgs = [
                {"role": "system", "content": model_design_prompt},
                {"role": "user", "content": str(model_design_input) + task_instruction}
        ]
            
        response = designAgent.invoke(input_msgs)
            
        # Parse with Pydantic
        model_design_response = parse_llm_response(response, ModelDesignResponse)
            
        # Log reasoning if available
        reasoning = getattr(response, 'reasoning', None)
        if reasoning:
            print(f"[Analysis Evaluator Reasoning]\n{reasoning}\n")
            
        return {
                "modelDesign_response": model_design_response.model_dump(),
                "logs": [f"Model Design response : {model_design_response.model_dump()}"],
                "from_stage": "model_design"
            }
        

def TrainerNode(state):
        """
        Trains models from ModelDesignResponse and validates them.
        Saves trained models to disk and returns paths/metrics to state.
        """
        assert state["to_stage"] == "training", "Invalid stage"
        
        # Get required data from state
        model_design_response = state.get("modelDesign_response")
        if not model_design_response:
            raise ValueError("modelDesign_response missing from state")
        
        x_train_id = state.get("x_train_id")
        y_train_id = state.get("y_train_id")
        x_val_id = state.get("x_val_id")
        y_val_id = state.get("y_val_id")
        task_type = state.get("task_type") or state.get("analysisEvaluator_response", {}).get("task_type", "binary")
        
        if not all([x_train_id, y_train_id, x_val_id, y_val_id]):
            raise ValueError("Training/validation data IDs missing from state")
        
        # Initialize trainer
        trainer = ModelTrainer(dataset_registry)
        
        # Train all models
        trained_models = trainer.train_models(
            model_design_response=model_design_response,
            x_train_id=x_train_id,
            y_train_id=y_train_id,
            x_val_id=x_val_id,
            y_val_id=y_val_id,
            task_type=task_type
        )
        
        # Prepare detailed log
        primary_metric = model_design_response.get("primary_metric", "accuracy")
        log_summary = f"Trained {len(trained_models)} model(s):"
        for m in trained_models:
            metric_value = m["metrics"].get(primary_metric, "N/A")
            if isinstance(metric_value, (int, float)):
                log_summary += f"\n  - {m['name']}: {primary_metric}={metric_value:.4f}, saved to {m['path']}"
            else:
                log_summary += f"\n  - {m['name']}: {primary_metric}={metric_value}, saved to {m['path']}"
        
        return {
            "trained_models": trained_models,
            "from_stage": "training",
            "logs": [log_summary]
        }
    
def EvaluatorNode(state):
        """
        Deterministic model evaluation and ranking.
        Analyzes trained models and provides decision-ready results.
        """
        assert state["to_stage"] == "evaluation", "Invalid stage"
        
        # Get trained models and primary metric
        trained_models = state.get("trained_models", [])
        model_design_response = state.get("modelDesign_response", {})
        primary_metric = model_design_response.get("primary_metric", "accuracy")
        
        if not trained_models:
            raise ValueError("No trained models found in state")
        
        # Run evaluation
        evaluator = ModelEvaluator(primary_metric=primary_metric)
        evaluation_results = evaluator.forward(trained_models)
        
        # Prepare log
        best_model = evaluation_results["best_model"]
        best_score = evaluation_results["best_score"]
        confidence = evaluation_results["confidence"]
        warnings_count = len(evaluation_results["warnings"])
        
        log_summary = (
            f"Model Evaluation Complete:\n"
            f"  Best Model: {best_model} ({primary_metric}={best_score:.4f})\n"
            f"  Confidence: {confidence}\n"
            f"  Warnings: {warnings_count}\n"
            f"  Recommendation: {evaluation_results['recommendation']}"
        )
        
        return {
            "evaluation_results": evaluation_results,
            "from_stage": "evaluation",
            "logs": [log_summary]
        }
    
def EvaluatorAgentNode(state):
        """
        LLM-based evaluator agent that recommends accept/retrain decision.
        """
        assert state["to_stage"] == "evaluator_agent", "Invalid stage"
        
        # Get evaluation results
        evaluation_results = state.get("evaluation_results")
        if not evaluation_results:
            raise ValueError("evaluation_results missing from state")
        
        # Prepare input for LLM
        eval_input = {
            "model_ranking": evaluation_results["model_ranking"],
            "best_model": evaluation_results["best_model"],
            "best_score": evaluation_results["best_score"],
            "confidence": evaluation_results["confidence"],
            "warnings": evaluation_results["warnings"],
            "recommendation": evaluation_results["recommendation"],
            "primary_metric": evaluation_results["primary_metric"]
        }
    
        
        model = ChatGroq(
            model="openai/gpt-oss-20b",
            reasoning_effort="medium",
            model_kwargs={
                "include_reasoning": True,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "evaluator_agent_response",
                        "strict": True,
                        "schema": EvaluatorAgentResponse.model_json_schema()
                    }
                }
            }
        )
        
        input_msgs = [
            {"role": "system", "content": EVALUATOR_AGENT_PROMPT},
            {"role": "user", "content": str(eval_input)}
        ]
        
        response = model.invoke(input_msgs)
        evaluator_agent_response = parse_llm_response(response, EvaluatorAgentResponse)
        
        # Log reasoning if available
        reasoning = getattr(response, 'reasoning', None)
        if reasoning:
            print(f"[Evaluator Agent Reasoning]\n{reasoning}\n")
        
        log_summary = (
            f"Evaluator Agent Decision: {evaluator_agent_response.decision}\n"
            f"  Reasoning: {evaluator_agent_response.reasoning[:200]}..."
        )
        
        return {
            "evaluator_agent_response": evaluator_agent_response.model_dump(),
            "from_stage": "evaluator_agent",
            "logs": [log_summary]
        }
    
def InferenceNode(state):
        """
        Multi-model inference node.
        Loads selected models, runs predictions on test data, saves to CSV.
        """
        assert state["to_stage"] == "inference", "Invalid stage"
        
        # Get required data
        selected_models = state.get("selected_models_for_inference", [])
        trained_models = state.get("trained_models", [])
        x_test_id = state.get("x_id")  # Preprocessed test features
        task_type = state.get("task_type")
        target_column = state.get("target_column")
        
        if not x_test_id:
            raise ValueError("Preprocessed test data (x_id) not found in state")
        
        # Get test data
        X_test = dataset_registry.get(x_test_id)
        
        print(f"\n{'='*80}")
        print(f"RUNNING INFERENCE ON {len(selected_models)} MODEL(S)")
        print(f"{'='*80}\n")
        
        import pickle
        from datetime import datetime
        import pandas as pd
        
        predictions_dir = Path("./predictions")
        predictions_dir.mkdir(exist_ok=True)
        
        inference_results = {}
        
        for model_name in selected_models:
            # Find model info
            model_info = next((m for m in trained_models if m["name"] == model_name), None)
            if not model_info:
                print(f"  ✗ Model {model_name} not found in trained models")
                continue
            
            model_path = model_info["path"]
            
            # Load model
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            # FEATURE ALIGNMENT: Ensure test data has same columns as training data
            # Get expected feature names from the model
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                current_features = X_test.columns.tolist()
                
                # Add missing columns (fill with 0)
                missing_cols = set(expected_features) - set(current_features)
                for col in missing_cols:
                    X_test[col] = 0
                
                # Remove extra columns
                extra_cols = set(current_features) - set(expected_features)
                X_test = X_test.drop(columns=list(extra_cols), errors='ignore')
                
                # Reorder columns to match training order
                X_test = X_test[expected_features]
            
            # Run predictions
            predictions = model.predict(X_test)
            
            # Load original test CSV to get ID column
            test_csv_path = state.get("test_dataset_path")
            test_df_original = pd.read_csv(test_csv_path)
            id_column_name = test_df_original.columns[0]  # Assume first column is ID
            id_values = test_df_original[id_column_name]
            
            # For classification, argmax is already done by predict()
            # Create DataFrame with ID and predictions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pred_filename = f"{model_name}_predictions_{timestamp}.csv"
            pred_path = predictions_dir / pred_filename
            
            pred_df = pd.DataFrame({
                id_column_name: id_values,
                target_column: predictions
            })
            pred_df.to_csv(pred_path, index=False)
            
            inference_results[model_name] = {
                "predictions_path": str(pred_path),
                "num_predictions": len(predictions)
            }
            
            print(f"  ✓ {model_name}: {len(predictions)} predictions saved to {pred_path}")
        
        print(f"\n{'='*80}")
        print(f"INFERENCE COMPLETE: {len(inference_results)}/{len(selected_models)} successful")
        print(f"{'='*80}\n")
        
        log_summary = f"Inference completed for {len(inference_results)} model(s)"
        for name, info in inference_results.items():
            log_summary += f"\n  - {name}: {info['num_predictions']} predictions → {info['predictions_path']}"
        
        return {
            "inference_results": inference_results,
            "from_stage": "inference",
            "logs": [log_summary]
        }




def SummarizerNode(state):
        """
        Generate professional summary of entire pipeline run.
        """
        
        print(f"\n{'='*80}")
        print("GENERATING PIPELINE SUMMARY")
        print(f"{'='*80}\n")
        
        # Collect all relevant information
        logs = state.get("logs", [])
        dataset_path = state.get("dataset_path", "Unknown")
        target_column = state.get("target_column", "Unknown")
        task_type = state.get("task_type", "Unknown")
        
        # Preprocessing info
        preprocess_config = state.get("analysisEvaluator_response", {})
        
        # Model info
        trained_models = state.get("trained_models", [])
        model_design = state.get("model_design_response", {})
        primary_metric = model_design.get("primary_metric", "Unknown")
        
        # Evaluation info
        evaluation_results = state.get("evaluation_results", {})
        
        # Inference info
        inference_results = state.get("inference_results", {})
        test_dataset_path = state.get("test_dataset_path", None)
        
        # Build context for LLM
        context = f"""
        PIPELINE LOGS:
        {chr(10).join(logs)}

        DATASET INFORMATION:
        - Path: {dataset_path}
        - Target Column: {target_column}
        - Task Type: {task_type}

        PREPROCESSING CONFIGURATION:
        {json.dumps(preprocess_config, indent=2) if preprocess_config else "Not available"}

        MODELS TRAINED:
        {json.dumps([{"name": m["name"], "metrics": m.get("metrics", {})} for m in trained_models], indent=2) if trained_models else "No models trained"}

        PRIMARY METRIC: {primary_metric}

        EVALUATION RESULTS:
        {json.dumps(evaluation_results, indent=2) if evaluation_results else "Not available"}

        INFERENCE RESULTS:
        Test Dataset: {test_dataset_path if test_dataset_path else "No inference performed"}
        {json.dumps(inference_results, indent=2) if inference_results else ""}
        """
        
        # Call LLM to generate summary
        model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.6
        )
        
        input_msgs = [
            {"role": "system", "content": SUMMARIZER_PROMPT},
            {"role": "user", "content": context}
        ]
        
        try:
            response = model.invoke(input_msgs)
            summary = response.content
        except Exception as e:
            print(f"⚠️  LLM summarizer failed: {e}")
            summary = "Summary generation failed. Please review logs for details."
        
        # Print summary
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80 + "\n")
        print(summary)
        print("\n" + "="*80 + "\n")
        
        return {
            "pipeline_summary": summary,
            "from_stage": "summarizer",
            "logs": [f"Pipeline summary generated"]
        }
