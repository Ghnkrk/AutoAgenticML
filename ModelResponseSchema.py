
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict

def parse_llm_response(response, schema):
    try:
        return schema.model_validate_json(response.content)
    except Exception as e:
        raise ValueError(f"Invalid LLM output:\n{response.content}") from e


class L0_OrchestratorResponse(BaseModel):
    model_config = {"extra": "forbid"}
    # Current phase
    current_phase: Literal[
        "start",
        "prelim",     # L1
        "modeling",   # L2
        "end"
    ]

    # Phase status signals
    phase_done: bool = Field(description="Whether the current phase is complete")

    # Routing
    next_phase: Optional[Literal[
        "prelim",
        "modeling",
        "end"
    ]] = Field(description="The next phase to transition to, or null if staying")

    # Metadata / logging
    notes: Optional[str] = Field(description="Reasoning for the decision")

class L1_OrchestratorResponse(BaseModel):
    """Schema for L1_Orchestrator routing decisions"""
    model_config = {"extra": "forbid"}  # Required for strict: true
    
    next_stage: Literal[
        "registry",
        "test_registry",
        "descriptive", 
        "descriptive_evaluator",
        "analysis",
        "analysis_evaluator",
        "human",
        "preprocess",
        "end"
    ] = Field(description="The next stage in the pipeline")
    
    # Optional fields must not have defaults to be 'required' in JSON schema for strict mode
    preprocess_mode: Optional[Literal["train", "test"]] = Field(
        description="Mode for preprocessing: 'train' when coming from human review, 'test' for inference, null otherwise"
    )
    
    notes: Optional[str] = Field(
        description="Brief technical note on the transition"
    )

class L2_OrchestratorResponse(BaseModel):
    model_config = {"extra": "forbid"}
    next_stage: Literal["model_design_metadata", "model_design", "human_model_selection", "training", "evaluation", "evaluator_agent", "human_evaluation", "human_inference", "inference", "end"]
    notes: Optional[str] = Field(default=..., description="Reasoning for the decision")

class DescriptiveEvaluatorResponse(BaseModel):
    """Schema for descriptive evaluator decisions"""
    model_config = {"extra": "forbid"}
    
    run: List[str] = Field(description="List of analyses to run")
    skip: List[str] = Field(description="List of columns to skip")
    notes: str = Field(description="Reasoning for decisions")


class FeaturesConfig(BaseModel):
    """Features to keep/drop"""
    model_config = {"extra": "forbid"}
    
    keep: List[str] = Field(description="Features to keep")
    drop: List[str] = Field(description="Features to drop")
    # Even explicitly empty lists must be provided in strict mode
    uncertain: List[str] = Field(description="Uncertain features")
    notes: Optional[str] = Field(default=None, description="Reasoning for feature selection")


class ImputeConfig(BaseModel):
    """Imputation strategies"""
    model_config = {"extra": "forbid"}
    
    numerical: str = Field(description="Strategy for numerical features")
    categorical: str = Field(description="Strategy for categorical features")
    notes: Optional[str] = Field(default=None, description="Reasoning for imputation")


class MissingValuesConfig(BaseModel):
    """Missing values handling"""
    model_config = {"extra": "forbid"}
    
    drop_threshold: float = Field(description="Threshold for dropping features")
    impute: ImputeConfig
    notes: Optional[str] = Field(default=None, description="Reasoning for missing values handling")


class EncodingConfig(BaseModel):
    """Encoding strategies"""
    model_config = {"extra": "forbid"}
    
    one_hot: List[str] = Field(description="Columns for one-hot encoding")
    ordinal: List[str] = Field(description="Columns for ordinal encoding")
    target: List[str] = Field(description="Columns for target encoding")
    notes: Optional[str] = Field(default=None, description="Reasoning for encoding strategy")


class ScalingConfig(BaseModel):
    """Scaling configuration"""
    model_config = {"extra": "forbid"}
    
    method: str = Field(description="Scaling method")
    notes: Optional[str] = Field(default=None, description="Reasoning for scaling choice")


class DimensionalityReductionConfig(BaseModel):
    """Dimensionality reduction configuration"""
    model_config = {"extra": "forbid"}
    
    use_pca: bool = Field(description="Whether to use PCA")
    variance_threshold: float = Field(description="Variance threshold for PCA")
    notes: Optional[str] = Field(default=None, description="Reasoning for dimensionality reduction")


class AnalysisEvaluatorResponse(BaseModel):
    """Schema for analysis evaluator decisions"""
    model_config = {"extra": "forbid"}
    
    task_type: str = Field(description="Task type (binary, multiclass, regression)")
    features: FeaturesConfig
    missing_values: MissingValuesConfig
    encoding: EncodingConfig
    scaling: ScalingConfig = Field(default_factory=lambda: ScalingConfig(method="standard"))
    dimensionality_reduction: DimensionalityReductionConfig = Field(default_factory=lambda: DimensionalityReductionConfig(use_pca=False, variance_threshold=0.95))
    notes: str = Field(default="Analysis completed successfully.", description="Reasoning for preprocessing decisions")

class ModelParam(BaseModel):
    model_config = {"extra": "forbid"}
    name: str
    value: float | str | bool | None

class ModelSpec(BaseModel):
    model_config = {"extra": "forbid"}
    name: Literal[
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "linear_svm",
        "linear_regression",
        "ridge_regression"
    ]
    params: List[ModelParam]
    rationale: str = Field(..., description="Why this model is suitable")

class ModelDesignResponse(BaseModel):
    model_config = {"extra": "forbid"}
    models: List[ModelSpec] = Field(
        ..., min_items=1, max_items=4,
        description="Ordered list of models to train"
    )
    primary_metric: Literal[
        "accuracy",
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "rmse",
        "rmsle",
        "mae",
        "r2"
    ]
    notes: str
class EvaluatorAgentResponse(BaseModel):
    """Schema for evaluator agent decisions after model evaluation"""
    model_config = {"extra": "forbid"}
    
    decision: Literal["accept_models", "retrain_models"] = Field(description="Accept current models or retrain")
    reasoning: str = Field(description="Detailed reasoning for the decision")
    models_to_retrain: List[str] = Field(description="Models to retrain (empty list if accepting all, or list of model names)")
    suggested_improvements: str = Field(description="Suggestions for improving model performance (empty string if none)")
