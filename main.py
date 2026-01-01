from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

from typing import TypedDict, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from typing_extensions import Annotated
import json
from datetime import datetime
import asyncio

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from Dataset.Registry import dataset_registry

from Orchestrators import (
    L0_Orchestrator,
    L1_Orchestrator,
    L2_Orchestrator,
    L0_router,
    L1_router,
    L2_router
)

from Nodes import (
    datasetRegistry,
    descriptiveNode,
    analysisNode,
    DAEvaluator,
    preprocessNode,
    metaDataForModelDesign,
    modelDesignNode,
    TrainerNode,
    EvaluatorNode,
    EvaluatorAgentNode,
    InferenceNode,
    SummarizerNode
)

from HumanNodes import (
    HumanReview_PreprocessConfig,
    HumanReview_ModelSelection,
    HumanReview_ModelEvaluation,
    HumanReview_InferenceDecision
)

from promptTemplate import (DESCRIPTIVE_PROMPT,
 ANALYSIS_PROMPT,
    L0_ORCHESTRATOR_PROMPT, 
    L1_ORCHESTRATOR_PROMPT, 
    L2_ORCHESTRATOR_PROMPT,
    MODEL_DESIGN_PROMPT,
    EVALUATOR_AGENT_PROMPT,
    SUMMARIZER_PROMPT
)

from descriptive import DescriptiveAnalysis
from analysis import Analysis
from preprocess import Preprocessor
from trainer import ModelTrainer
from evaluator import ModelEvaluator

from ModelResponseSchema import (
    L0_OrchestratorResponse,
    L1_OrchestratorResponse,
    L2_OrchestratorResponse,
    DescriptiveEvaluatorResponse,
    AnalysisEvaluatorResponse,
    ModelDesignResponse,
    EvaluatorAgentResponse,
    parse_llm_response
)

load_dotenv()

class State(TypedDict):

    #----Phase tracker----
    current_phase: str
    phase_done: bool
    next_phase: str

    #----Stage tracker----
    from_stage: str
    to_stage: str
    
    # ----- Input -----
    dataset_id: str     
    dataset_path: str              
    target_column: str
    num_columns: List[str]
    cat_columns: List[str]
    task_type: Optional[str]
    
    # ----- Evaluator Responses -----
    descEvaluator_response: Dict
    analysisEvaluator_response: Dict
    
    # ----- Descriptive -----
    descriptive_results: Dict

    # ----- Analysis -----
    analysis_results: Dict
    feature_recommendations: Optional[Dict]

    # ----- Preprocessing -----
    preprocess_mode: str
    preprocessing_artifacts: Optional[Dict]
    x_train_id: Optional[str]
    x_val_id: Optional[str]
    y_train_id: Optional[str]
    y_val_id: Optional[str]
    x_id: Optional[str]  # For test/inference
    y_id: Optional[str]  # For test/inference
    has_target: Optional[bool]  # Whether target is available in test data
    
    # ----- Human-in-loop -----
    selected_features: Optional[List[str]]
    user_notes: Optional[Dict]

    # ----- Model Design -----
    model_metadata: Optional[Dict]
    modelDesign_response: Optional[Dict]
    
    # ----- Model Training -----
    trained_models: Optional[List[Dict]]  # List of {name, path, metrics, timestamp}
    
    # ----- Model Evaluation -----
    evaluation_results: Optional[Dict]  # Results from ModelEvaluator
    evaluator_agent_response: Optional[Dict]  # LLM decision on accept/retrain
    
    # ----- Inference -----
    test_dataset_path: Optional[str]  # Path to test CSV file
    inference_results: Optional[Dict]  # Predictions per model
    selected_models_for_inference: Optional[List[str]]  # User-selected models
    return_to_inference: Optional[bool]  # Flag for cross-phase routing

    # ----- Logs -----
    logs: Annotated[List[str], list.__add__]






def main():

    graph = StateGraph(State)

    # Adding nodes 
    graph.add_node("L0_Orchestrator", L0_Orchestrator)
    graph.add_node("L1_Orchestrator", L1_Orchestrator)
    graph.add_node("L2_Orchestrator", L2_Orchestrator)
    graph.add_node("DatasetRegistry", datasetRegistry)
    graph.add_node("DescriptiveNode", descriptiveNode)
    graph.add_node("AnalysisNode", analysisNode)
    graph.add_node("Evaluator", DAEvaluator)
    graph.add_node("HumanReview_PreprocessConfig", HumanReview_PreprocessConfig)
    graph.add_node("PreprocessNode", preprocessNode)
    graph.add_node("MetaDataForModelDesign", metaDataForModelDesign)
    graph.add_node("ModelDesignNode", modelDesignNode)
    graph.add_node("HumanReview_ModelSelection", HumanReview_ModelSelection)
    graph.add_node("TrainerNode", TrainerNode)
    graph.add_node("EvaluatorNode", EvaluatorNode)
    graph.add_node("EvaluatorAgentNode", EvaluatorAgentNode)
    graph.add_node("HumanReview_ModelEvaluation", HumanReview_ModelEvaluation)
    graph.add_node("HumanReview_InferenceDecision", HumanReview_InferenceDecision)
    graph.add_node("InferenceNode", InferenceNode)
    graph.add_node("SummarizerNode", SummarizerNode)

    # Defining edges
    graph.add_edge(START, "L0_Orchestrator")

    graph.add_conditional_edges(
        "L0_Orchestrator",
        L0_router,
        {
            "L1_Orchestrator": "L1_Orchestrator",
            "L2_Orchestrator": "L2_Orchestrator",
            "SummarizerNode" : "SummarizerNode"
        }
    )

    graph.add_conditional_edges(
        "L1_Orchestrator", 
        L1_router,
        {
            "DatasetRegistry": "DatasetRegistry",
            "DescriptiveNode": "DescriptiveNode",
            "AnalysisNode": "AnalysisNode",
            "Evaluator": "Evaluator",
            "HumanReview_PreprocessConfig": "HumanReview_PreprocessConfig",
            "PreprocessNode": "PreprocessNode",
            "L0_Orchestrator": "L0_Orchestrator"
        }
    )

    graph.add_conditional_edges(
        "L2_Orchestrator", 
        L2_router,
        {
            "MetaDataForModelDesign": "MetaDataForModelDesign",
            "ModelDesignNode": "ModelDesignNode",
            "HumanReview_ModelSelection": "HumanReview_ModelSelection",
            "TrainerNode": "TrainerNode",
            "EvaluatorNode": "EvaluatorNode",
            "EvaluatorAgentNode": "EvaluatorAgentNode",
            "HumanReview_ModelEvaluation": "HumanReview_ModelEvaluation",
            "HumanReview_InferenceDecision": "HumanReview_InferenceDecision",
            "InferenceNode": "InferenceNode",
            "L0_Orchestrator": "L0_Orchestrator"
        }
    )
#--------Layer 1 Edges ----------------------
    graph.add_edge("DatasetRegistry", "L1_Orchestrator")
    graph.add_edge("DescriptiveNode", "L1_Orchestrator")
    graph.add_edge("AnalysisNode", "L1_Orchestrator")
    graph.add_edge("Evaluator", "L1_Orchestrator")
    graph.add_edge("HumanReview_PreprocessConfig", "L1_Orchestrator")
    graph.add_edge("PreprocessNode", "L1_Orchestrator")
#--------Layer 2 Edges ----------------------
    graph.add_edge("MetaDataForModelDesign", "L2_Orchestrator")
    graph.add_edge("ModelDesignNode", "L2_Orchestrator")
    graph.add_edge("HumanReview_ModelSelection", "L2_Orchestrator")
    graph.add_edge("TrainerNode", "L2_Orchestrator")
    graph.add_edge("EvaluatorNode", "L2_Orchestrator")
    graph.add_edge("EvaluatorAgentNode", "L2_Orchestrator")
    graph.add_edge("HumanReview_ModelEvaluation", "L2_Orchestrator")
    graph.add_edge("HumanReview_InferenceDecision", "L2_Orchestrator")
    graph.add_edge("InferenceNode", "L2_Orchestrator")
#--------End Edges ----------------------
    graph.add_edge("SummarizerNode", END)

    
    app = graph.compile()  # No checkpointer for now


    # Initial State for the pipeline
    initial_state = {
        "from_stage": "user",
        "current_phase": "start",
        "phase_done": False,
        "dataset_path": "train.csv",
        "target_column": "Survived",
        "num_columns": [],
        "cat_columns": [],
        "task_type": "binary",
        "return_to_inference": False,  # Flag for cross-phase routing
        "logs": ["Starting the agentic ML pipeline..."]
    }

    # Invoke the graph (no config needed without checkpointer)
    result = app.invoke(initial_state, {"recursion_limit": 50})
    #-----------------------------Logging and printing----------------------------------------
    return app

if __name__ == "__main__":
    main()
    
