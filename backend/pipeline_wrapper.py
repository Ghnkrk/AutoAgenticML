"""
Pipeline Wrapper
Wraps the existing ML pipeline to add WebSocket updates and human interaction handling
"""
import asyncio
import sys
import os
from typing import Dict, Any, Callable
from pathlib import Path

# Add parent directory to path
import sys
import os
print(f"[DEBUG WRAPPER] Current dir: {os.getcwd()}")
print(f"[DEBUG WRAPPER] Absolute file: {os.path.abspath(__file__)}")
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"[DEBUG WRAPPER] Appending parent dir to path: {parent_dir}")
sys.path.append(parent_dir)
print(f"[DEBUG WRAPPER] sys.path: {sys.path}")

try:
    print(f"[DEBUG WRAPPER] Attempting to import main...")
    import main
    print(f"[DEBUG WRAPPER] Imported main from: {main.__file__}")
    from main import State
    print(f"[DEBUG WRAPPER] Imported State successfully")
except ImportError as e:
    print(f"[ERROR WRAPPER] Failed to import main: {e}")
    raise
except Exception as e:
    print(f"[ERROR WRAPPER] Error during import: {e}")
    raise
from Orchestrators import (
    L0_Orchestrator, L1_Orchestrator, L2_Orchestrator,
    L0_router, L1_router, L2_router
)
from Nodes import (
    datasetRegistry, descriptiveNode, analysisNode, DAEvaluator,
    preprocessNode, metaDataForModelDesign, modelDesignNode,
    TrainerNode, EvaluatorNode, EvaluatorAgentNode, InferenceNode, SummarizerNode
)
from HumanNodes import (
    HumanReview_PreprocessConfig, HumanReview_ModelSelection,
    HumanReview_ModelEvaluation, HumanReview_InferenceDecision
)


class PipelineWrapper:
    """Wraps pipeline execution with WebSocket updates and human interaction"""
    
    def __init__(self, pipeline_id: str, update_callback: Callable):
        self.pipeline_id = pipeline_id
        self.update_callback = update_callback
        self.human_input_queue = asyncio.Queue()
        self.current_node = None
        self.progress = 0
        
        # Node progress weights (approximate)
        self.node_weights = {
            "DatasetRegistry": 5,
            "DescriptiveNode": 5,
            "Evaluator": 5,
            "AnalysisNode": 10,
            "HumanReview_PreprocessConfig": 5,
            "PreprocessNode": 10,
            "MetaDataForModelDesign": 5,
            "ModelDesignNode": 5,
            "HumanReview_ModelSelection": 5,
            "TrainerNode": 20,
            "EvaluatorNode": 5,
            "EvaluatorAgentNode": 5,
            "HumanReview_ModelEvaluation": 5,
            "HumanReview_InferenceDecision": 5,
            "InferenceNode": 10,
            "SummarizerNode": 5
        }
        self.total_weight = sum(self.node_weights.values())
        self.completed_weight = 0
    
    async def run(self, dataset_path: str, target_column: str, task_type: str) -> Dict[str, Any]:
        """Run the pipeline with WebSocket updates"""
        
        # Determine actual task type
        if task_type == "auto":
            import pandas as pd
            df = pd.read_csv(dataset_path)
            if target_column in df.columns:
                unique_values = df[target_column].nunique()
                if unique_values == 2:
                    task_type = "binary"
                elif unique_values < 20 and df[target_column].dtype == 'object':
                    task_type = "multiclass"
                else:
                    task_type = "regression"
            else:
                task_type = "binary"
        
        # Initial state
        state: State = {
            "from_stage": "user",
            "current_phase": "start",
            "phase_done": False,
            "dataset_path": dataset_path,
            "target_column": target_column,
            "num_columns": [],
            "cat_columns": [],
            "task_type": task_type,
            "return_to_inference": False,
            "logs": ["Starting the agentic ML pipeline..."]
        }
        
        await self.send_update({
            "type": "status",
            "status": "running",
            "message": f"Pipeline started with task type: {task_type}",
            "progress": 0
        })
        
        # Node mapping
        NODE_MAP = {
            "DatasetRegistry": datasetRegistry,
            "DescriptiveNode": descriptiveNode,
            "AnalysisNode": analysisNode,
            "Evaluator": DAEvaluator,
            "PreprocessNode": preprocessNode,
            "MetaDataForModelDesign": metaDataForModelDesign,
            "ModelDesignNode": modelDesignNode,
            "TrainerNode": TrainerNode,
            "EvaluatorNode": EvaluatorNode,
            "EvaluatorAgentNode": EvaluatorAgentNode,
            "InferenceNode": InferenceNode,
            "SummarizerNode": SummarizerNode,
        }

        # Human interaction type mapping
        HUMAN_INPUT_MAP = {
            "HumanReview_PreprocessConfig": "preprocessing_config",
            "HumanReview_ModelSelection": "model_selection",
            "HumanReview_ModelEvaluation": "evaluation_decision",
            "HumanReview_InferenceDecision": "inference_decision"
        }
        
        # Execute pipeline validation loop
        try:
            # Initial L0 call
            state = await self.execute_node("L0_Orchestrator", L0_Orchestrator, state)
            
            # --- Main Execution Loop ---
            while True:
                # Route based on L0 state
                next_phase_node = L0_router(state)
                # Note: L0_router determines which sub-orchestrator to run
                
                print(f"[DEBUG WRAPPER] L0 Router -> {next_phase_node}")
                
                if next_phase_node == "SummarizerNode":
                    state = await self.execute_node("SummarizerNode", SummarizerNode, state)
                    break
                
                elif next_phase_node == "L1_Orchestrator":
                     # --- L1 Loop ---
                     while True:
                         # Run L1 to decide next step in prelim
                         state = await self.execute_node("L1_Orchestrator", L1_Orchestrator, state)
                         target_node_name = L1_router(state)
                         print(f"[DEBUG WRAPPER] L1 Router -> {target_node_name}")
                         
                         if target_node_name == "L0_Orchestrator":
                             break # Return to L0 to verify next phase
                             
                         # Execute Target Node
                         if target_node_name in HUMAN_INPUT_MAP:
                             input_type = HUMAN_INPUT_MAP[target_node_name]
                             state = await self.execute_human_node(target_node_name, input_type, state)
                         elif target_node_name in NODE_MAP:
                             state = await self.execute_node(target_node_name, NODE_MAP[target_node_name], state)
                         else:
                             raise ValueError(f"Unknown node routed by L1: {target_node_name}")
                             
                     # Re-run L0 to process phase completion
                     state = await self.execute_node("L0_Orchestrator", L0_Orchestrator, state)

                elif next_phase_node == "L2_Orchestrator":
                     # --- L2 Loop ---
                     while True:
                         state = await self.execute_node("L2_Orchestrator", L2_Orchestrator, state)
                         target_node_name = L2_router(state)
                         print(f"[DEBUG WRAPPER] L2 Router -> {target_node_name}")
                         
                         if target_node_name == "L0_Orchestrator":
                             break
                             
                         if target_node_name in HUMAN_INPUT_MAP:
                             input_type = HUMAN_INPUT_MAP[target_node_name]
                             state = await self.execute_human_node(target_node_name, input_type, state)
                         elif target_node_name in NODE_MAP:
                             state = await self.execute_node(target_node_name, NODE_MAP[target_node_name], state)
                         else: 
                             raise ValueError(f"Unknown node routed by L2: {target_node_name}")
                             
                     state = await self.execute_node("L0_Orchestrator", L0_Orchestrator, state)
                
                else:
                    raise ValueError(f"Unknown phase node from L0: {next_phase_node}")

            
            await self.send_update({
                "type": "status",
                "status": "completed",
                "message": "Pipeline completed successfully!",
                "progress": 100
            })
            
            return state
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Pipeline error: {error_details}")
            
            await self.send_update({
                "type": "error",
                "message": f"{str(e)}\n\nDetails: {error_details[:500]}"
            })
            raise
    
    async def execute_node(self, node_name: str, node_func: Callable, state: Dict) -> Dict:
        """Execute a node and send updates"""
        self.current_node = node_name
        
        await self.send_update({
            "type": "node_start",
            "node": node_name,
            "message": f"Executing {node_name}..."
        })
        
        # Execute node in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, node_func, state)
        
        # Merge result into state
        state.update(result)
        
        # Update progress
        self.completed_weight += self.node_weights.get(node_name, 5)
        self.progress = min(100, int((self.completed_weight / self.total_weight) * 100))
        
        # Send log if available
        if "logs" in result and result["logs"]:
            for log in result["logs"]:
                await self.send_update({
                    "type": "log",
                    "message": log
                })
        
        await self.send_update({
            "type": "node_complete",
            "node": node_name,
            "progress": self.progress
        })
        
        return state
    
    async def execute_human_node(self, node_name: str, input_type: str, state: Dict) -> Dict:
        """Execute a human review node with UI interaction"""
        self.current_node = node_name
        
        print(f"[DEBUG WRAPPER] Entering execute_human_node: {node_name}")
        print(f"[DEBUG WRAPPER] State keys: {list(state.keys())}")

        # Prepare data for UI
        if input_type == "preprocessing_config":
            config = state.get("analysisEvaluator_response")
            
            if not config:
                print(f"[CRITICAL ERROR] analysisEvaluator_response is MISSING! Using fallback config.")
                # Fallback to prevent frontend crash
                config = {
                    "features": {"drop": [], "uncertain": [], "keep": []},
                    "scaling": {"method": "standard"},
                    "dimensionality_reduction": {"use_pca": False, "variance_threshold": 0.95},
                    "missing_values": {"drop_threshold": 0.5, "impute": {"numerical": "mean", "categorical": "mode"}},
                    "encoding": {"one_hot": [], "ordinal": [], "target": []},
                    "task_type": state.get("task_type", "binary"),
                    "notes": "Fallback config due to missing analysis response"
                }
            
            request_data = {
                "config": config,
                "num_columns": state.get("num_columns", []),
                "cat_columns": state.get("cat_columns", [])
            }
        
        elif input_type == "model_selection":
            design_response = state.get("modelDesign_response", {})
            request_data = {
                "models": design_response.get("models", []),
                "primary_metric": design_response.get("primary_metric", "accuracy"),
                "task_type": state.get("task_type")
            }
        
        elif input_type == "evaluation_decision":
            request_data = {
                "evaluation_results": state.get("evaluation_results", {}),
                "agent_response": state.get("evaluator_agent_response", {})
            }
        
        elif input_type == "inference_decision":
            request_data = {
                "trained_models": state.get("trained_models", [])
            }
        
        else:
            request_data = {}
        
        # Send human input request
        await self.send_update({
            "type": "human_input_request",
            "input_type": input_type,
            "node": node_name,
            "data": request_data
        })
        
        # Wait for response
        response = await self.human_input_queue.get()
        
        # Apply response to state
        state = self.apply_human_response(input_type, response, state)
        
        # Update progress
        self.completed_weight += self.node_weights.get(node_name, 5)
        self.progress = int((self.completed_weight / self.total_weight) * 100)
        
        await self.send_update({
            "type": "node_complete",
            "node": node_name,
            "progress": self.progress
        })
        
        return state
    
    def apply_human_response(self, input_type: str, response: Dict, state: Dict) -> Dict:
        """Apply human response to state"""
        if input_type == "preprocessing_config":
            if response.get("accepted"):
                state["analysisEvaluator_response"] = response.get("config")
                state["from_stage"] = "human"
        
        elif input_type == "model_selection":
            if response.get("accepted"):
                state["modelDesign_response"] = {
                    "models": response.get("models"),
                    "primary_metric": response.get("primary_metric"),
                    "notes": "Human reviewed and approved"
                }
                state["from_stage"] = "human_model_selection"
        
        elif input_type == "evaluation_decision":
            if response.get("decision") == "retrain":
                state["from_stage"] = "human_evaluation_retrain"
            else:
                state["from_stage"] = "human_evaluation"
        
        elif input_type == "inference_decision":
            if response.get("run_inference"):
                # Convert relative path to absolute path
                from pathlib import Path
                test_path = response.get("test_dataset_path", "")
                # If it's a relative path like "uploads/file.csv", make it absolute
                if not Path(test_path).is_absolute():
                    base_dir = Path(__file__).parent.parent
                    test_path = str(base_dir / test_path)
                
                state["test_dataset_path"] = test_path
                state["selected_models_for_inference"] = response.get("selected_models")
                state["return_to_inference"] = True
                state["has_target"] = False
            state["from_stage"] = "human_inference"
        
        return state
    
    async def send_update(self, data: Dict):
        """Send update via callback"""
        await self.update_callback(data)
    
    async def receive_human_response(self, response: Dict):
        """Receive human response from UI"""
        await self.human_input_queue.put(response)
