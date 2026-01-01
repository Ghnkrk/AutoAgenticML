




import json
from langchain_groq import ChatGroq
from promptTemplate import (
    L0_ORCHESTRATOR_PROMPT, 
    L1_ORCHESTRATOR_PROMPT, 
    L2_ORCHESTRATOR_PROMPT
)
from ModelResponseSchema import (
    L0_OrchestratorResponse,
    L1_OrchestratorResponse,
    L2_OrchestratorResponse,
    parse_llm_response
)


def L0_Orchestrator(state):
            
        model = ChatGroq(
            model="openai/gpt-oss-20b",
            model_kwargs={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "L0_orchestrator_response",
                        "strict": True,  
                        "schema": L0_OrchestratorResponse.model_json_schema()
                    }
                }
            }
        )
        
        input_context = json.dumps({
            "current_phase": state.get("current_phase", "start"),
            "phase_done": state.get("phase_done", False),
            "from_stage": state["from_stage"]
        })

        input_msgs = [
            {"role": "system", "content": L0_ORCHESTRATOR_PROMPT},
            {"role": "user", "content": input_context}
        ]
        
        # ROBUSTNESS: Add fallback logic if LLM fails
        try:
            response = model.invoke(input_msgs)
            L0_Orchestrator_response = parse_llm_response(response, L0_OrchestratorResponse)
        except Exception as e:
            print(f"\n⚠️  L0 Orchestrator LLM failed: {e}")
            print(f"⚠️  Using fallback routing based on current_phase='{state['current_phase']}'")
            
            # Fallback routing: start→prelim→modeling→end
            fallback_map = {
                "start": "prelim",
                "prelim": "modeling",
                "modeling": "end"
            }
            
            next_phase = fallback_map.get(state.get("current_phase", "start"), "end")
            
            class FallbackResponse:
                def __init__(self, next_phase):
                    self.next_phase = next_phase
                    self.notes = f"Fallback routing from {state.get('current_phase', 'start')}"
            
            L0_Orchestrator_response = FallbackResponse(next_phase)
        
        # Handle test preprocessing routing for inference
        return_to_inference = state.get("return_to_inference", False)
        current_phase = state.get("current_phase", "start")
        
        if return_to_inference:
            # Override LLM decision for test preprocessing flow
            if current_phase == "modeling":
                # Coming from L2 human_inference, route to prelim for test preprocessing
                next_phase = "prelim"
                notes = "Routing to prelim for test data preprocessing before inference"
            elif current_phase == "prelim":
                # Test preprocessing done, route back to modeling for inference
                next_phase = "modeling"
                notes = "Test preprocessing complete, returning to modeling for inference"
            else:
                next_phase = L0_Orchestrator_response.next_phase
                notes = L0_Orchestrator_response.notes
        else:
            # Normal flow - use LLM decision
            next_phase = L0_Orchestrator_response.next_phase
            notes = L0_Orchestrator_response.notes
        
        result = {
            "current_phase": next_phase,
            "next_phase": next_phase,
            "logs": [f"L0 Orchestrator response : {{'next_phase': '{next_phase}', 'notes': '{notes}'}}"]
        }
        
        # Reset from_stage when transitioning between phases (but not on initial start)
        if current_phase in ["prelim", "modeling"] and next_phase != current_phase:
            result["from_stage"] = next_phase

        return result


def L0_router(state):

        next_phase = state["next_phase"]

        # Check if we're returning from test preprocessing for inference
        return_to_inference = state.get("return_to_inference", False)
        
        if return_to_inference and next_phase == "prelim":
            # Coming from L2 for test preprocessing
            return "L1_Orchestrator"
        elif return_to_inference and next_phase == "modeling":
            # Test preprocessing done, return to L2 for inference
            return "L2_Orchestrator"
        elif next_phase == "prelim":
            return "L1_Orchestrator"

        elif next_phase == "modeling":
            return "L2_Orchestrator"

        elif next_phase == "end":
            return "SummarizerNode"
        
        else:
            raise ValueError(f"Unknown next_phase in L0: {next_phase}")

def L1_Orchestrator(state):
        """
        Orchestrator with guaranteed structured output.
        Uses Groq's strict mode for 100% schema adherence.
        """
        model = ChatGroq(
            model="openai/gpt-oss-20b",
            model_kwargs={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "L1_orchestrator_response",
                        "strict": True,  
                        "schema": L1_OrchestratorResponse.model_json_schema()
                    }
                }
            }
        )
        
        input_msgs = [
            {"role": "system", "content": L1_ORCHESTRATOR_PROMPT},
            {"role": "user", "content": state["from_stage"]}
        ]
        
        # ROBUSTNESS: Add fallback logic if LLM fails
        try:
            response = model.invoke(input_msgs)
            L1_Orchestrator_response = parse_llm_response(response, L1_OrchestratorResponse)
        except Exception as e:
            print(f"\n⚠️  L1 Orchestrator LLM failed: {e}")
            print(f"⚠️  Using fallback routing based on from_stage='{state['from_stage']}'")
            
            # Fallback routing for prelim phase
            fallback_map = {
                "user": "registry",
                "registry": "descriptive",
                "descriptive": "descriptive_evaluator",
                "descriptive_evaluator": "analysis",
                "analysis": "analysis_evaluator",
                "analysis_evaluator": "preprocess",
                "preprocess": "end",
                "test_registry": "preprocess"
            }
            
            next_stage = fallback_map.get(state.get("from_stage", "user"), "end")
            
            class FallbackResponse:
                def __init__(self, next_stage):
                    self.next_stage = next_stage
                    self.preprocess_mode = "train"
            
            L1_Orchestrator_response = FallbackResponse(next_stage)


        # Handle test preprocessing flow (bypass descriptive/analysis)
        return_to_inference = state.get("return_to_inference", False)
        from_stage = state["from_stage"]
        
        if return_to_inference:
            # Test preprocessing flow: human_inference → test_registry → preprocess → end
            # When coming from L2 for test preprocessing, from_stage will be "human_inference"
            if from_stage == "human_inference":
                next_stage = "test_registry"
            elif from_stage == "test_registry":
                next_stage = "preprocess"
            elif from_stage == "preprocess":
                next_stage = "end"
            else:
                # Fallback to LLM response
                next_stage = L1_Orchestrator_response.next_stage
            
            phase_done = (next_stage == "end")
        else:
            # Normal training flow
            next_stage = L1_Orchestrator_response.next_stage
            phase_done = (next_stage == "end")
            
        print(f"\n[DEBUG L1] from_stage={from_stage} -> to_stage={next_stage}")
        
        result = {
            "to_stage": next_stage,
            "phase_done": phase_done,
            "preprocess_mode": "test" if (return_to_inference and next_stage == "preprocess") else L1_Orchestrator_response.preprocess_mode,
            "logs": [f"Orchestrator response : {{'next_stage': '{next_stage}', 'notes': 'Test preprocessing flow' if return_to_inference else 'Training flow'}}"]
        }
        
        # Preserve column lists
        if "num_columns" in state:
            result["num_columns"] = state["num_columns"]
        if "cat_columns" in state:
            result["cat_columns"] = state["cat_columns"]
            
        return result

def L1_router(state):
        to_stage = state["to_stage"]
        print(f"[DEBUG L1 Router] Routing to_stage: {to_stage}")

        if to_stage == "registry":
            return "DatasetRegistry"
        
        elif to_stage == "test_registry":
            return "DatasetRegistry"  # Reuse same registry node

        elif to_stage == "descriptive":
            return "DescriptiveNode"

        elif to_stage == "descriptive_evaluator":
            return "Evaluator"

        elif to_stage == "analysis":
            return "AnalysisNode"

        elif to_stage == "analysis_evaluator":
            return "Evaluator"

        elif to_stage == "human":
            return "HumanReview_PreprocessConfig"

        elif to_stage == "preprocess":
            return "PreprocessNode"

        elif to_stage == "end":
            return "L0_Orchestrator"

        else:   
            raise ValueError(f"Unknown to_stage in L1: {to_stage}")

def L2_Orchestrator(state):
        """
        Orchestrator with guaranteed structured output.
        Uses Groq's strict mode for 100% schema adherence.
        """
        model = ChatGroq(
            model="openai/gpt-oss-20b",
            model_kwargs={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "L2_orchestrator_response",
                        "strict": True,  
                        "schema": L2_OrchestratorResponse.model_json_schema()
                    }
                }
            }
        )
        
        input_msgs = [
            {"role": "system", "content": L2_ORCHESTRATOR_PROMPT},
            {"role": "user", "content": state["from_stage"]}
        ]
        
        # ROBUSTNESS: Add fallback logic if LLM fails
        try:
            response = model.invoke(input_msgs)
            L2_Orchestrator_response = parse_llm_response(response, L2_OrchestratorResponse)
        except Exception as e:
            print(f"\n⚠️  L2 Orchestrator LLM failed: {e}")
            print(f"⚠️  Using fallback routing based on from_stage='{state['from_stage']}'")
            
            # Fallback routing map
            fallback_map = {
                "modeling": "model_design_metadata",
                "model_design_metadata": "model_design",
                "model_design": "human_model_selection",
                "human_model_selection": "training",
                "training": "evaluation",
                "evaluation": "evaluator_agent",
                "evaluator_agent": "human_evaluation",
                "human_evaluation": "human_inference",
                "human_inference": "end",
                "inference": "end"
            }
            
            next_stage = fallback_map.get(state["from_stage"], "end")
            
            # Create a fallback response object
            class FallbackResponse:
                def __init__(self, next_stage):
                    self.next_stage = next_stage
                    self.notes = f"Fallback routing from {state['from_stage']}"
            
            L2_Orchestrator_response = FallbackResponse(next_stage)

        # Handle test preprocessing routing for inference
        return_to_inference = state.get("return_to_inference", False)
        from_stage = state.get("from_stage", "")
        
        # DEBUG: Print state values
        print(f"\n[DEBUG L2] return_to_inference={return_to_inference}, from_stage={from_stage}")
        
        if return_to_inference:
            if from_stage == "human_inference":
                # User selected inference, route to L0 for test preprocessing
                next_stage = "end"
                phase_done = True  # Hand control to L0
                notes = "Routing to L0 for test data preprocessing"
                clear_flag = False
            elif from_stage == "preprocess":
                # Returning from L1 test preprocessing directly, route to inference
                next_stage = "inference"
                phase_done = False
                notes = "Routing to inference after test preprocessing"
                clear_flag = True  # Clear flag after routing to inference
            elif from_stage == "modeling":
                # Returning from L0 after test preprocessing completed, route to inference
                next_stage = "inference"
                phase_done = False
                notes = "Routing to inference after test preprocessing (via L0)"
                clear_flag = True  # Clear flag after routing to inference
            else:
                # Fallback to LLM decision
                next_stage = L2_Orchestrator_response.next_stage
                phase_done = (next_stage == "end")
                notes = L2_Orchestrator_response.notes
                clear_flag = False
        else:
            # Normal flow - use LLM decision, but override for initial modeling entry
            if from_stage == "modeling":
                # Just entered modeling phase from L0, must start with metadata
                next_stage = "model_design_metadata"
                phase_done = False
                notes = "Starting modeling phase with metadata creation"
                clear_flag = False
            else:
                next_stage = L2_Orchestrator_response.next_stage
                phase_done = (next_stage == "end")
                notes = L2_Orchestrator_response.notes
                clear_flag = False
        
        result = {
            "to_stage": next_stage,
            "phase_done": phase_done,
            "logs": [f"Orchestrator response : {{'next_stage': '{next_stage}', 'notes': '{notes}'}}"]
        }
        
        # Clear return_to_inference flag after routing to inference
        if clear_flag:
            result["return_to_inference"] = False
        
        return result

def L2_router(state):
        to_stage = state["to_stage"]

        if to_stage == "model_design_metadata":
            return "MetaDataForModelDesign"

        elif to_stage == "model_design":
            return "ModelDesignNode"

        elif to_stage == "human_model_selection":
            return "HumanReview_ModelSelection"

        elif to_stage == "training":
            return "TrainerNode"
        
        elif to_stage == "evaluation":
            return "EvaluatorNode"
        
        elif to_stage == "evaluator_agent":
            return "EvaluatorAgentNode"
        
        elif to_stage == "human_evaluation":
            return "HumanReview_ModelEvaluation"
        
        elif to_stage == "human_inference":
            return "HumanReview_InferenceDecision"
        
        elif to_stage == "inference":
            return "InferenceNode"

        elif to_stage == "end":
            return "L0_Orchestrator"

        else:   
            raise ValueError(f"Unknown to_stage in L2: {to_stage}")  