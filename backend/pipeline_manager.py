"""
Pipeline Manager
Handles async pipeline execution, state management, and WebSocket communication
"""
import asyncio
import uuid
from typing import Dict, Optional, Any
from datetime import datetime
from fastapi import WebSocket
import sys
import os

# Add parent directory to path to import pipeline modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PipelineState:
    """Represents the state of a running pipeline"""
    def __init__(self, pipeline_id: str):
        self.pipeline_id = pipeline_id
        self.status = "initializing"  # initializing, running, waiting_human, completed, failed
        self.current_node = None
        self.progress = 0
        self.message = ""
        self.logs = []
        self.human_input_queue = asyncio.Queue()
        self.human_response_queue = asyncio.Queue()
        self.websocket: Optional[WebSocket] = None
        self.task: Optional[asyncio.Task] = None
        self.result = None
        self.error = None
        self.wrapper = None  # PipelineWrapper instance
        self.pending_human_request = None # Store data for pending human input


class PipelineManager:
    """Manages multiple pipeline executions"""
    
    def __init__(self):
        self.pipelines: Dict[str, PipelineState] = {}
    
    async def start_pipeline(self, dataset_path: str, target_column: str, task_type: str) -> str:
        """Start a new pipeline execution"""
        pipeline_id = str(uuid.uuid4())
        print(f"[DEBUG] start_pipeline called for {pipeline_id}")
        print(f"[DEBUG] Dataset: {dataset_path}, Target: {target_column}, Task: {task_type}")
        
        state = PipelineState(pipeline_id)
        self.pipelines[pipeline_id] = state
        
        # Start pipeline in background
        print(f"[DEBUG] Creating background task...")
        state.task = asyncio.create_task(
            self._run_pipeline(pipeline_id, dataset_path, target_column, task_type)
        )
        print(f"[DEBUG] Task created: {state.task}")
        
        return pipeline_id
    
    async def _run_pipeline(self, pipeline_id: str, dataset_path: str, target_column: str, task_type: str):
        """Run the pipeline asynchronously"""
        state = self.pipelines[pipeline_id]
        
        try:
            print(f"[DEBUG] Importing PipelineWrapper...")
            from pipeline_wrapper import PipelineWrapper
            print(f"[DEBUG] Starting pipeline {pipeline_id}")
            state.status = "running"
            await self._send_update(pipeline_id, {
                "type": "status",
                "status": "running",
                "message": "Pipeline started"
            })
            
            # Create pipeline wrapper with update callback
            async def update_callback(data):
                print(f"[DEBUG] Sending update: {data.get('type')}")
                await self._send_update(pipeline_id, data)
            
            wrapper = PipelineWrapper(pipeline_id, update_callback)
            
            # Store wrapper for human response handling
            state.wrapper = wrapper
            
            print(f"[DEBUG] Running wrapper for {pipeline_id}")
            # Run pipeline
            result = await wrapper.run(dataset_path, target_column, task_type)
            
            print(f"[DEBUG] Pipeline completed for {pipeline_id}")
            state.result = result
            state.status = "completed"
            state.progress = 100
            await self._send_update(pipeline_id, {
                "type": "status",
                "status": "completed",
                "message": "Pipeline completed successfully",
                "summary": result.get("pipeline_summary", "")
            })
            
        except Exception as e:
            print(f"[ERROR] Pipeline {pipeline_id} failed: {e}")
            import traceback
            traceback.print_exc()
            state.status = "failed"
            state.error = str(e)
            await self._send_update(pipeline_id, {
                "type": "error",
                "message": str(e)
            })
    
    async def send_human_response(self, pipeline_id: str, response_data: dict) -> bool:
        """Send human response to waiting pipeline"""
        if pipeline_id not in self.pipelines:
            return False
        
        state = self.pipelines[pipeline_id]
        
        # Forward response to pipeline wrapper
        if state.wrapper:
            await state.wrapper.receive_human_response(response_data)
        
        # Resume pipeline
        state.status = "running"
        await self._send_update(pipeline_id, {
            "type": "status",
            "status": "running",
            "message": "Resumed after human input"
        })
        
        return True
    
    def get_status(self, pipeline_id: str) -> Optional[dict]:
        """Get current pipeline status"""
        if pipeline_id not in self.pipelines:
            return None
        
        state = self.pipelines[pipeline_id]
        return {
            "pipeline_id": pipeline_id,
            "status": state.status,
            "current_node": state.current_node,
            "progress": state.progress,
            "message": state.message,
            "result": state.result,
            "pending_human_request": state.pending_human_request
        }
    
    def register_websocket(self, pipeline_id: str, websocket: WebSocket):
        """Register WebSocket for pipeline updates"""
        if pipeline_id in self.pipelines:
            self.pipelines[pipeline_id].websocket = websocket
    
    def unregister_websocket(self, pipeline_id: str):
        """Unregister WebSocket"""
        if pipeline_id in self.pipelines:
            self.pipelines[pipeline_id].websocket = None
    
    async def _send_update(self, pipeline_id: str, data: dict):
        """Send update via WebSocket"""
        if pipeline_id not in self.pipelines:
            return
        
        state = self.pipelines[pipeline_id]
        
        # Add to logs
        state.logs.append({
            "timestamp": datetime.now().isoformat(),
            "data": data
        })
        
        # Track pending human request
        if data.get("type") == "human_input_request":
            state.pending_human_request = data
            state.status = "waiting_human"
        elif data.get("type") == "node_complete" and state.status == "waiting_human" and data.get("node") == state.current_node:
             # Only clear if we are completing the node we were waiting for
             # But actually, node_complete comes after response is processed.
             # So safely clear it.
            state.pending_human_request = None
            state.status = "running"
        
        # Send via WebSocket if connected
        if state.websocket:
            try:
                await state.websocket.send_json(data)
            except Exception as e:
                print(f"Failed to send WebSocket update: {e}")
    
    async def request_human_input(self, pipeline_id: str, input_type: str, data: dict) -> dict:
        """Request human input and wait for response"""
        if pipeline_id not in self.pipelines:
            raise ValueError("Pipeline not found")
        
        state = self.pipelines[pipeline_id]
        state.status = "waiting_human"
        
        # Send request via WebSocket
        await self._send_update(pipeline_id, {
            "type": "human_input_request",
            "input_type": input_type,
            "data": data
        })
        
        # Wait for response
        response = await state.human_response_queue.get()
        
        return response
