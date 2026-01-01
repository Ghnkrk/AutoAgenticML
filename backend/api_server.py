"""
FastAPI Backend for ML Pipeline
Provides REST API and WebSocket endpoints for pipeline execution
"""
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import shutil
from pathlib import Path
import json
from datetime import datetime

# Import pipeline manager
from pipeline_manager import PipelineManager

app = FastAPI(title="AgenticML API", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline manager
pipeline_manager = PipelineManager()

# Determine base directory (works in Docker and local)
# If running from backend/, go up one level; if from project root, stay there
BASE_DIR = Path(__file__).parent.parent  # Go to project root from backend/

# Directories
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"
PREDICTIONS_DIR = BASE_DIR / "predictions"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Request/Response Models
# ============================================================================

class PipelineStartRequest(BaseModel):
    dataset_filename: str
    target_column: str
    task_type: str  # "auto", "binary", "multiclass", "regression"


class PipelineStatusResponse(BaseModel):
    pipeline_id: str
    status: str  # "running", "waiting_human", "completed", "failed"
    current_node: Optional[str]
    progress: int  # 0-100
    message: Optional[str]


class HumanResponseRequest(BaseModel):
    response_data: dict


# ============================================================================
# File Upload/Download Endpoints
# ============================================================================

@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a dataset file"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Save file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "filename": file.filename,
            "size": file_path.stat().st_size,
            "message": "File uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files/download/{file_type}/{filename}")
async def download_file(file_type: str, filename: str):
    """Download a file (model, prediction, log, metadata)"""
    try:
        # Determine directory based on file type
        if file_type == "model":
            file_path = TRAINED_MODELS_DIR / filename
        elif file_type == "prediction":
            file_path = PREDICTIONS_DIR / filename
        elif file_type == "log":
            file_path = Path("..") / filename
        elif file_type == "metadata":
            file_path = TRAINED_MODELS_DIR / filename
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files/list/{file_type}")
async def list_files(file_type: str):
    """List available files for download"""
    try:
        if file_type == "models":
            files = list(TRAINED_MODELS_DIR.glob("*.pkl"))
            # Also get corresponding metadata files
            result = []
            for pkl_file in files:
                metadata_file = pkl_file.with_suffix('.json').name.replace('.pkl', '_metadata.json')
                metadata_path = TRAINED_MODELS_DIR / metadata_file
                result.append({
                    "model": pkl_file.name,
                    "metadata": metadata_file if metadata_path.exists() else None,
                    "size": pkl_file.stat().st_size,
                    "modified": datetime.fromtimestamp(pkl_file.stat().st_mtime).isoformat()
                })
            return result
        
        elif file_type == "predictions":
            files = list(PREDICTIONS_DIR.glob("*.csv"))
            return [{
                "filename": f.name,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            } for f in files]
        
        else:
            raise HTTPException(status_code=400, detail="Invalid file type")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Pipeline Endpoints
# ============================================================================

@app.post("/api/pipeline/start")
async def start_pipeline(request: PipelineStartRequest):
    """Start a new pipeline execution"""
    try:
        print(f"[DEBUG] Endpoint /api/pipeline/start called with: {request}")
        
        # Validate dataset file exists
        dataset_path = UPLOAD_DIR / request.dataset_filename
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        print(f"[DEBUG] Dataset path: {dataset_path}")
        
        # Start pipeline
        pipeline_id = await pipeline_manager.start_pipeline(
            dataset_path=str(dataset_path),
            target_column=request.target_column,
            task_type=request.task_type
        )
        
        print(f"[DEBUG] Pipeline started with ID: {pipeline_id}")
        
        return {
            "pipeline_id": pipeline_id,
            "message": "Pipeline started successfully"
        }
    
    except Exception as e:
        print(f"[ERROR] Failed to start pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pipeline/status/{pipeline_id}")
async def get_pipeline_status(pipeline_id: str):
    """Get current pipeline status"""
    try:
        status = pipeline_manager.get_status(pipeline_id)
        if not status:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return status
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipeline/respond/{pipeline_id}")
async def send_human_response(pipeline_id: str, request: HumanResponseRequest):
    """Send human response to pipeline"""
    print(f"[DEBUG API] send_human_response called for {pipeline_id}")
    print(f"[DEBUG API] Request data keys: {request.response_data.keys()}")
    try:
        success = await pipeline_manager.send_human_response(
            pipeline_id=pipeline_id,
            response_data=request.response_data
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Pipeline not found or not waiting for input")
        
        return {"message": "Response received, pipeline resumed"}
    
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[ERROR API] Detailed Traceback:\n{err_msg}")
        raise HTTPException(status_code=500, detail=f"{str(e)} : {err_msg}")


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/pipeline/{pipeline_id}")
async def websocket_endpoint(websocket: WebSocket, pipeline_id: str):
    """WebSocket for real-time pipeline updates"""
    await websocket.accept()
    
    try:
        # Register WebSocket connection
        pipeline_manager.register_websocket(pipeline_id, websocket)
        
        # Keep connection alive and send updates
        while True:
            # Wait for messages (ping/pong to keep alive)
            try:
                data = await websocket.receive_text()
                # Echo back for keep-alive
                await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    
    finally:
        # Unregister on disconnect
        pipeline_manager.unregister_websocket(pipeline_id)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Serve Frontend (for development)
# ============================================================================

# Mount static files
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disabled to prevent killing background tasks
    )
