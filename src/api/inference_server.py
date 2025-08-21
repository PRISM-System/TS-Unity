#!/usr/bin/env python3
"""
FastAPI Inference Server for TS-Unity

This server provides REST API endpoints for time series inference,
supporting forecasting, anomaly detection, imputation, and classification.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Optional, Dict, Any, Union
import time
import json
from datetime import datetime

# Add src directory to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from config.base_config import BaseConfig, AnomalyDetectionConfig, ForecastingConfig
from core.pipeline import InferencePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TS-Unity Inference API",
    description="Time Series Unified Framework Inference API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global inference pipeline
inference_pipeline: Optional[InferencePipeline] = None
current_config: Optional[BaseConfig] = None


# Pydantic models for API requests/responses
class InferenceRequest(BaseModel):
    """Base inference request model."""
    task_type: str = Field(..., description="Task type: forecasting, anomaly_detection, imputation, classification")
    data: List[List[float]] = Field(..., description="Time series data as 2D array (time_steps, features)")
    num_steps: Optional[int] = Field(1, description="Number of steps to predict ahead (for forecasting)")
    window_size: Optional[int] = Field(None, description="Window size for sliding window inference")
    stride: Optional[int] = Field(1, description="Stride for sliding window inference")


class ForecastingRequest(InferenceRequest):
    """Forecasting-specific request model."""
    task_type: str = Field("forecasting", description="Task type (fixed to forecasting)")
    num_steps: int = Field(..., description="Number of steps to predict ahead")


class AnomalyDetectionRequest(InferenceRequest):
    """Anomaly detection-specific request model."""
    task_type: str = Field("anomaly_detection", description="Task type (fixed to anomaly detection)")
    threshold: Optional[float] = Field(None, description="Anomaly threshold for binary classification")


class InferenceResponse(BaseModel):
    """Base inference response model."""
    task_type: str
    status: str
    timestamp: str
    inference_time: float
    data_shape: List[int]
    predictions: Union[List[List[float]], List[float]]
    metadata: Dict[str, Any]


class ForecastingResponse(InferenceResponse):
    """Forecasting-specific response model."""
    num_steps: int
    forecast_horizon: List[str]


class AnomalyDetectionResponse(InferenceResponse):
    """Anomaly detection-specific response model."""
    anomaly_scores: List[List[float]]
    anomalies_detected: int
    anomaly_rate: float
    threshold_used: Optional[float]
    detection_method: Optional[Dict[str, Any]] = Field(None, description="Information about the detection method used")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    task_type: Optional[str]
    model_loaded: bool


class ModelLoadRequest(BaseModel):
    """Model loading request model."""
    task_type: str = Field(..., description="Task type to load")
    checkpoint_path: str = Field(..., description="Path to model checkpoint")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Configuration overrides")


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "TS-Unity Inference API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        task_type=current_config.task_name if current_config else None,
        model_loaded=inference_pipeline is not None
    )


@app.post("/load_model", response_model=Dict[str, str])
async def load_model(request: ModelLoadRequest):
    """Load a trained model for inference."""
    global inference_pipeline, current_config
    
    try:
        logger.info(f"Loading model for task: {request.task_type}")
        
        # Create configuration based on task type
        if request.task_type == "anomaly_detection":
            config = AnomalyDetectionConfig(
                task_name='anomaly_detection',
                model='AnomalyTransformer',
                seq_len=100,
                enc_in=7,
                dec_in=7,
                c_out=7,
                anomaly_ratio=0.1,
                win_size=100
            )
        elif request.task_type in ["long_term_forecast", "short_term_forecast"]:
            config = ForecastingConfig(
                task_name=request.task_type,
                model='Autoformer',
                seq_len=100,
                enc_in=7,
                dec_in=7,
                c_out=7,
                pred_len=24
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported task type: {request.task_type}")
        
        # Apply config overrides if provided
        if request.config_overrides:
            for key, value in request.config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Check if checkpoint exists
        if not os.path.exists(request.checkpoint_path):
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {request.checkpoint_path}")
        
        # Initialize inference pipeline
        inference_pipeline = InferencePipeline(config, request.checkpoint_path)
        current_config = config
        
        logger.info(f"Model loaded successfully for task: {request.task_type}")
        
        return {
            "status": "success",
            "message": f"Model loaded successfully for task: {request.task_type}",
            "task_type": request.task_type,
            "checkpoint_path": request.checkpoint_path
        }
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Run inference on time series data."""
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")
    
    try:
        # Validate input data
        if not request.data or len(request.data) == 0:
            raise HTTPException(status_code=400, detail="Input data cannot be empty")
        
        # Convert to numpy array
        data_array = np.array(request.data)
        
        # Validate data shape
        if data_array.ndim != 2:
            raise HTTPException(status_code=400, detail="Input data must be 2D (time_steps, features)")
        
        if data_array.shape[1] != current_config.enc_in:
            raise HTTPException(
                status_code=400, 
                detail=f"Input features {data_array.shape[1]} don't match expected {current_config.enc_in}"
            )
        
        # Run inference
        start_time = time.time()
        
        if request.window_size:
            # Sliding window inference
            results = inference_pipeline.predict_with_sliding_window(
                data_array, 
                window_size=request.window_size,
                stride=request.stride,
                num_steps=request.num_steps
            )
            predictions = results['predictions']
        else:
            # Batch inference
            predictions = inference_pipeline.predict_batch(data_array, num_steps=request.num_steps)
        
        inference_time = time.time() - start_time
        
        # Create response based on task type
        if request.task_type in ["long_term_forecast", "short_term_forecast"]:
            return ForecastingResponse(
                task_type=request.task_type,
                status="success",
                timestamp=datetime.now().isoformat(),
                inference_time=inference_time,
                data_shape=list(data_array.shape),
                predictions=predictions.tolist(),
                metadata={
                    "num_steps": request.num_steps,
                    "window_size": request.window_size,
                    "stride": request.stride
                },
                num_steps=request.num_steps,
                forecast_horizon=[f"t+{i+1}" for i in range(request.num_steps)]
            )
        elif request.task_type == "anomaly_detection":
            # Calculate anomaly statistics
            flat_scores = predictions.flatten()
            threshold = request.threshold or (np.mean(flat_scores) + 2 * np.std(flat_scores))
            anomalies_detected = np.sum(flat_scores > threshold)
            anomaly_rate = anomalies_detected / len(flat_scores) * 100
            
            return AnomalyDetectionResponse(
                task_type=request.task_type,
                status="success",
                timestamp=datetime.now().isoformat(),
                inference_time=inference_time,
                data_shape=list(data_array.shape),
                predictions=predictions.tolist(),
                metadata={
                    "window_size": request.window_size,
                    "stride": request.stride
                },
                anomaly_scores=predictions.tolist(),
                anomalies_detected=int(anomalies_detected),
                anomaly_rate=float(anomaly_rate),
                threshold_used=float(threshold)
            )
        else:
            # Generic response for other tasks
            return InferenceResponse(
                task_type=request.task_type,
                status="success",
                timestamp=datetime.now().isoformat(),
                inference_time=inference_time,
                data_shape=list(data_array.shape),
                predictions=predictions.tolist(),
                metadata={
                    "num_steps": request.num_steps,
                    "window_size": request.window_size,
                    "stride": request.stride
                }
            )
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/forecast", response_model=ForecastingResponse)
async def run_forecasting(request: ForecastingRequest):
    """Run forecasting inference."""
    # Convert to generic inference request
    inference_request = InferenceRequest(
        task_type=request.task_type,
        data=request.data,
        num_steps=request.num_steps,
        window_size=request.window_size,
        stride=request.stride
    )
    
    # Run inference
    response = await run_inference(inference_request)
    
    # Convert to forecasting response
    return ForecastingResponse(
        task_type=response.task_type,
        status=response.status,
        timestamp=response.timestamp,
        inference_time=response.inference_time,
        data_shape=response.data_shape,
        predictions=response.predictions,
        metadata=response.metadata,
        num_steps=request.num_steps,
        forecast_horizon=[f"t+{i+1}" for i in range(request.num_steps)]
    )


@app.post("/detect_anomalies", response_model=AnomalyDetectionResponse)
async def run_anomaly_detection(request: AnomalyDetectionRequest):
    """Run anomaly detection inference."""
    # Convert to generic inference request
    inference_request = InferenceRequest(
        task_type=request.task_type,
        data=request.data,
        num_steps=1,  # Anomaly detection doesn't use num_steps
        window_size=request.window_size,
        stride=request.stride
    )
    
    # Run inference
    response = await run_inference(inference_request)
    
    # Convert to anomaly detection response
    predictions = np.array(response.predictions)
    flat_scores = predictions.flatten()
    threshold = request.threshold or (np.mean(flat_scores) + 2 * np.std(flat_scores))
    anomalies_detected = np.sum(flat_scores > threshold)
    anomaly_rate = anomalies_detected / len(flat_scores) * 100
    
    # Get detection method information
    detection_method = None
    if hasattr(response, 'metadata') and response.metadata:
        detection_method = response.metadata.get('detection_method')
    
    return AnomalyDetectionResponse(
        task_type=response.task_type,
        status=response.status,
        timestamp=response.timestamp,
        inference_time=response.inference_time,
        data_shape=response.data_shape,
        predictions=response.predictions,
        metadata=response.metadata,
        anomaly_scores=response.predictions,
        anomalies_detected=int(anomalies_detected),
        anomaly_rate=float(anomaly_rate),
        threshold_used=float(threshold),
        detection_method=detection_method
    )


@app.post("/inference/file")
async def run_file_inference(
    file_path: str,
    task_type: str,
    num_steps: int = 1,
    output_path: Optional[str] = None
):
    """Run inference on data from file."""
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Input file not found: {file_path}")
        
        # Run file inference
        start_time = time.time()
        results = inference_pipeline.predict_from_file(
            file_path, num_steps=num_steps, output_path=output_path
        )
        inference_time = time.time() - start_time
        
        return {
            "status": "success",
            "task_type": task_type,
            "timestamp": datetime.now().isoformat(),
            "inference_time": inference_time,
            "results": results,
            "output_path": output_path
        }
        
    except Exception as e:
        logger.error(f"File inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"File inference failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    global inference_pipeline, current_config
    
    if inference_pipeline is None or current_config is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    return {
        "task_type": current_config.task_name,
        "model": current_config.model,
        "sequence_length": current_config.seq_len,
        "input_features": current_config.enc_in,
        "output_features": current_config.c_out,
        "is_initialized": inference_pipeline.is_initialized,
        "buffer_size": len(inference_pipeline.data_buffer) if inference_pipeline.data_buffer else 0
    }


@app.delete("/model")
async def unload_model():
    """Unload the current model."""
    global inference_pipeline, current_config
    
    if inference_pipeline is None:
        return {"status": "success", "message": "No model to unload"}
    
    try:
        inference_pipeline.close()
        inference_pipeline = None
        current_config = None
        
        return {"status": "success", "message": "Model unloaded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server."""
    logger.info(f"Starting TS-Unity Inference API server on {host}:{port}")
    
    uvicorn.run(
        "inference_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TS-Unity Inference API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    start_server(host=args.host, port=args.port, reload=args.reload)
