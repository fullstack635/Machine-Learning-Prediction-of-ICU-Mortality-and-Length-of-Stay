from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List
import joblib
import os
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Model Tester API",
    description="API for testing machine learning models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: Any
    confidence: float | None = None
    model_info: Dict[str, Any] | None = None

# Global model variables
model = None
los_model = None

def load_model():
    """Load the primary model from the .pkl file"""
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        logger.info(f"Looking for model at: {model_path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in current directory: {os.listdir(os.path.dirname(__file__))}")
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Primary model loaded successfully")
            
            # Log model information
            logger.info(f"Model type: {type(model)}")
            if hasattr(model, 'feature_names_in_'):
                logger.info(f"Expected feature names: {model.feature_names_in_}")
            if hasattr(model, 'n_features_in_'):
                logger.info(f"Expected number of features: {model.n_features_in_}")
            
            return True
        else:
            logger.error(f"Model file not found at {model_path}")
            logger.error("Please upload model.pkl to the backend directory on Railway")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def load_los_model():
    """Load the LOS model from the .pkl file"""
    global los_model
    try:
        model_path = os.path.join(os.path.dirname(__file__), "los_lgbm_pipeline.pkl")
        logger.info(f"Looking for LOS model at: {model_path}")
        
        if os.path.exists(model_path):
            los_model = joblib.load(model_path)
            logger.info("LOS model loaded successfully")
            
            # Log model information
            logger.info(f"LOS Model type: {type(los_model)}")
            if hasattr(los_model, 'feature_names_in_'):
                logger.info(f"LOS Expected feature names: {los_model.feature_names_in_}")
            if hasattr(los_model, 'n_features_in_'):
                logger.info(f"LOS Expected number of features: {los_model.n_features_in_}")
            
            return True
        else:
            logger.error(f"LOS Model file not found at {model_path}")
            logger.error("Please upload los_lgbm_pipeline.pkl to the backend directory on Railway")
            return False
    except Exception as e:
        logger.error(f"Error loading LOS model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Starting backend application...")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in directory: {os.listdir('.')}")
    
    try:
        if not load_model():
            logger.warning("Primary model could not be loaded. API will return errors for predictions.")
        else:
            logger.info("✅ Primary model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load primary model: {e}")
    
    try:
        if not load_los_model():
            logger.warning("LOS model could not be loaded. API will return errors for LOS predictions.")
        else:
            logger.info("✅ LOS model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load LOS model: {e}")
    
    logger.info("🎉 Backend startup completed")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "message": "Model Tester API is running",
        "primary_model_loaded": model is not None,
        "los_model_loaded": los_model is not None,
        "timestamp": "2025-10-26T04:32:08Z"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "primary_model_loaded": model is not None,
        "los_model_loaded": los_model is not None,
        "primary_model_type": type(model).__name__ if model else None,
        "los_model_type": type(los_model).__name__ if los_model else None
    }

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint that doesn't require models"""
    return {
        "status": "ok",
        "message": "Backend is running and accessible",
        "timestamp": "2025-10-26T04:32:08Z"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction using the loaded model"""
    if model is None:
        logger.error("Primary model is not loaded")
        raise HTTPException(
            status_code=503, 
            detail="Primary model not loaded. Please check server logs and ensure model.pkl is uploaded."
        )
    
    try:
        # Convert features to the format expected by the model
        features = request.features
        
        # Handle different model types
        if hasattr(model, 'predict'):
            # Convert features to pandas DataFrame
            if isinstance(features, dict):
                # Create DataFrame with proper column names
                df = pd.DataFrame([features])
                logger.info(f"Created DataFrame with columns: {list(df.columns)}")
                logger.info(f"DataFrame shape: {df.shape}")
            else:
                df = pd.DataFrame([features])
            
            prediction = model.predict(df)[0]
            
            # Try to get prediction probabilities if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(df)[0]
                    confidence = float(max(proba))
                except:
                    pass
            
            response_data = {
                "prediction": float(prediction),
                "model_info": {
                    "type": type(model).__name__,
                    "feature_count": len(df.columns)
                }
            }
            if confidence is not None:
                response_data["confidence"] = confidence
            return PredictionResponse(**response_data)
        else:
            # Custom model with callable
            prediction = model(features)
            return PredictionResponse(
                prediction=float(prediction),
                model_info={"type": type(model).__name__}
            )
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-los", response_model=PredictionResponse)
async def predict_los(request: PredictionRequest):
    """Make a prediction using the LOS model"""
    if los_model is None:
        logger.error("LOS model is not loaded")
        raise HTTPException(
            status_code=503, 
            detail="LOS Model not loaded. Please check server logs and ensure los_lgbm_pipeline.pkl is uploaded."
        )
    
    try:
        # Convert features to the format expected by the model
        features = request.features
        
        # Handle different model types
        if hasattr(los_model, 'predict'):
            # Convert features to pandas DataFrame
            if isinstance(features, dict):
                # Create DataFrame with proper column names
                df = pd.DataFrame([features])
                logger.info(f"Created DataFrame with columns: {list(df.columns)}")
                logger.info(f"DataFrame shape: {df.shape}")
            else:
                df = pd.DataFrame([features])
            
            prediction = los_model.predict(df)[0]
            
            # Try to get prediction probabilities if available
            confidence = None
            if hasattr(los_model, 'predict_proba'):
                try:
                    proba = los_model.predict_proba(df)[0]
                    confidence = float(max(proba))
                except:
                    pass
            
            response_data = {
                "prediction": float(prediction),
                "model_info": {
                    "type": type(los_model).__name__,
                    "feature_count": len(df.columns)
                }
            }
            if confidence is not None:
                response_data["confidence"] = confidence
            return PredictionResponse(**response_data)
        else:
            # Custom model with callable
            prediction = los_model(features)
            return PredictionResponse(
                prediction=float(prediction),
                model_info={"type": type(los_model).__name__}
            )
            
    except Exception as e:
        logger.error(f"LOS Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LOS Prediction failed: {str(e)}")

# Note: Frontend is deployed separately on Vercel

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
