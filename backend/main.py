from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import os

from backend.spatial_api import router as spatial_router

app = FastAPI(
    title="Geo-Optimized Real Estate Valuation & Reverse-Prediction Engine",
    description="FOA PBL Project Backend - Custom NumPy Optimization & Regularization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the spatial region-query router
app.include_router(spatial_router)

from backend.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    ReverseSearchRequest, 
    ReverseSearchResponse
)
from src.inverse_solver import inverse_optimization_solver
from src.preprocessing import SpatialPreprocessor

# Load trained weights and preprocessor parameters on startup
WEIGHTS_PATH = "data/trained_weights.json"
preprocessor = SpatialPreprocessor()

if os.path.exists(WEIGHTS_PATH):
    with open(WEIGHTS_PATH, "r") as f:
        weights_data = json.load(f)
        GLOBAL_THETA_RIDGE = np.array(weights_data["ridge_theta"])
        GLOBAL_THETA_LASSO = np.array(weights_data["lasso_theta"])
        preprocessor.means = np.array(weights_data["means"])
        preprocessor.stds = np.array(weights_data["stds"])
else:
    # Fallback weights if training script hasn't been run yet
    GLOBAL_THETA_RIDGE = np.zeros(6)
    GLOBAL_THETA_LASSO = np.zeros(6)

@app.get("/", tags=["Status"])
def read_root():
    return {
        "project": "House Price Prediction Using Gradient Descent and Regularization",
        "course": "Foundations of AI and Optimization Theory (FOA)",
        "status": "Online with Dynamic Trained Weights & Inverse Clamping"
    }

@app.get("/meta", tags=["Metadata"])
def meta():
    """
    Exposes Z-score standardization means and standard deviations 
    so the frontend can perform inverse transformations into natural units.
    """
    if preprocessor.means is None or preprocessor.stds is None:
        raise HTTPException(status_code=400, detail="Preprocessor parameters not loaded. Run training script first.")
    return {
        "means": preprocessor.means.tolist(),
        "stds": preprocessor.stds.tolist()
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Forward Prediction"])
def predict_house_price(payload: PredictionRequest):
    try:
        x_raw = np.array(payload.features).reshape(1, -1)
        
        # Apply Z-score standardization using trained preprocessor metrics
        x_scaled = preprocessor.transform(x_raw)
        
        # Add bias term column of ones
        x_design = np.hstack([np.ones((x_scaled.shape[0], 1)), x_scaled])[0]
        
        if payload.model_type.lower() == "lasso":
            theta = GLOBAL_THETA_LASSO
            model_name = "Custom Lasso Regression (L1)"
        else:
            theta = GLOBAL_THETA_RIDGE
            model_name = "Custom Ridge Regression (L2)"
            
        predicted_price = float(np.dot(x_design, theta))
        
        return PredictionResponse(
            predicted_price=max(0.0, predicted_price),
            model_used=model_name,
            weights_applied=theta.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reverse-search", response_model=ReverseSearchResponse, tags=["Reverse Prediction"])
def reverse_budget_search(payload: ReverseSearchRequest):
    try:
        theta = GLOBAL_THETA_RIDGE
        x_base_raw = np.array(payload.base_features)
        
        # Scale base features for optimization space alignment
        x_base_scaled = preprocessor.transform(x_base_raw.reshape(1, -1))[0]
        x_base_design = np.hstack([1.0, x_base_scaled])
        
        # Solve inverse optimization projection in scaled space
        optimal_features_design = inverse_optimization_solver(theta, x_base_design, payload.target_budget)
        optimal_features_scaled = optimal_features_design[1:]
        
        # Inverse transform back to physical (natural) units
        optimal_features_natural = preprocessor.inverse_transform_features(optimal_features_scaled.reshape(1, -1))[0]
        
        # Apply physical boundary constraints (keeps outputs realistic)
        # Features: [area_sqft, distance_to_hub_km, bhk, age_years, amenity_score]
        optimal_features_natural[0] = max(300.0, optimal_features_natural[0])
        optimal_features_natural[1] = max(0.5, optimal_features_natural[1])
        optimal_features_natural[2] = max(1.0, min(6.0, optimal_features_natural[2]))
        optimal_features_natural[3] = max(0.0, optimal_features_natural[3])
        optimal_features_natural[4] = max(0.0, min(5.0, optimal_features_natural[4]))
        
        achieved_price = float(np.dot(theta, optimal_features_design))
        
        return ReverseSearchResponse(
            target_budget=payload.target_budget,
            optimal_feature_configuration=optimal_features_natural.tolist(),
            achieved_price=achieved_price
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))