from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ..., 
        description="Normalized vector of structural and spatial features x ∈ ℝⁿ (e.g., sqft, bedrooms, distance to transit)"
    )
    model_type: str = Field(
        "ridge", 
        description="Optimization model choice: 'ridge' (L2 continuous shrinkage) or 'lasso' (L1 feature selection sparsity)"
    )
    lambda_reg: float = Field(
        0.1, 
        ge=0.0, 
        description="Regularization hyperparameter λ controlling the penalty strength"
    )

class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Estimated real estate price ŷ computed via θᵀx")
    model_used: str = Field(..., description="Name of the custom optimization model executed")
    weights_applied: List[float] = Field(..., description="Learned coefficient vector θ from the optimization core")
    status: str = "success"

class ReverseSearchRequest(BaseModel):
    target_budget: float = Field(
        ..., 
        gt=0.0, 
        description="User's desired target budget y_target set via the map/budget interface"
    )
    base_features: List[float] = Field(
        ..., 
        description="Baseline feature configuration x_base for the spatial zone"
    )
    lambda_reg: float = Field(0.1, ge=0.0, description="Regularization hyperparameter λ used during training")

class ReverseSearchResponse(BaseModel):
    target_budget: float = Field(..., description="The user's target budget constraint")
    optimal_feature_configuration: List[float] = Field(
        ..., 
        description="Optimized feature trade-offs solving the Lagrangian projection problem"
    )
    achieved_price: float = Field(
        ..., 
        description="Verification check: predicted price θᵀx matching the target budget within tolerance"
    )
    status: str = "success"