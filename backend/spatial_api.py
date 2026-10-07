"""Spatial + requirement-filter endpoints for the VestaEst dashboard (India)."""
import json
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.inverse_solver import inverse_optimization_solver
from src.preprocessing import SpatialPreprocessor

router = APIRouter()
FEATURES = ["area_sqft", "distance_to_hub_km", "bhk", "age_years", "amenity_score"]
AMENITIES = ["parking", "lift", "gated_community", "swimming_pool", "gas_pipeline"]
PRICE = "price_inr"
_cache = {}


def _load():
    if not _cache:
        try:
            df = pd.read_csv("data/housing_spatial.csv")
            with open("data/trained_weights.json") as f:
                w = json.load(f)
        except FileNotFoundError:
            raise HTTPException(503, "Run `python train.py` first (data files missing).")
        if PRICE not in df.columns:
            raise HTTPException(500, "Old dataset found. Re-run `python train.py` to build the India dataset.")
        pre = SpatialPreprocessor()
        pre.means, pre.stds = np.array(w["means"]), np.array(w["stds"])
        design = np.hstack([np.ones((len(df), 1)), pre.transform(df[FEATURES].values.astype(float))])
        _cache.update(df=df, w=w, pre=pre, design=design)
    return _cache["df"], _cache["w"], _cache["pre"], _cache["design"]


class RegionQuery(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    model_type: str = "ridge"
    target_budget: Optional[float] = None          # Rs
    property_types: List[str] = Field(default_factory=list)
    bhk: List[int] = Field(default_factory=list)   # 5 means "4+ BHK"
    possession: List[str] = Field(default_factory=list)
    amenities: List[str] = Field(default_factory=list)
    min_area: Optional[float] = None               # sq ft
    max_area: Optional[float] = None


@router.get("/map-data", tags=["Spatial Query"])
def map_data():
    df, *_ = _load()
    pts = [{"lat": a, "lon": b, "price": c} for a, b, c in zip(df.latitude, df.longitude, df[PRICE])]
    return {"points": pts, "bounds": [[df.latitude.min(), df.longitude.min()], [df.latitude.max(), df.longitude.max()]]}


@router.post("/region-query", tags=["Spatial Query"])
def region_query(q: RegionQuery):
    df, w, pre, design = _load()
    theta = np.array(w["lasso_theta"] if q.model_type.lower() == "lasso" else w["ridge_theta"])

    mask = df.latitude.between(q.min_lat, q.max_lat) & df.longitude.between(q.min_lon, q.max_lon)
    if q.property_types:
        mask &= df.property_type.isin(q.property_types)
    if q.bhk:
        mask &= df.bhk.isin(q.bhk) | ((df.bhk >= 5) if 5 in q.bhk else False)
    if q.possession:
        mask &= df.possession.isin(q.possession)
    for a in q.amenities:
        if a in AMENITIES:
            mask &= df[a] == 1
    if q.min_area is not None:
        mask &= df.area_sqft >= q.min_area
    if q.max_area is not None:
        mask &= df.area_sqft <= q.max_area

    sub = df[mask]
    if sub.empty:
        return {"count": 0}
    preds = np.maximum(0.0, design[mask.values] @ theta)
    avg = sub[FEATURES].mean()
    out = {
        "count": int(len(sub)),
        "stats": {"avg_price": float(sub[PRICE].mean()), "min_price": float(sub[PRICE].min()),
                  "max_price": float(sub[PRICE].max()), "avg_predicted": float(preds.mean()),
                  "avg_ppsf": float(sub[PRICE].sum() / sub.area_sqft.sum()), "avg_area": float(sub.area_sqft.mean())},
        "avg_features": {k: float(avg[k]) for k in FEATURES},
        "points": [{"lat": r.latitude, "lon": r.longitude, "price": r.price_inr, "pred": float(p), "locality": r.locality,
                    "bhk": int(r.bhk), "area": int(r.area_sqft), "type": r.property_type, "possession": r.possession}
                   for r, p in zip(sub.itertuples(), preds)],
        "reverse": None,
    }
    if q.target_budget:
        x0 = np.hstack([1.0, pre.transform(avg.values.reshape(1, -1))[0]])
        x_opt = inverse_optimization_solver(theta, x0, q.target_budget)
        needed = x_opt[1:] * pre.stds + pre.means
        # keep results physically realistic
        needed[0] = max(300.0, needed[0])            # area sq ft
        needed[1] = max(0.5, needed[1])              # km to hub
        needed[2] = min(6.0, max(1.0, needed[2]))    # BHK
        needed[3] = max(0.0, needed[3])              # age
        needed[4] = min(5.0, max(0.0, needed[4]))    # amenity score
        x_cl = np.hstack([1.0, pre.transform(needed.reshape(1, -1))[0]])
        out["reverse"] = {
            "target_budget": q.target_budget,
            "achieved_price": float(theta @ x_cl),
            "config": {k: float(v) for k, v in zip(FEATURES, needed)},
            "affordable_count": int((sub[PRICE] <= q.target_budget).sum()),
        }
    return out
