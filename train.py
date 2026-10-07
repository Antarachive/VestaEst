"""Builds an India (Hyderabad) listings dataset and trains the Ridge / Lasso models.

NOTE: the listings are SYNTHETIC. Locality centres are approximate real locations and the
Rs/sq ft levels are rough market ranges, but no row is a real listing. To use real data,
replace data/housing_spatial.csv with a file that has the same columns and re-run the
training part (see the columns written in build_dataset below).
"""
import json
import os

import numpy as np
import pandas as pd

from src.optimizers import CustomLassoRegression, CustomRidgeRegression
from src.preprocessing import SpatialPreprocessor

FEATURES = ["area_sqft", "distance_to_hub_km", "bhk", "age_years", "amenity_score"]
HUB = (17.4435, 78.3772)  # HITEC City, used as the distance reference
AMENITIES = ["parking", "lift", "gated_community", "swimming_pool", "gas_pipeline"]

# locality: (latitude, longitude, typical Rs per sq ft, relative number of listings)
LOCALITIES = {
    "Gachibowli": (17.4401, 78.3489, 9200, 10), "Madhapur": (17.4483, 78.3915, 10500, 7),
    "Kondapur": (17.4600, 78.3640, 8500, 9), "HITEC City": (17.4435, 78.3772, 9800, 6),
    "Financial District": (17.4170, 78.3430, 9000, 6), "Kokapet": (17.3950, 78.3290, 9500, 6),
    "Narsingi": (17.3920, 78.3570, 7600, 5), "Manikonda": (17.4010, 78.3830, 6800, 6),
    "Tellapur": (17.4600, 78.2850, 6200, 4), "Miyapur": (17.4967, 78.3590, 6300, 7),
    "Kukatpally": (17.4948, 78.3996, 6500, 7), "Nizampet": (17.5160, 78.3830, 6000, 5),
    "Bachupally": (17.5440, 78.3680, 5700, 6), "Kompally": (17.5360, 78.4850, 5300, 6),
    "Jubilee Hills": (17.4326, 78.4071, 17500, 4), "Banjara Hills": (17.4156, 78.4347, 16500, 4),
    "Begumpet": (17.4440, 78.4670, 9800, 3), "Ameerpet": (17.4375, 78.4483, 8000, 3),
    "Secunderabad": (17.4399, 78.4983, 8000, 4), "Uppal": (17.4058, 78.5591, 5200, 6),
    "LB Nagar": (17.3457, 78.5522, 5000, 5), "Dilsukhnagar": (17.3688, 78.5247, 6000, 3),
    "Kothapet": (17.3680, 78.5460, 5400, 3), "Attapur": (17.3650, 78.4290, 5300, 4),
    "Mehdipatnam": (17.3950, 78.4350, 6800, 3), "Tolichowki": (17.4000, 78.4170, 6500, 3),
    "Shamshabad": (17.2610, 78.3980, 4300, 4), "Hayathnagar": (17.3280, 78.6000, 3800, 3),
    "Medchal": (17.6290, 78.4810, 3500, 3),
}
AREA_MEAN = {1: 620, 2: 1100, 3: 1650, 4: 2500, 5: 3600}
TYPE_ADJ = {"Villa": 0.12, "Independent House": 0.0, "Builder Floor": -0.05, "Apartment": 0.0}
POSSESSION = ["Ready to Move", "In 1 Year", "In 2 Years", "In 3 Years", "After 3 Years"]


def haversine_km(lat, lon, lat0, lon0):
    p = np.pi / 180
    a = np.sin((lat - lat0) * p / 2) ** 2 + np.cos(lat * p) * np.cos(lat0 * p) * np.sin((lon - lon0) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def build_dataset(m=1500, seed=42):
    rng = np.random.default_rng(seed)
    names = list(LOCALITIES)
    w = np.array([LOCALITIES[n][3] for n in names], float)
    rows = []
    for loc in rng.choice(names, size=m, p=w / w.sum()):
        lat0, lon0, ppsf0, _ = LOCALITIES[loc]
        ptype = rng.choice(["Apartment", "Independent House", "Villa", "Builder Floor"], p=[.5, .22, .13, .15])
        if ptype in ("Independent House", "Villa"):
            bhk = rng.choice([2, 3, 4, 5], p=[.1, .4, .35, .15])
        elif ptype == "Builder Floor":
            bhk = rng.choice([1, 2, 3, 4], p=[.1, .45, .38, .07])
        else:
            bhk = rng.choice([1, 2, 3, 4], p=[.1, .42, .38, .10])
        area = AREA_MEAN[bhk] * (1.15 if ptype == "Villa" else 1.0) * rng.normal(1, 0.12)
        poss = rng.choice(POSSESSION, p=[.55, .15, .12, .10, .08])
        age = 0 if poss != "Ready to Move" else rng.uniform(1, 20)
        house = ptype in ("Independent House", "Villa")
        p = {"parking": .95 if house else .9, "lift": .1 if house else .75,
             "gated_community": .8 if ptype == "Villa" else (.35 if house else .6),
             "swimming_pool": .35 if ptype == "Villa" else (.1 if house else .3), "gas_pipeline": .4 if house else .45}
        flags = {a: int(rng.random() < p[a]) for a in AMENITIES}
        score = sum(flags.values())
        lat, lon = lat0 + rng.normal(0, 0.009), lon0 + rng.normal(0, 0.009)
        ppsf = (ppsf0 * (1 + TYPE_ADJ[ptype]) * (1 - 0.01 * min(age, 20)) * (1 + 0.015 * score)
                * (0.96 if poss != "Ready to Move" else 1.0) * rng.lognormal(0, 0.07))
        price = round(area * ppsf, -4)
        rows.append({"locality": loc, "property_type": ptype, "bhk": int(bhk), "bathrooms": int(min(bhk + rng.integers(0, 2), 6)),
                     "area_sqft": round(area), "age_years": round(age, 1), "possession": poss, **flags,
                     "amenity_score": score, "distance_to_hub_km": round(float(haversine_km(lat, lon, *HUB)), 2),
                     "price_per_sqft": round(price / area), "price_inr": price,
                     "latitude": round(lat, 5), "longitude": round(lon, 5)})
    return pd.DataFrame(rows)


def process_and_train_models(data_path="data/housing_spatial.csv", weights_path="data/trained_weights.json"):
    os.makedirs("data", exist_ok=True)
    df = build_dataset()
    df.to_csv(data_path, index=False)
    print(f"Saved {len(df)} listings to {data_path}")

    X, y = df[FEATURES].values.astype(float), df["price_inr"].values.astype(float)
    pre = SpatialPreprocessor()
    Xd = np.hstack([np.ones((len(X), 1)), pre.fit_transform(X)])

    ridge = CustomRidgeRegression(lambda_reg=0.1, lr=0.05, max_iters=1500); ridge.fit(Xd, y)
    lasso = CustomLassoRegression(lambda_reg=0.1, lr=0.05, max_iters=1500); lasso.fit(Xd, y)
    for name, mdl in (("Ridge", ridge), ("Lasso", lasso)):
        r2 = 1 - np.sum((y - mdl.predict(Xd)) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"{name} R^2 on training data: {r2:.3f}")

    with open(weights_path, "w") as f:
        json.dump({"ridge_theta": ridge.theta.tolist(), "lasso_theta": lasso.theta.tolist(),
                   "means": pre.means.tolist(), "stds": pre.stds.tolist(), "features": FEATURES}, f, indent=4)
    print(f"Saved weights to {weights_path}")


if __name__ == "__main__":
    process_and_train_models()
