"""
Feature Engineering and Spatial Transformation Pipeline
Extracts domain-specific spatial clusters, structural ratios, and interaction features.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


class SpatialAndStructuralTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to compute spatial clusters, distance interactions,
    and structural real estate ratios.
    """
    def __init__(self, n_spatial_clusters: int = 8, random_state: int = 42):
        self.n_spatial_clusters = n_spatial_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_spatial_clusters, random_state=random_state, n_init=10)
        self.cluster_centers_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        coords = X[["latitude", "longitude"]].values
        self.kmeans.fit(coords)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        current_year = 2024
        
        # 1. Spatial cluster assignment
        coords = df[["latitude", "longitude"]].values
        df["spatial_cluster"] = self.kmeans.predict(coords).astype(str)
        
        # 2. Structural Age Features
        df["property_age"] = current_year - df["year_built"]
        df["effective_age"] = current_year - df["year_renovated"]
        df["years_since_reno"] = np.where(
            df["is_renovated"] == 1,
            current_year - df["year_renovated"],
            df["property_age"]
        )
        
        # 3. Structural Ratios & Room Densities
        df["living_to_lot_ratio"] = df["living_area_sqft"] / (df["lot_area_sqft"] + 1e-5)
        df["bath_to_bed_ratio"] = df["bathrooms"] / (df["bedrooms"] + 1e-5)
        df["sqft_per_room"] = df["living_area_sqft"] / (df["bedrooms"] + df["bathrooms"] + 1.0)
        df["sqft_per_bedroom"] = df["living_area_sqft"] / (df["bedrooms"] + 1e-5)
        
        # 4. Amenities & Quality Index Interactions
        df["amenity_index"] = (
            df["has_pool"] * 2.5
            + df["garage_capacity"] * 1.2
            + df["stories"] * 0.8
        )
        df["quality_school_interaction"] = df["quality_grade"] * df["school_rating"]
        df["safety_school_score"] = df["school_rating"] / (df["crime_rate_index"] + 0.1)
        
        # 5. Spatial Non-linear Proximity
        df["log_dist_to_cbd"] = np.log1p(df["distance_to_cbd_km"])
        df["log_dist_to_transit"] = np.log1p(df["distance_to_transit_km"])
        df["cbd_transit_interaction"] = df["distance_to_cbd_km"] * df["distance_to_transit_km"]
        
        return df


def build_preprocessor(n_clusters: int = 8) -> Pipeline:
    """
    Constructs the complete Scikit-Learn preprocessing pipeline.
    """
    categorical_cols = ["zone_name", "property_type", "spatial_cluster"]
    
    numeric_cols = [
        "latitude", "longitude", "distance_to_cbd_km", "distance_to_transit_km",
        "living_area_sqft", "lot_area_sqft", "bedrooms", "bathrooms", "stories",
        "garage_capacity", "has_pool", "year_built", "year_renovated", "is_renovated",
        "school_rating", "crime_rate_index", "quality_grade",
        "property_age", "effective_age", "years_since_reno", "living_to_lot_ratio",
        "bath_to_bed_ratio", "sqft_per_room", "sqft_per_bedroom", "amenity_index",
        "quality_school_interaction", "safety_school_score", "log_dist_to_cbd",
        "log_dist_to_transit", "cbd_transit_interaction"
    ]
    
    col_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="drop"
    )
    
    feature_pipeline = Pipeline([
        ("custom_spatial", SpatialAndStructuralTransformer(n_spatial_clusters=n_clusters)),
        ("column_transforms", col_transformer)
    ])
    
    return feature_pipeline
