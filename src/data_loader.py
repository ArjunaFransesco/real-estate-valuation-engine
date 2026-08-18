"""
Data Loading and Synthetic Real Estate Dataset Generation Module
Generates high-fidelity metropolitan housing transaction data with spatial attributes.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass



CBD_COORDINATES = (-6.2088, 106.8456)  # Metropolitan Central Business District


def generate_synthetic_housing_data(
    n_samples: int = 5000,
    random_state: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generates a realistic hedonic pricing real estate dataset with spatial,
    structural, and neighborhood attributes.
    """
    np.random.seed(random_state)
    
    # 1. Spatial Attributes (Metropolitan coordinates with Gaussian cluster around CBD)
    cbd_lat, cbd_lon = CBD_COORDINATES
    
    # Cluster centers (Sub-markets: Downtown, Tech Hub, Suburban North, Luxury Waterfront)
    cluster_centers = [
        (cbd_lat, cbd_lon, 0.03, "Downtown / CBD"),
        (cbd_lat + 0.08, cbd_lon - 0.05, 0.04, "Tech Corridor"),
        (cbd_lat - 0.12, cbd_lon + 0.06, 0.05, "South Hills"),
        (cbd_lat + 0.14, cbd_lon + 0.07, 0.06, "North Suburbs"),
        (cbd_lat - 0.05, cbd_lon - 0.12, 0.04, "West Waterfront")
    ]
    
    samples_per_cluster = n_samples // len(cluster_centers)
    records = []
    
    for c_lat, c_lon, spread, zone_name in cluster_centers:
        n_c = samples_per_cluster
        lats = np.random.normal(c_lat, spread, n_c)
        lons = np.random.normal(c_lon, spread, n_c)
        
        for i in range(n_c):
            lat, lon = lats[i], lons[i]
            
            # Distance calculations (approx Euclidean in km)
            dist_to_cbd = np.sqrt((lat - cbd_lat)**2 + (lon - cbd_lon)**2) * 111.0
            dist_to_transit = np.clip(np.random.exponential(scale=1.5 + dist_to_cbd * 0.1), 0.1, 15.0)
            
            # Property structural features
            property_type = np.random.choice(
                ["Single Family", "Townhouse", "Condominium", "Luxury Villa"],
                p=[0.45, 0.25, 0.20, 0.10]
            )
            
            if property_type == "Condominium":
                living_area = int(np.random.normal(950, 250))
                lot_area = living_area
                bedrooms = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
                bathrooms = np.random.choice([1.0, 1.5, 2.0], p=[0.4, 0.3, 0.3])
                stories = 1
                garage_capacity = np.random.choice([0, 1, 2], p=[0.3, 0.6, 0.1])
                has_pool = 1 if np.random.rand() < 0.6 else 0
            elif property_type == "Townhouse":
                living_area = int(np.random.normal(1600, 350))
                lot_area = int(living_area * np.random.uniform(1.1, 1.6))
                bedrooms = np.random.choice([2, 3, 4], p=[0.2, 0.6, 0.2])
                bathrooms = np.random.choice([1.5, 2.0, 2.5], p=[0.2, 0.5, 0.3])
                stories = np.random.choice([2, 3], p=[0.7, 0.3])
                garage_capacity = np.random.choice([1, 2], p=[0.6, 0.4])
                has_pool = 1 if np.random.rand() < 0.25 else 0
            elif property_type == "Luxury Villa":
                living_area = int(np.random.normal(4200, 900))
                lot_area = int(living_area * np.random.uniform(2.5, 6.0))
                bedrooms = np.random.choice([4, 5, 6, 7], p=[0.3, 0.4, 0.2, 0.1])
                bathrooms = np.random.choice([3.5, 4.0, 5.0, 6.0], p=[0.3, 0.4, 0.2, 0.1])
                stories = np.random.choice([2, 3], p=[0.8, 0.2])
                garage_capacity = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
                has_pool = 1 if np.random.rand() < 0.9 else 0
            else:  # Single Family
                living_area = int(np.random.normal(2200, 550))
                lot_area = int(living_area * np.random.uniform(1.8, 3.5))
                bedrooms = np.random.choice([2, 3, 4, 5], p=[0.1, 0.45, 0.35, 0.1])
                bathrooms = np.random.choice([1.5, 2.0, 2.5, 3.0], p=[0.15, 0.4, 0.3, 0.15])
                stories = np.random.choice([1, 2], p=[0.4, 0.6])
                garage_capacity = np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2])
                has_pool = 1 if np.random.rand() < 0.3 else 0
                
            living_area = max(450, living_area)
            lot_area = max(living_area, lot_area)
            
            year_built = int(np.random.randint(1975, 2024))
            is_renovated = 1 if (2024 - year_built > 15 and np.random.rand() < 0.35) else 0
            year_renovated = int(np.random.randint(year_built + 5, 2024)) if is_renovated else year_built
            
            # Neighborhood & quality indexes
            school_rating = np.clip(int(np.random.normal(7.0 - 0.15 * dist_to_cbd + (0.8 if zone_name == "North Suburbs" else 0), 1.6)), 1, 10)
            crime_rate_index = np.clip(np.random.normal(3.5 + 0.12 * dist_to_cbd, 1.2), 0.5, 10.0)
            quality_grade = np.clip(int(np.random.normal(6.5 + (2.0 if property_type == "Luxury Villa" else 0.0), 1.4)), 1, 10)
            
            # Hedonic Pricing Base Function
            base_sqft_rate = 190.0  # Base $ / sqft
            location_factor = np.exp(-0.04 * dist_to_cbd) * 1.5 + (1.2 if zone_name == "West Waterfront" else 0.8)
            transit_penalty = np.exp(-0.08 * dist_to_transit)
            school_premium = 1.0 + (school_rating - 5) * 0.04
            quality_premium = 1.0 + (quality_grade - 5) * 0.08
            age_depreciation = max(0.65, 1.0 - (2024 - year_renovated) * 0.007)
            amenity_bonus = 1.0 + (0.07 if has_pool else 0.0) + garage_capacity * 0.03
            
            estimated_price = (
                (living_area * base_sqft_rate * location_factor)
                + (lot_area * 25.0)
                + (bedrooms * 12000.0)
                + (bathrooms * 16000.0)
            ) * school_premium * quality_premium * age_depreciation * amenity_bonus * (1.0 + 0.15 * transit_penalty)
            
            # Add market stochastic noise (log-normal)
            noise = np.random.normal(1.0, 0.06)
            price = max(85000.0, round(estimated_price * noise, -2))
            
            records.append({
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "zone_name": zone_name,
                "distance_to_cbd_km": round(dist_to_cbd, 2),
                "distance_to_transit_km": round(dist_to_transit, 2),
                "property_type": property_type,
                "living_area_sqft": living_area,
                "lot_area_sqft": lot_area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "stories": stories,
                "garage_capacity": garage_capacity,
                "has_pool": has_pool,
                "year_built": year_built,
                "year_renovated": year_renovated,
                "is_renovated": is_renovated,
                "school_rating": school_rating,
                "crime_rate_index": round(crime_rate_index, 2),
                "quality_grade": quality_grade,
                "price": price
            })
            
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[+] Generated and saved {len(df)} real estate records to {output_path}")
        
    return df


def load_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """Loads dataset from path or synthesizes if not present."""
    if data_path and os.path.exists(data_path):
        return pd.read_csv(data_path)
    
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "properties_dataset.csv"
    )
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    
    return generate_synthetic_housing_data(output_path=default_path)


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "price"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separates feature matrix X and target vector y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
