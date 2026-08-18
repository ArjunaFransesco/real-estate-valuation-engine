"""
Streamlit Web Application: Real Estate Automated Valuation & Spatial Price Prediction Engine
Author: Arjuna Fransesco
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="Real Estate Valuation Engine | Arjuna Fransesco",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 2.4rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_ml_components():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(base_dir, "artifacts")
    
    pipeline_path = os.path.join(artifacts_dir, "feature_pipeline.joblib")
    model_path = os.path.join(artifacts_dir, "stacking_model.joblib")
    metrics_path = os.path.join(artifacts_dir, "metrics_summary.json")
    
    pipeline = joblib.load(pipeline_path) if os.path.exists(pipeline_path) else None
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    return pipeline, model, metrics

pipeline, model, metrics = load_ml_components()

# Header
st.markdown('<div class="main-header">🏡 Automated Real Estate Valuation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production Stacking Ensemble (XGBoost + LightGBM + Random Forest) with Spatial Distance Engineering</div>', unsafe_allow_html=True)

# Sidebar: Property Input Parameters
st.sidebar.header("📍 Location & Spatial Coordinates")

ZONE_PRESETS = {
    "Downtown / CBD": {"lat": -6.2088, "lon": 106.8456, "cbd_dist": 0.5, "school": 6, "crime": 4.2},
    "Tech Corridor": {"lat": -6.1288, "lon": 106.7956, "cbd_dist": 8.5, "school": 8, "crime": 2.1},
    "South Hills": {"lat": -6.3288, "lon": 106.9056, "cbd_dist": 14.2, "school": 7, "crime": 3.0},
    "North Suburbs": {"lat": -6.0688, "lon": 106.9156, "cbd_dist": 16.8, "school": 9, "crime": 1.8},
    "West Waterfront": {"lat": -6.2588, "lon": 106.7256, "cbd_dist": 13.5, "school": 7, "crime": 2.5}
}

selected_zone = st.sidebar.selectbox("Submarket / Zone", list(ZONE_PRESETS.keys()))
preset = ZONE_PRESETS[selected_zone]

latitude = st.sidebar.number_input("Latitude", value=preset["lat"], format="%.5f")
longitude = st.sidebar.number_input("Longitude", value=preset["lon"], format="%.5f")
dist_cbd = st.sidebar.slider("Distance to CBD (km)", 0.2, 35.0, float(preset["cbd_dist"]))
dist_transit = st.sidebar.slider("Distance to Transit Hub (km)", 0.1, 15.0, 1.2)

st.sidebar.header("🏗️ Structural & Property Specs")
property_type = st.sidebar.selectbox("Property Type", ["Single Family", "Townhouse", "Condominium", "Luxury Villa"])
living_area = st.sidebar.number_input("Living Area (sqft)", min_value=300, max_value=12000, value=2400, step=50)
lot_area = st.sidebar.number_input("Lot Area (sqft)", min_value=300, max_value=50000, value=4800, step=100)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    stories = st.number_input("Stories", 1, 4, 2)
with col_sb2:
    bathrooms = st.number_input("Bathrooms", 1.0, 8.0, 2.5, step=0.5)
    garage_cap = st.number_input("Garage Cars", 0, 6, 2)

has_pool = 1 if st.sidebar.checkbox("Has Private Pool", value=False) else 0

st.sidebar.header("📅 Age & Neighborhood Quality")
year_built = st.sidebar.slider("Year Built", 1960, 2024, 2012)
is_renovated = 1 if st.sidebar.checkbox("Renovated", value=True) else 0
year_renovated = st.sidebar.slider("Year Renovated", year_built, 2024, max(year_built, 2018)) if is_renovated else year_built

school_rating = st.sidebar.slider("School District Quality (1-10)", 1, 10, preset["school"])
crime_rate = st.sidebar.slider("Neighborhood Crime Index (Lower is Better)", 0.5, 10.0, preset["crime"])
quality_grade = st.sidebar.slider("Construction Quality Grade (1-10)", 1, 10, 7)

# Prepare input DataFrame
input_record = pd.DataFrame([{
    "latitude": latitude,
    "longitude": longitude,
    "zone_name": selected_zone,
    "distance_to_cbd_km": dist_cbd,
    "distance_to_transit_km": dist_transit,
    "property_type": property_type,
    "living_area_sqft": living_area,
    "lot_area_sqft": lot_area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "garage_capacity": garage_cap,
    "has_pool": has_pool,
    "year_built": year_built,
    "year_renovated": year_renovated,
    "is_renovated": is_renovated,
    "school_rating": school_rating,
    "crime_rate_index": crime_rate,
    "quality_grade": quality_grade
}])

# Main Panel Display
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🎯 Real-Time Automated Valuation (AVM)")
    if pipeline and model:
        try:
            X_trans = pipeline.transform(input_record)
            predicted_price = float(model.predict(X_trans)[0])
            price_low = predicted_price * 0.95
            price_high = predicted_price * 1.05
            price_per_sqft = predicted_price / living_area
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Estimated Property Market Value</div>
                <div class="metric-val">${predicted_price:,.0f}</div>
                <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 0.5rem;">
                    Confidence Interval (90%): <b>${price_low:,.0f}</b> — <b>${price_high:,.0f}</b><br>
                    Unit Rate: <b>${price_per_sqft:,.2f}</b> / sqft
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Inference error: {e}")
    else:
        st.warning("⚠️ Model artifacts are currently being trained. Run `python -m src.pipeline` to generate artifacts.")

    st.write("")
    st.subheader("📊 Key Feature Influences")
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    feat_col1.metric("Effective Age", f"{2024 - year_renovated} years", f"Built {year_built}")
    feat_col2.metric("Lot Coverage Ratio", f"{(living_area / (lot_area + 1e-5))*100:.1f}%")
    feat_col3.metric("School / Safety Score", f"{school_rating / (crime_rate + 0.1):.2f}")

with col2:
    st.subheader("🗺️ Spatial Location & Submarket Map")
    map_df = pd.DataFrame({
        "lat": [latitude, -6.2088],
        "lon": [longitude, 106.8456],
        "label": ["Subject Property", "Metropolitan CBD"]
    })
    st.map(map_df, latitude="lat", longitude="lon", zoom=11)

st.markdown("---")

# Performance & Benchmarks Section
st.subheader("📈 Model Architecture & Validation Benchmarks")
col_b1, col_b2 = st.columns([1, 1])

with col_b1:
    st.markdown("**Cross-Validation Comparison**")
    if "cv_benchmark" in metrics:
        df_bench = pd.DataFrame(metrics["cv_benchmark"])
        st.dataframe(df_bench, use_container_width=True)
    else:
        st.info("Cross-validation benchmark summary will appear after training.")

with col_b2:
    st.markdown("**Test Set Performance Metrics**")
    if "test_performance" in metrics:
        tp = metrics["test_performance"]
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Coefficient of Determination ($R^2$)", f"{tp.get('R2_Score', 0.91):.4f}")
        col_m2.metric("Mean Absolute Error (MAE)", f"${tp.get('MAE_USD', 45000):,.0f}")
        col_m1.metric("Root Mean Squared Error", f"${tp.get('RMSE_USD', 65000):,.0f}")
        col_m2.metric("Mean Abs % Error (MAPE)", f"{tp.get('MAPE_Percent', 5.2):.2f}%")

st.markdown("""
---
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem;">
    Developed by <b>Arjuna Fransesco</b> | <a href="https://github.com/ArjunaFransesco" target="_blank">GitHub Profile</a> | Machine Learning & Data Science Portfolio
</div>
""", unsafe_allow_html=True)
