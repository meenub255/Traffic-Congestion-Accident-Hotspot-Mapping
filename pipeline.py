import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

print("🏎️ Starting Ultra-Advanced Traffic Hotspot Predictive Workflow...")

# ==========================================
# STEP 1: LOAD & CLEAN ACCIDENT DATA (NYPD)
# ==========================================
print("\n[1/5] Ingesting and pre-processing accident logs...")
df = pd.read_csv("database.csv")
df_cleaned = df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
df_cleaned = df_cleaned[df_cleaned['BOROUGH'] == 'BROOKLYN'].copy()

# Feature Engineering: Core Crash Severity and Time Profiles
df_cleaned['severity_score'] = (df_cleaned['PERSONS INJURED'] * 1) + (df_cleaned['PERSONS KILLED'] * 5)

# Extract Time Contexts for Feature Engineering
df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['DATE'] + ' ' + df_cleaned['TIME'], format='mixed')
df_cleaned['hour'] = df_cleaned['timestamp'].dt.hour
df_cleaned['is_rush_hour'] = df_cleaned['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
df_cleaned['is_night'] = df_cleaned['hour'].isin([23, 0, 1, 2, 3, 4]).astype(int)

# Convert to Spatial Layer
geometry = [Point(xy) for xy in zip(df_cleaned['LONGITUDE'], df_cleaned['LATITUDE'])]
accidents_gdf = gpd.GeoDataFrame(df_cleaned, geometry=geometry, crs="EPSG:4326")


# ==========================================
# STEP 2: DOWNLOAD EXTENDED ROAD & TRANSIT GEOMETRY
# ==========================================
print("\n[2/5] Downloading Brooklyn spatial network and public transit hubs...")
# Download Road Network
G = ox.graph_from_place("Brooklyn, New York City, New York, USA", network_type="drive")
G_proj = ox.project_graph(G, to_crs="EPSG:32618")
nodes, edges = ox.graph_to_gdfs(G_proj)

accidents_gdf = accidents_gdf.to_crs(nodes.crs)

# --- ADDITION 1: Fetch Public Transit Infrastructure Hubs ---
print("     -> Pulling subway station data tags via OSMnx...")
try:
    transit_gdf = ox.features_from_place("Brooklyn, New York City, New York, USA", tags={"station": "subway"})
    transit_gdf = transit_gdf.to_crs(nodes.crs)
    # Reduce to representative point geometry centroids for clean distance lookups
    transit_points = transit_gdf.geometry.centroid
except Exception as e:
    print(f"     ⚠️ Warning fetching transit tags ({e}). Defaulting to dummy coordinates.")
    transit_points = gpd.GeoSeries([nodes.geometry.iloc[0]], crs=nodes.crs)


# ==========================================
# STEP 3: SPATIAL JOIN (SNAPPING)
# ==========================================
print("\n[3/5] Performing spatial indexing and segment assignment...")
accidents_with_roads = gpd.sjoin_nearest(accidents_gdf, edges, how="left", distance_col="distance_to_road")
accidents_snapped = accidents_with_roads[accidents_with_roads["distance_to_road"] <= 30]


# ==========================================
# STEP 4: ADVANCED FEATURE ENGINEERING
# ==========================================
print("\n[4/5] Engineering advanced structural, temporal & proximity features...")

# --- ADDITION 2: Grouping Crash Counts by Temporal Rush Hour Windows ---
road_stats = accidents_snapped.groupby(['u', 'v', 'key']).agg(
    total_accidents=('UNIQUE KEY', 'count'),
    historical_severity=('severity_score', 'sum'),
    rush_hour_crashes=('is_rush_hour', 'sum'),
    night_crashes=('is_night', 'sum')
).reset_index()

edges_with_data = edges.merge(road_stats, on=['u', 'v', 'key'], how='left')
edges_with_data['total_accidents'] = edges_with_data['total_accidents'].fillna(0)
edges_with_data['historical_severity'] = edges_with_data['historical_severity'].fillna(0)
edges_with_data['rush_hour_crashes'] = edges_with_data['rush_hour_crashes'].fillna(0)
edges_with_data['night_crashes'] = edges_with_data['night_crashes'].fillna(0)

# Create Target Risk Tiers
def assign_risk_tier(count):
    if count == 0: return 0
    elif count <= 5: return 1
    else: return 2

edges_with_data['risk_tier'] = edges_with_data['total_accidents'].apply(assign_risk_tier)

# Clean Numeric Attributes
def clean_osmnx_numeric(val):
    if isinstance(val, (list, np.ndarray, set)):
        val = list(val)[0] if len(val) > 0 else np.nan
    if pd.isna(val) or val is None: return np.nan
    cleaned = ''.join(c for c in str(val) if c.isdigit())
    return int(cleaned) if cleaned else np.nan

edges_with_data['clean_maxspeed'] = edges_with_data['maxspeed'].apply(clean_osmnx_numeric)
edges_with_data['clean_lanes'] = edges_with_data['lanes'].apply(clean_osmnx_numeric)
edges_with_data['clean_maxspeed'] = edges_with_data['clean_maxspeed'].fillna(edges_with_data['clean_maxspeed'].median())
edges_with_data['clean_lanes'] = edges_with_data['clean_lanes'].fillna(edges_with_data['clean_lanes'].median())
edges_with_data['road_length'] = edges_with_data.geometry.length

# Node Intersection Complexity
edges_with_data = edges_with_data.join(nodes[['street_count']], on='u', rsuffix='_start')
edges_with_data = edges_with_data.join(nodes[['street_count']], on='v', rsuffix='_end')
edges_with_data['intersection_complexity'] = edges_with_data['street_count'].fillna(0) + edges_with_data['street_count_end'].fillna(0)

# Encode Road Types
edges_with_data['highway_clean'] = edges_with_data['highway'].apply(lambda x: x[0] if isinstance(x, list) else str(x))
top_highways = edges_with_data['highway_clean'].value_counts().index[:5]
for hw in top_highways:
    edges_with_data[f'is_hw_{hw}'] = (edges_with_data['highway_clean'] == hw).astype(int)

# --- ADDITION 1 (Execution): Calculate Transit Proximity Matrix ---
print("     -> Calculating minimum spatial distance profiles to nearest subway nodes...")
edges_with_data['dist_to_transit'] = edges_with_data.geometry.apply(lambda geom: transit_points.distance(geom).min())


# ==========================================
# STEP 5: TRAIN ENSEMBLE MODELS (RF vs XGBOOST)
# ==========================================
print("\n[5/5] Executing Comparative Model Framework...")

# Compile our completely expanded feature matrix
feature_columns = [
    'clean_maxspeed', 'clean_lanes', 'road_length', 
    'intersection_complexity', 'historical_severity',
    'rush_hour_crashes', 'night_crashes', 'dist_to_transit'
] + [f'is_hw_{hw}' for hw in top_highways]

X = edges_with_data[feature_columns]
y = edges_with_data['risk_tier']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model A: Random Forest Baseline
rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(f"🌲 Random Forest Validation Accuracy: {accuracy_score(y_test, rf_pred) * 100:.2f}%")

# --- ADDITION 3: Train Gradient Boosted Trees via XGBoost ---
xgb_model = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"🚀 XGBoost Optimized Validation Accuracy: {xgb_acc * 100:.2f}%")

print("\n📋 Detailed XGBoost Classification Metrics:")
print(classification_report(y_test, xgb_pred, target_names=["Low Risk (0)", "Medium Risk (1)", "High Risk (2)"]))

# Save the predictions from our winning model architecture
edges_with_data['predicted_risk'] = xgb_model.predict(X)
columns_to_export = [
    'geometry', 'total_accidents', 'risk_tier', 'predicted_risk', 
    'rush_hour_crashes', 'night_crashes', 'dist_to_transit'
]
edges_with_data[columns_to_export].to_file("brooklyn_predicted_risk.geojson", driver="GeoJSON")

print("\n🎉 Fully enhanced 'brooklyn_predicted_risk.geojson' file generated successfully!")
print("💡 Tip: You can now map 'rush_hour_crashes' or 'night_crashes' in QGIS separately to study temporal risks!")