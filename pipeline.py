import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("🏎️ Starting Advanced Traffic Hotspot Predictive Workflow...")

# ==========================================
# STEP 1: LOAD & CLEAN ACCIDENT DATA (NYPD)
# ==========================================
print("\n[1/5] Ingesting and pre-processing accident logs...")
df = pd.read_csv("database.csv")
df_cleaned = df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
df_cleaned = df_cleaned[df_cleaned['BOROUGH'] == 'BROOKLYN'].copy()

# Feature Engineering: Calculate Crash Severity
df_cleaned['severity_score'] = (df_cleaned['PERSONS INJURED'] * 1) + (df_cleaned['PERSONS KILLED'] * 5)

# Convert to Spatial Layer
geometry = [Point(xy) for xy in zip(df_cleaned['LONGITUDE'], df_cleaned['LATITUDE'])]
accidents_gdf = gpd.GeoDataFrame(df_cleaned, geometry=geometry, crs="EPSG:4326")


# ==========================================
# STEP 2: DOWNLOAD EXTENDED ROAD GEOMETRY
# ==========================================
print("\n[2/5] Downloading Brooklyn spatial network and intersection nodes...")
G = ox.graph_from_place("Brooklyn, New York City, New York, USA", network_type="drive")
G_proj = ox.project_graph(G, to_crs="EPSG:32618")
nodes, edges = ox.graph_to_gdfs(G_proj)

accidents_gdf = accidents_gdf.to_crs(nodes.crs)


# ==========================================
# STEP 3: SPATIAL JOIN (SNAPPING)
# ==========================================
print("\n[3/5] Performing spatial indexing and segment assignment...")
accidents_with_roads = gpd.sjoin_nearest(accidents_gdf, edges, how="left", distance_col="distance_to_road")
accidents_snapped = accidents_with_roads[accidents_with_roads["distance_to_road"] <= 30]


# ==========================================
# STEP 4: ADVANCED FEATURE ENGINEERING
# ==========================================
print("\n[4/5] Engineering advanced structural & network features...")

# Aggregate crashes AND historical severity scores per road segment
road_stats = accidents_snapped.groupby(['u', 'v', 'key']).agg(
    total_accidents=('UNIQUE KEY', 'count'),
    historical_severity=('severity_score', 'sum')
).reset_index()

edges_with_data = edges.merge(road_stats, on=['u', 'v', 'key'], how='left')
edges_with_data['total_accidents'] = edges_with_data['total_accidents'].fillna(0)
edges_with_data['historical_severity'] = edges_with_data['historical_severity'].fillna(0)

# Create Target Risk Tiers
def assign_risk_tier(count):
    if count == 0: return 0
    elif count <= 5: return 1
    else: return 2

edges_with_data['risk_tier'] = edges_with_data['total_accidents'].apply(assign_risk_tier)

# --- Feature 1 & 2: Clean Numeric Attributes ---
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

# --- Feature 3: Map Node Degrees (Intersection Complexity) ---
# Map how many roads connect to the start (u) and end (v) nodes of each segment
edges_with_data = edges_with_data.join(nodes[['street_count']], on='u', rsuffix='_start')
edges_with_data = edges_with_data.join(nodes[['street_count']], on='v', rsuffix='_end')
edges_with_data['intersection_complexity'] = edges_with_data['street_count'].fillna(0) + edges_with_data['street_count_end'].fillna(0)

# --- Feature 4: Encode Road Types (Highway Classification) ---
# Extract primary type string from OSMnx list formats
edges_with_data['highway_clean'] = edges_with_data['highway'].apply(lambda x: x[0] if isinstance(x, list) else str(x))
# One-hot encode the top 5 most common road types to avoid data explosion
top_highways = edges_with_data['highway_clean'].value_counts().index[:5]
for hw in top_highways:
    edges_with_data[f'is_hw_{hw}'] = (edges_with_data['highway_clean'] == hw).astype(int)


# ==========================================
# STEP 5: TRAIN TUNED MACHINE LEARNING MODEL
# ==========================================
print("\n[5/5] Training Optimized Random Forest Classifier...")

# Compile our expanded feature list
feature_columns = [
    'clean_maxspeed', 'clean_lanes', 'road_length', 
    'intersection_complexity', 'historical_severity'
] + [f'is_hw_{hw}' for hw in top_highways]

X = edges_with_data[feature_columns]
y = edges_with_data['risk_tier']

# Stratified split to ensure balanced risk representation across training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize classifier with regularized parameters to prevent overfitting
model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Output evaluation metrics
y_pred = model.predict(X_test)
print(f"\n🚀 Enhanced Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\n📋 Detailed Classification Metrics:")
print(classification_report(y_test, y_pred, target_names=["Low Risk (0)", "Medium Risk (1)", "High Risk (2)"]))

# Save updated risk layers back out
edges_with_data['predicted_risk'] = model.predict(X)
columns_to_export = ['geometry', 'total_accidents', 'risk_tier', 'predicted_risk'] + feature_columns
edges_with_data[columns_to_export].to_file("brooklyn_predicted_risk.geojson", driver="GeoJSON")

print("\n🎉 Upgraded 'brooklyn_predicted_risk.geojson' file saved. Re-import into QGIS to view your optimized roadmap!")