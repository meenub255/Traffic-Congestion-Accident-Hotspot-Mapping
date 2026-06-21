import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("🏎️ Starting Traffic Congestion & Accident Hotspot Mapping Workflow...")

# ==========================================
# STEP 1: LOAD & CLEAN ACCIDENT DATA (NYPD)
# ==========================================
print("\n[1/5] Loading and cleaning accidents dataset...")
df = pd.read_csv("database.csv")

# Clean rows missing coordinates
df_cleaned = df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()

# Focus strictly on Brooklyn
df_cleaned = df_cleaned[df_cleaned['BOROUGH'] == 'BROOKLYN'].copy()

# Feature Engineering: Timestamps and Severity
df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['DATE'] + ' ' + df_cleaned['TIME'], format='mixed')
df_cleaned['hour'] = df_cleaned['timestamp'].dt.hour
df_cleaned['day_of_week'] = df_cleaned['timestamp'].dt.dayofweek
df_cleaned['severity_score'] = (df_cleaned['PERSONS INJURED'] * 1) + (df_cleaned['PERSONS KILLED'] * 5)

# Convert to Spatial GeoDataFrame
geometry = [Point(xy) for xy in zip(df_cleaned['LONGITUDE'], df_cleaned['LATITUDE'])]
accidents_gdf = gpd.GeoDataFrame(df_cleaned, geometry=geometry, crs="EPSG:4326")


# ==========================================
# STEP 2: DOWNLOAD ROAD NETWORK (OSMnx)
# ==========================================
print("\n[2/5] Fetching Brooklyn road network from OpenStreetMap...")
G = ox.graph_from_place("Brooklyn, New York City, New York, USA", network_type="drive")
G_proj = ox.project_graph(G, to_crs="EPSG:32618") # Projecting to metric UTM zone
nodes, edges = ox.graph_to_gdfs(G_proj)

# Match coordinate systems
accidents_gdf = accidents_gdf.to_crs(nodes.crs)


# ==========================================
# STEP 3: SPATIAL JOIN
# ==========================================
print("\n[3/5] Snapping accidents to nearest road segments...")
accidents_with_roads = gpd.sjoin_nearest(accidents_gdf, edges, how="left", distance_col="distance_to_road")
accidents_snapped = accidents_with_roads[accidents_with_roads["distance_to_road"] <= 30]


# ==========================================
# STEP 4: AGGREGATION & CLEANING
# ==========================================
print("\n[4/5] Engineering features on road segments...")
road_stats = accidents_snapped.groupby(['u', 'v', 'key']).agg(
    total_accidents=('UNIQUE KEY', 'count')
).reset_index()

edges_with_data = edges.merge(road_stats, on=['u', 'v', 'key'], how='left')
edges_with_data['total_accidents'] = edges_with_data['total_accidents'].fillna(0)

# Create 3 risk tiers (0=Low, 1=Medium, 2=High)
def assign_risk_tier(count):
    if count == 0: return 0
    elif count <= 5: return 1
    else: return 2

edges_with_data['risk_tier'] = edges_with_data['total_accidents'].apply(assign_risk_tier)

# Clean OSM features
def clean_osmnx_numeric(val):
    # 1. Handle arrays/lists/sets first by extracting the first element
    if isinstance(val, (list, np.ndarray, set)):
        if len(val) > 0:
            val = val[0]
        else:
            return np.nan
            
    # 2. Now safe to check for standard null values
    if pd.isna(val) or val is None: 
        return np.nan
        
    # 3. Strip text and keep only numbers
    cleaned = ''.join(c for c in str(val) if c.isdigit())
    return int(cleaned) if cleaned else np.nan

edges_with_data['clean_maxspeed'] = edges_with_data['maxspeed'].apply(clean_osmnx_numeric)
edges_with_data['clean_lanes'] = edges_with_data['lanes'].apply(clean_osmnx_numeric)
edges_with_data['clean_maxspeed'] = edges_with_data['clean_maxspeed'].fillna(edges_with_data['clean_maxspeed'].median())
edges_with_data['clean_lanes'] = edges_with_data['clean_lanes'].fillna(edges_with_data['clean_lanes'].median())
edges_with_data['road_length'] = edges_with_data.geometry.length


# ==========================================
# STEP 5: TRAIN CLASSIFIER
# ==========================================
print("\n[5/5] Training Random Forest Classifier...")
X = edges_with_data[['clean_maxspeed', 'clean_lanes', 'road_length']]
y = edges_with_data['risk_tier']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"   -> Model Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")

# Save predictions to file
edges_with_data['predicted_risk'] = model.predict(X)
columns_to_keep = ['geometry', 'total_accidents', 'risk_tier', 'clean_maxspeed', 'clean_lanes', 'road_length', 'predicted_risk']
edges_with_data[columns_to_keep].to_file("brooklyn_predicted_risk.geojson", driver="GeoJSON")

print("\n🎉 Output file generated successfully: 'brooklyn_predicted_risk.geojson'")