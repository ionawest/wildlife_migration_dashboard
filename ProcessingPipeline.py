import os
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN, KMeans
from shapely.geometry import box
import warnings

warnings.filterwarnings('ignore')

# 1. CREATE THE CACHE DIRECTORY
os.makedirs("cache", exist_ok=True)

species_df = pd.read_csv("species_list.csv")

# We will collect GeoDataFrames directly instead of raw dictionaries
master_habitat_gdfs = []
master_points_gdfs = []

for index, row in species_df.iterrows():
    sci_name = row['scientific_name']
    com_name = row['common_name']
    
    # Create safe filenames by replacing spaces with underscores
    safe_sci_name = sci_name.replace(" ", "_")
    cache_points_file = f"cache/{safe_sci_name}_points.geojson"
    cache_habitats_file = f"cache/{safe_sci_name}_habitats.geojson"
    
    print(f"\n--- Processing {com_name} ({sci_name}) ---")
    
    # ========================================================
    # 2. THE CACHE CHECK (INCREMENTAL PROCESSING)
    # ========================================================
    if os.path.exists(cache_points_file) and os.path.exists(cache_habitats_file):
        print(f"[\u2713] Cached data found. Skipping API and KDE math for {com_name}...")
        master_points_gdfs.append(gpd.read_file(cache_points_file))
        master_habitat_gdfs.append(gpd.read_file(cache_habitats_file))
        continue # Instantly skip to the next bird in the CSV!
        
    # ========================================================
    # 3. FRESH PROCESSING (If not in cache)
    # ========================================================
    print(f"[!] No cache found. Querying GBIF API for {com_name}...")
    match_url = f"https://api.gbif.org/v1/species/match?name={sci_name}"
    taxon_key = requests.get(match_url).json().get("usageKey")
    
    if not taxon_key:
        print(f"Could not find GBIF taxon key for {sci_name}. Skipping.")
        continue
        
    records = []
    for m in range(1, 13):
        limit, offset = 300, 0
        month_records = []
        
        # Pull up to 750 records per month (12 x 750 = 9000 total)
        while len(month_records) < 750:
            res = requests.get("https://api.gbif.org/v1/occurrence/search", params={
                "taxonKey": taxon_key, 
                "hasCoordinate": "true", 
                "month": m,             # <--- We force the API to filter by month here
                "limit": limit, 
                "offset": offset
            }).json()
            
            batch = res.get("results", [])
            if not batch: break
            month_records.extend(batch)
            offset += limit
            if res.get("endOfRecords", True): break
            
        records.extend(month_records)
        
    df = pd.DataFrame(records).dropna(subset=['decimalLatitude', 'decimalLongitude', 'month'])
    if df.empty: continue
    
    df['month'] = df['month'].astype(int)
    
    # Inject names and implement the data diet (Pruning & Rounding)
    df['species_sci'] = sci_name
    df['species_com'] = com_name
    
    cols_to_keep = ['decimalLatitude', 'decimalLongitude', 'month', 'species_sci', 'species_com']
    df_clean = df[cols_to_keep].copy()
    
    df_clean['decimalLatitude'] = df_clean['decimalLatitude'].round(4)
    df_clean['decimalLongitude'] = df_clean['decimalLongitude'].round(4)
    
    # Ensure balanced temporal representation 
    balanced_dfs = []
    for m in range(1, 13):
        month_subset = df_clean[df_clean['month'] == m]
        if len(month_subset) > 150:
            month_subset = month_subset.sample(n=150, random_state=42)
        balanced_dfs.append(month_subset)
    
    df_clean_export = pd.concat(balanced_dfs)
    # Export this specific bird's raw points to the Cache
    species_points_gdf = gpd.GeoDataFrame(
        df_clean_export, geometry=gpd.points_from_xy(df_clean_export.decimalLongitude, df_clean_export.decimalLatitude), crs="EPSG:4326"
    )
    species_points_gdf.to_file(cache_points_file, driver="GeoJSON")
    master_points_gdfs.append(species_points_gdf)

    # ---------------------------------------------------------
    # Mathematical Habitat Modeling
    # ---------------------------------------------------------
    species_habitat_features = []
    
    for current_month in range(1, 13):
        month_data = df_clean[df_clean['month'] == current_month]
        if len(month_data) < 30: continue
            
        x_coords, y_coords = month_data['decimalLongitude'].values, month_data['decimalLatitude'].values
        
        # DBSCAN Outlier Rejection
        radians_coords = np.radians(np.column_stack([y_coords, x_coords])) 
        db = DBSCAN(eps=200/6371.0, min_samples=5, algorithm='ball_tree', metric='haversine')
        labels = db.fit_predict(radians_coords)
        
        core_mask = labels != -1
        core_x, core_y = x_coords[core_mask], y_coords[core_mask]
        
        if len(core_x) < 30: continue
        
        # K-MEANS Multi-Modal Clustering
        lon_range, lat_range = core_x.max() - core_x.min(), core_y.max() - core_y.min()
        num_clusters = max(1, min(5, int((lat_range + lon_range) / 30)))
        
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(np.column_stack([core_x, core_y]))
        
        month_polygons = []
        
        for c in range(num_clusters):
            cx, cy = core_x[cluster_labels == c], core_y[cluster_labels == c]
            if len(cx) < 15: continue
                
            buffer_x, buffer_y = max(0.5, (cx.max() - cx.min()) * 0.15), max(0.5, (cy.max() - cy.min()) * 0.15)
            x_grid = np.linspace(cx.min() - buffer_x, cx.max() + buffer_x, 100)
            y_grid = np.linspace(cy.min() - buffer_y, cy.max() + buffer_y, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            grid_positions = np.vstack([X.ravel(), Y.ravel()])
            
            grid_width, grid_height = x_grid[1] - x_grid[0], y_grid[1] - y_grid[0]

            kernel = gaussian_kde(np.vstack([cx, cy]))
            kernel.set_bandwidth(bw_method=max(0.05, min(kernel.scotts_factor() * 0.4, 0.15)))
            
            density = kernel(grid_positions)
            core_indices = np.where(density > density.max() * 0.03)[0]
            
            for idx in core_indices:
                lon, lat = grid_positions[0][idx], grid_positions[1][idx]
                month_polygons.append(box(lon - grid_width/2, lat - grid_height/2, lon + grid_width/2, lat + grid_height/2))
                
        if month_polygons:
            dissolved = gpd.GeoDataFrame(geometry=month_polygons, crs="EPSG:4326").dissolve()
            if not dissolved.empty:
                species_habitat_features.append({
                    "geometry": dissolved.geometry.iloc[0],
                    "month": current_month,
                    "species_sci": sci_name,
                    "species_com": com_name
                })
                
    # Export this specific bird's mathematical habitats to the Cache
    if species_habitat_features:
        species_hab_gdf = gpd.GeoDataFrame(species_habitat_features, crs="EPSG:4326")
        species_hab_gdf.to_file(cache_habitats_file, driver="GeoJSON")
        master_habitat_gdfs.append(species_hab_gdf)

# ========================================================
# 4. STITCHING THE MASTER FILES & NEW MATHEMATICS
# ========================================================
print("\n--- Stitching Data & Calculating Corridors ---")
if master_habitat_gdfs:
    final_hab_gdf = pd.concat(master_habitat_gdfs, ignore_index=True)
    
    # 1. AREA MATH: Calculate physical habitat area in square kilometers for the UI Chart
    # We temporarily convert to a cylindrical equal-area projection (EPSG:6933) for accurate math
    final_hab_gdf['area_sqkm'] = final_hab_gdf.to_crs(epsg=6933).geometry.area / 10**6
    
    # 2. MIGRATION CORRIDORS: Connect the monthly centroids
    corridor_features = []
    from shapely.geometry import LineString
    
    for sci_name, group in final_hab_gdf.groupby('species_sci'):
        # Ensure months are in chronological order
        group = group.sort_values('month')
        centroids = group.geometry.centroid.tolist()
        
        # We need at least 2 months of data to draw a migration line
        if len(centroids) > 1:
            migration_line = LineString(centroids)
            corridor_features.append({
                "geometry": migration_line,
                "species_sci": sci_name,
                "species_com": group.iloc[0]['species_com']
            })
            
    # Save the new layers
    final_hab_gdf.to_file("master_habitats.geojson", driver="GeoJSON")
    
    if corridor_features:
        corridor_gdf = gpd.GeoDataFrame(corridor_features, crs="EPSG:4326")
        corridor_gdf.to_file("master_corridors.geojson", driver="GeoJSON")

if master_points_gdfs:
    final_pts_gdf = pd.concat(master_points_gdfs, ignore_index=True)
    final_pts_gdf.to_file("master_points.geojson", driver="GeoJSON")

print("Done! Web dashboard is ready to refresh.")