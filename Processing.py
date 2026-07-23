import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN, KMeans
from shapely.geometry import box
import warnings

warnings.filterwarnings('ignore')

species_df = pd.read_csv("species_list.csv")

master_features = []
all_species_dfs = []

for index, row in species_df.iterrows():
    sci_name = row['scientific_name']
    com_name = row['common_name']
    
    print(f"\n--- Processing {com_name} ({sci_name}) ---")
    
    # 1. Query GBIF API directly into memory
    match_url = f"https://api.gbif.org/v1/species/match?name={sci_name}"
    taxon_key = requests.get(match_url).json().get("usageKey")
    
    if not taxon_key:
        print(f"Could not find GBIF taxon key for {com_name} ({sci_name}). Skipping.")
        continue
        
    records = []
    limit, offset = 300, 0
    while len(records) < 9000:
        res = requests.get("https://api.gbif.org/v1/occurrence/search", params={
            "taxonKey": taxon_key, "hasCoordinate": "true", "limit": limit, "offset": offset
        }).json()
        
        batch = res.get("results", [])
        if not batch: break
        records.extend(batch)
        offset += limit
        if res.get("endOfRecords", True): break
        print(f"Retrieved {len(records)} records for {com_name} ({sci_name})...")

    # 2. Load into Pandas & Clean
    df = pd.DataFrame(records).dropna(subset=['decimalLatitude', 'decimalLongitude', 'month'])
    if df.empty: continue
    
    df['month'] = df['month'].astype(int)
    df_clean = df.dropna(subset=['decimalLatitude', 'decimalLongitude', 'month']).copy()
    df_clean['species_sci'] = sci_name
    df_clean['species_com'] = com_name
    
    # 1. COLUMN PRUNING: Discard the 150+ unused GBIF columns
    columns_to_keep = ['decimalLatitude', 'decimalLongitude', 'month', 'species_sci', 'species_com']
    df_clean = df_clean.dropna(subset=['decimalLatitude', 'decimalLongitude', 'month'])
    df_clean = df_clean[columns_to_keep].copy()
    
    # 2. COORDINATE ROUNDING: 4 decimal places gives ~11-meter precision (perfect for global maps)
    df_clean['decimalLatitude'] = df_clean['decimalLatitude'].round(4)
    df_clean['decimalLongitude'] = df_clean['decimalLongitude'].round(4)
    
    # 3. RANDOM SUBSAMPLING: Take a representative sample of up to 1,500 birds per species
    # This preserves the exact geographic spread without overloading the browser
    if len(df_clean) > 1500:
        df_clean = df_clean.sample(n=1500, random_state=42)
        
    all_species_dfs.append(df_clean)
    print(f"Downsampled {com_name} to {len(df_clean)} highly efficient points for web rendering.")
    
    print(f"Loaded {len(df_clean)} valid coordinate records across 12 months.")

    # 3. Spatiotemporal Loop (Month by Month)
    for current_month in range(1, 13):
        month_data = df_clean[df_clean['month'] == current_month]
        if len(month_data) < 30:
            print(f"Month {current_month:02d}: Insufficient data ({len(month_data)} points). Skipping.")
            continue
            
        x_coords = month_data['decimalLongitude'].values
        y_coords = month_data['decimalLatitude'].values
        
        # ---------------------------------------------------------
        # STEP A: DBSCAN Outlier Rejection
        # ---------------------------------------------------------
        radians_coords = np.radians(np.column_stack([y_coords, x_coords])) 
        epsilon_radians = 200 / 6371.0 
        
        db = DBSCAN(eps=epsilon_radians, min_samples=5, algorithm='ball_tree', metric='haversine')
        labels = db.fit_predict(radians_coords)
        
        core_mask = labels != -1
        
        core_x = x_coords[core_mask]
        core_y = y_coords[core_mask]
        
        if len(core_x) < 30:
            print(f"Month {current_month:02d}: Insufficient core data after dropping isolated outliers.")
            continue
            
        print(f"Month {current_month:02d}: Modeling {len(core_x)} clustered observations...")
        
        # ---------------------------------------------------------
        # STEP B: K-MEANS Multi-Modal Clustering (The Horseshoe Fix)
        # ---------------------------------------------------------
        lon_range = core_x.max() - core_x.min()
        lat_range = core_y.max() - core_y.min()
        
        # Determine how many sub-clusters we need based on spatial spread
        num_clusters = max(1, min(5, int((lat_range + lon_range) / 30)))
        
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(np.column_stack([core_x, core_y]))
        
        month_polygons = []
        
        # ---------------------------------------------------------
        # STEP C: Adaptive KDE for each Sub-Cluster
        # ---------------------------------------------------------
        for c in range(num_clusters):
            c_mask = cluster_labels == c
            cx = core_x[c_mask]
            cy = core_y[c_mask]
            
            # Skip tiny nodes
            if len(cx) < 15:
                continue
                
            # Adaptive Grid for this specific sub-cluster node
            c_lon_range = cx.max() - cx.min()
            c_lat_range = cy.max() - cy.min()
            
            buffer_x = max(0.5, c_lon_range * 0.15)
            buffer_y = max(0.5, c_lat_range * 0.15)
            
            x_grid = np.linspace(cx.min() - buffer_x, cx.max() + buffer_x, 100)
            y_grid = np.linspace(cy.min() - buffer_y, cy.max() + buffer_y, 100)
            X, Y = np.meshgrid(x_grid, y_grid)
            grid_positions = np.vstack([X.ravel(), Y.ravel()])
            
            grid_width = x_grid[1] - x_grid[0]
            grid_height = y_grid[1] - y_grid[0]

            # Adaptive Bandwidth (Smoothing)
            core_coords = np.vstack([cx, cy])
            kernel = gaussian_kde(core_coords)
            
            base_bw = kernel.scotts_factor()
            clamped_bw = max(0.05, min(base_bw * 0.4, 0.15))
            kernel.set_bandwidth(bw_method=clamped_bw)
            
            density = kernel(grid_positions)
            
            # Thresholding (Using your 0.03 preference)
            density_threshold = density.max() * 0.03
            core_indices = np.where(density > density_threshold)[0]
            
            if len(core_indices) == 0:
                continue
                
            for idx in core_indices:
                lon = grid_positions[0][idx]
                lat = grid_positions[1][idx]
                cell = box(lon - grid_width/2, lat - grid_height/2, 
                            lon + grid_width/2, lat + grid_height/2)
                month_polygons.append(cell)
                
        # ---------------------------------------------------------
        # STEP D: Stitching Sub-Clusters Together
        # ---------------------------------------------------------
        if month_polygons:
            temp_gdf = gpd.GeoDataFrame(geometry=month_polygons, crs="EPSG:4326")
            dissolved = temp_gdf.dissolve()
            
            if not dissolved.empty:
                master_features.append({
                    "geometry": dissolved.geometry.iloc[0],
                    "month": current_month,
                    "species_sci": sci_name,  # Updated property
                    "species_com": com_name   # New property
                })

# 4. Final File Export Operations
final_gdf = gpd.GeoDataFrame(master_features, crs="EPSG:4326")
final_gdf.to_file("master_habitats.geojson", driver="GeoJSON")

master_df = pd.concat(all_species_dfs)
final_points_gdf = gpd.GeoDataFrame(
    master_df, 
    geometry=gpd.points_from_xy(master_df.decimalLongitude, master_df.decimalLatitude), 
    crs="EPSG:4326"
)
final_points_gdf.to_file("master_points.geojson", driver="GeoJSON")

print("\nDone! Outputs saved directly to master_habitats.geojson and master_points.geojson.")