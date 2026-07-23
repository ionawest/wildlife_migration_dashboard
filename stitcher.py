import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from rasterstats import zonal_stats
import urllib.request
import zipfile
import warnings

warnings.filterwarnings('ignore')

# ========================================================
# 1. AUTOMATED CLIMATE DATA ACQUISITION
# ========================================================
def setup_climate_data():
    os.makedirs("climate_data", exist_ok=True)
    tavg_file = "climate_data/wc2.1_10m_tavg.zip"
    prec_file = "climate_data/wc2.1_10m_prec.zip"
    
    # We use a custom opener to bypass security blocks that reject Python bots
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)
    
    # Using the updated 'geodata.ucdavis.edu' servers
    if not os.path.exists("climate_data/wc2.1_10m_tavg_12.tif"):
        print("Downloading Global Temperature Models (approx 100MB)...")
        urllib.request.urlretrieve("https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_tavg.zip", tavg_file)
        with zipfile.ZipFile(tavg_file, 'r') as zip_ref:
            zip_ref.extractall("climate_data")
            
    if not os.path.exists("climate_data/wc2.1_10m_prec_12.tif"):
        print("Downloading Global Precipitation Models (approx 100MB)...")
        urllib.request.urlretrieve("https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_prec.zip", prec_file)
        with zipfile.ZipFile(prec_file, 'r') as zip_ref:
            zip_ref.extractall("climate_data")

setup_climate_data()

# ========================================================
# 2. READ EXISTING CACHE
# ========================================================
print("--- Reading from existing cache ---")
habitat_files = [f for f in os.listdir("cache") if f.endswith("_habitats.geojson")]
point_files = [f for f in os.listdir("cache") if f.endswith("_points.geojson")]

master_habitat_gdfs = []
master_points_gdfs = []

for file in habitat_files:
    master_habitat_gdfs.append(gpd.read_file(f"cache/{file}"))
    
for file in point_files:
    master_points_gdfs.append(gpd.read_file(f"cache/{file}"))

final_hab_gdf = pd.concat(master_habitat_gdfs, ignore_index=True)
final_pts_gdf = pd.concat(master_points_gdfs, ignore_index=True)

# ========================================================
# 3. ZONAL STATISTICS (Climate Math)
# ========================================================
print("Calculating Ecological Climate Math (Zonal Stats)...")
temp_list = []
rain_list = []

for idx, row in final_hab_gdf.iterrows():
    month_str = f"{int(row['month']):02d}"
    
    # Target the specific climate model for the specific month
    temp_tif = f"climate_data/wc2.1_10m_tavg_{month_str}.tif"
    rain_tif = f"climate_data/wc2.1_10m_prec_{month_str}.tif"
    
    try:
        # Calculate the mathematical mean of the pixels inside the polygon bounds
        t_stat = zonal_stats(row['geometry'], temp_tif, stats="mean")[0]['mean']
        r_stat = zonal_stats(row['geometry'], rain_tif, stats="mean")[0]['mean']
        
        temp_list.append(round(t_stat, 1) if t_stat is not None else -99)
        rain_list.append(round(r_stat, 1) if r_stat is not None else -99)
    except Exception:
        temp_list.append(-99)
        rain_list.append(-99)

# Inject the math back into the GeoJSON
final_hab_gdf['avg_temp_c'] = temp_list
final_hab_gdf['avg_rain_mm'] = rain_list

# ========================================================
# 4. MIGRATION CORRIDORS & EXPORT
# ========================================================
corridor_features = []
for sci_name, group in final_hab_gdf.groupby('species_sci'):
    group = group.sort_values('month')
    centroids = group.geometry.centroid.tolist()
    if len(centroids) > 1:
        corridor_features.append({
            "geometry": LineString(centroids),
            "species_sci": sci_name,
            "species_com": group.iloc[0]['species_com']
        })

print("Saving Master GeoJSON files for the web dashboard...")
final_hab_gdf.to_file("master_habitats.geojson", driver="GeoJSON")
final_pts_gdf.to_file("master_points.geojson", driver="GeoJSON")

if corridor_features:
    gpd.GeoDataFrame(corridor_features, crs="EPSG:4326").to_file("master_corridors.geojson", driver="GeoJSON")

print("Done! Refresh your web browser.")