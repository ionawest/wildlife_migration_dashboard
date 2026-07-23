import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

print("Reprojecting WorldClim data to Web Mercator...")
print("This involves non-linear warping and will take roughly 1-2 minutes...")

dst_crs = 'EPSG:3857'
# Web Mercator mathematical limits (cuts off the infinite poles)
bounds_4326 = (-180.0, -85.051129, 180.0, 85.051129)

for m in range(1, 13):
    month_str = f"{m:02d}"
    
    # 1. Precipitation
    rain_png = f"climate_data/precip_{month_str}.png"
    with rasterio.open(f"climate_data/wc2.1_10m_prec_{month_str}.tif") as src:
        # Calculate the new Web Mercator transform matrix and grid size
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *bounds_4326
        )
        
        dst_array = np.empty((height, width), dtype=np.float32)
        
        # Warp the array mathematically
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        
        arr = np.where(dst_array < 0, np.nan, dst_array)
        norm_arr = np.clip(arr / 400.0, 0, 1)
        cmap = plt.get_cmap('YlGnBu')
        colored = cmap(norm_arr)
        colored[np.isnan(arr), 3] = 0
        plt.imsave(rain_png, colored)

    # 2. Temperature
    temp_png = f"climate_data/temp_{month_str}.png"
    with rasterio.open(f"climate_data/wc2.1_10m_tavg_{month_str}.tif") as src:
        dst_array = np.empty((height, width), dtype=np.float32)
        
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        
        arr = np.where(dst_array < -100, np.nan, dst_array)
        norm_arr = np.clip((arr + 20) / 60.0, 0, 1)
        cmap = plt.get_cmap('jet')
        colored = cmap(norm_arr)
        colored[np.isnan(arr), 3] = 0
        plt.imsave(temp_png, colored)
        
    print(f"Generated warped Web Mercator maps for Month {month_str}")

print("Done! You now have accurately projected climate heatmaps.")