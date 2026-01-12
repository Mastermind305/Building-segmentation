

import os
import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape

def raster_to_shapefile(raster_path, shp_path, min_area=1):
    """
    Convert binary raster mask (0/1) into polygons and save as Shapefile.

    Args:
        raster_path (str): Path to binary mask raster (GeoTIFF).
        shp_path (str): Path to output Shapefile (.shp).
        min_area (float): Minimum polygon area to keep (filter small noise).
    """
    with rasterio.open(raster_path) as src:
        mask = src.read(1)  # read first band
        transform = src.transform
        crs = src.crs

        # Polygonize (raster -> vector)
        results = (
            {"properties": {"value": v}, "geometry": s}
            for s, v in shapes(mask, transform=transform)
            if v == 1  # keep only mask=1
        )

        # Convert to GeoDataFrame
        geoms = [shape(feature["geometry"]) for feature in results]
        gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)

        # Filter out small polygons
        gdf = gdf[gdf.geometry.area > min_area]

        # Save shapefile
        if len(gdf) > 0:
            gdf.to_file(shp_path, driver="ESRI Shapefile")
            print(f"✅ Shapefile saved: {shp_path}")
        else:
            print(f"⚠️ No polygons detected in {raster_path}")

def batch_polygonize(mask_folder, output_folder, min_area=10):
   
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(mask_folder):
        if file.endswith("_mask01.tif"):
            raster_path = os.path.join(mask_folder, file)
            shp_path = os.path.join(output_folder, file.replace("_mask01.tif", ".shp"))
            raster_to_shapefile(raster_path, shp_path, min_area=min_area)

if __name__ == "__main__":
    # Example usage
    mask_folder = r"D:\Drone-Image-Based-Landcover-Segmentation\predictions"
    output_folder = os.path.join(mask_folder, "vectors_shp")

    batch_polygonize(mask_folder, output_folder, min_area=50)
