import os
import torch
import numpy as np
import rasterio
import cv2
import geopandas as gpd
import folium
import time
from pathlib import Path
from rasterio.features import shapes
from shapely.geometry import shape
import warnings

# Set PROJ_LIB path
proj_lib_path = r"C:\Users\TCS\anaconda3\envs\newtorchenv\Library\share\proj"
os.environ['PROJ_LIB'] = proj_lib_path

# Import from existing modules
from deemodel import prepare_model
from trygen import get_transform

try:
    from buildingregulariser import regularize_geodataframe
    REGULARIZER_AVAILABLE = True
except ImportError:
    print("⚠️ buildingregulariser not available. Regularization will be skipped.")
    REGULARIZER_AVAILABLE = False


def predict_and_save_mask(image_path, checkpoint_path, save_dir="single_prediction"):
    """Run inference on one image and save GeoTIFF mask."""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = prepare_model(num_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Loaded model weights from checkpoint: {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Read image using rasterio
    with rasterio.open(image_path) as src:
        meta = src.meta.copy()
        img_array = src.read([1, 2, 3])  # (3, H, W)
        img_rgb = np.transpose(img_array, (1, 2, 0))  # (H, W, 3)

    print(f"📸 Input image shape: {img_rgb.shape}")

    # Preprocess for model
    transform = get_transform()
    img_tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device).float()

    print(f"🔹 Transformed tensor shape: {img_tensor.shape}")

    # Predict
    with torch.no_grad():
        output = model(img_tensor)["out"]
        pred = torch.sigmoid(output)
        pred_mask = (pred > 0.5).float().cpu().numpy().squeeze()

    # Convert mask to 0-1 for GeoTIFF
    pred_mask_01 = pred_mask.astype(np.uint8)

    # Save GeoTIFF mask
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(save_dir, f"{base_name}_mask01.tif")

    meta.update({"count": 1, "dtype": "uint8"})
    with rasterio.open(mask_path, "w", **meta) as dst:
        dst.write(pred_mask_01, 1)

    print(f"✅ Saved mask (GeoTIFF): {mask_path}")
    return mask_path


def mask_to_shapefile(mask_tif_path, output_shp_path=None, min_area=1):
    """
    Convert a binary mask GeoTIFF to a shapefile with polygons.

    Args:
        mask_tif_path: Path to the mask .tif file
        output_shp_path: Path for output .shp file (optional)
        min_area: Minimum polygon area to keep (filter small noise)

    Returns:
        GeoDataFrame with the polygons
    """
    # Generate output path if not provided
    if output_shp_path is None:
        base_name = os.path.splitext(mask_tif_path)[0]
        output_shp_path = f"{base_name}.shp"

    print(f"Reading mask from: {mask_tif_path}")

    # Read the mask with rasterio
    with rasterio.open(mask_tif_path) as src:
        mask_array = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs

        print(f"CRS: {crs}")
        print(f"Shape: {mask_array.shape}")

        # Convert raster to vector polygons
        geometries = []
        values = []

        for geom, value in shapes(mask_array, transform=transform):
            if value > 0:  # Only keep masked areas (non-zero)
                geom_shape = shape(geom)
                if geom_shape.area > min_area:  # Filter by area
                    geometries.append(geom_shape)
                    values.append(value)

        print(f"Found {len(geometries)} polygon(s) after filtering")

    # Create GeoDataFrame
    if len(geometries) > 0:
        try:
            gdf = gpd.GeoDataFrame(
                {'value': values, 'area': [geom.area for geom in geometries]},
                geometry=geometries,
                crs=crs
            )

            # Save to shapefile with error handling
            try:
                gdf.to_file(output_shp_path, driver='ESRI Shapefile')
                print(f"✅ Shapefile saved successfully: {output_shp_path}")
            except Exception as e:
                print(f"WARNING: CRS Error: {e}")
                print("Trying alternative method with WKT CRS...")

                # Convert CRS to WKT format which is more compatible
                try:
                    if crs is not None:
                        crs_wkt = crs.to_wkt()
                        gdf_wkt = gpd.GeoDataFrame(
                            {'value': values, 'area': [geom.area for geom in geometries]},
                            geometry=geometries,
                            crs=crs_wkt
                        )
                        gdf_wkt.to_file(output_shp_path, driver='ESRI Shapefile')
                        print(f"✅ Shapefile saved with WKT CRS: {output_shp_path}")
                except Exception as e2:
                    # Save without CRS if all else fails
                    print(f"WARNING: WKT method also failed: {e2}")
                    print("Saving without CRS information (coordinates preserved)...")
                    gdf_no_crs = gpd.GeoDataFrame(
                        {'value': values, 'area': [geom.area for geom in geometries]},
                        geometry=geometries
                    )
                    gdf_no_crs.to_file(output_shp_path, driver='ESRI Shapefile')
                    print(f"✅ Shapefile saved (no CRS): {output_shp_path}")

            print(f"   Total polygons: {len(gdf)}")
            print(f"   Total area: {gdf['area'].sum():.2f} square units")

            return gdf

        except Exception as e:
            print(f"ERROR: Error creating shapefile: {e}")
            return None
    else:
        print("WARNING: No polygons found in the mask!")
        return None


def regularize_buildings(gdf, simplify_tolerance=2.0):
    """
    Regularize building polygons using buildingregulariser.

    Args:
        gdf: GeoDataFrame with building polygons
        simplify_tolerance: Tolerance for simplification

    Returns:
        Regularized GeoDataFrame
    """
    if not REGULARIZER_AVAILABLE:
        print("⚠️ Skipping regularization - buildingregulariser not available")
        return gdf

    start = time.time()
    regularized_gdf = regularize_geodataframe(
        gdf,
        simplify_tolerance=simplify_tolerance,
    )
    elapsed = time.time() - start
    print(f"⏱️ Regularized {len(gdf)} polygons in {elapsed:.2f}s ({len(gdf)/elapsed:.0f} polygons/sec)")
    return regularized_gdf


def get_latlon_from_tif(tif_path):
    """
    Extract center latitude and longitude from a GeoTIFF file.

    Args:
        tif_path: Path to GeoTIFF file

    Returns:
        tuple: (message, center_lat, center_lon)
    """
    try:
        with rasterio.open(tif_path) as src:
            # Get bounds
            bounds = src.bounds
            crs = src.crs

            # Calculate center in image coordinates
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.top + bounds.bottom) / 2

            # Transform to lat/lon (EPSG:4326)
            from rasterio.warp import transform
            lon, lat = transform(crs, 'EPSG:4326', [center_x], [center_y])

            return f"✅ Successfully extracted coordinates from {tif_path}", lat[0], lon[0]
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None


def create_visualization_map(original_gdf, regularized_gdf, center_coords, save_path="building_map.html"):
    """
    Create an interactive Folium map with original and regularized buildings.

    Args:
        original_gdf: GeoDataFrame with original buildings
        regularized_gdf: GeoDataFrame with regularized buildings
        center_coords: [lat, lon] for map center
        save_path: Path to save HTML map

    Returns:
        Folium map object
    """
    # Create Folium map with satellite tiles
    m = folium.Map(
        location=center_coords,
        zoom_start=140,
        max_zoom=200,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
    )

    # Add original buildings layer
    original_gdf.to_crs(epsg=4326).explore(
        m=m,
        name="Original Buildings",
        style_kwds={
            "color": "white",
            "weight": 3,
            "fillOpacity": 0.0,
        },
    )

    # Add regularized buildings layer
    if regularized_gdf is not None and len(regularized_gdf) > 0:
        regularized_gdf.to_crs(epsg=4326).explore(
            m=m,
            name="Regularized Buildings",
            style_kwds={"fillColor": "red", "color": "blue", "weight": 3, "fillOpacity": 0.4},
        )

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save map
    m.save(save_path)
    print(f"✅ Interactive map saved: {save_path}")
    return m


def process_image_to_buildings(image_path, checkpoint_path, min_area=50, simplify_tolerance=2.0, output_dir="building_output"):
    """
    Complete pipeline: Image -> Mask -> Shapefile -> Regularization -> Visualization

    Args:
        image_path: Path to input GeoTIFF image
        checkpoint_path: Path to model checkpoint
        min_area: Minimum polygon area to keep
        simplify_tolerance: Tolerance for regularization
        output_dir: Directory for all outputs

    Returns:
        tuple: (original_gdf, regularized_gdf)
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print("\n" + "="*60)
    print("🏗️  BUILDING EXTRACTION AND REGULARIZATION PIPELINE")
    print("="*60 + "\n")

    # Step 1: Generate mask
    print("STEP 1: Generating segmentation mask...")
    mask_path = predict_and_save_mask(image_path, checkpoint_path, save_dir=output_dir)

    # Step 2: Convert to shapefile
    print("\nSTEP 2: Converting mask to shapefile...")
    shp_path = os.path.join(output_dir, f"{base_name}.shp")
    original_gdf = mask_to_shapefile(mask_path, shp_path, min_area=min_area)

    if original_gdf is None or len(original_gdf) == 0:
        print("❌ No buildings detected. Pipeline stopped.")
        return None, None

    # Step 3: Regularize buildings
    print("\nSTEP 3: Regularizing building polygons...")
    regularized_gdf = regularize_buildings(original_gdf, simplify_tolerance=simplify_tolerance)

    # Save regularized shapefile
    if regularized_gdf is not None and len(regularized_gdf) > 0:
        regularized_shp_path = os.path.join(output_dir, f"{base_name}_regularized.shp")
        regularized_gdf.to_file(regularized_shp_path, driver="ESRI Shapefile")
        print(f"✅ Regularized shapefile saved: {regularized_shp_path}")

    # Step 4: Get center coordinates
    print("\nSTEP 4: Extracting center coordinates...")
    message, center_lat, center_lon = get_latlon_from_tif(image_path)
    print(message)

    if center_lat is not None and center_lon is not None:
        print(f"📍 Center Latitude: {center_lat:.6f}")
        print(f"📍 Center Longitude: {center_lon:.6f}")

        # Step 5: Create visualization
        print("\nSTEP 5: Creating interactive map...")
        map_path = os.path.join(output_dir, f"{base_name}_map.html")
        create_visualization_map(
            original_gdf,
            regularized_gdf,
            [center_lat, center_lon],
            save_path=map_path
        )
    else:
        print("⚠️ Could not create map - center coordinates not available")

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60 + "\n")

    return original_gdf, regularized_gdf


if __name__ == "__main__":
    # Configuration
    image_path = r"patch_59.tif"  # 🔹 Change this to your image
    checkpoint_path = "best_checkpoint.pth"
    output_dir = "building_output"
    min_area = 50  # Minimum polygon area (adjust based on your data)
    simplify_tolerance = 2.0  # Regularization tolerance

    # Run complete pipeline
    original_buildings, regularized_buildings = process_image_to_buildings(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        min_area=min_area,
        simplify_tolerance=simplify_tolerance,
        output_dir=output_dir
    )

    # Print summary statistics
    if original_buildings is not None:
        print("\n📊 SUMMARY STATISTICS:")
        print(f"   Original buildings: {len(original_buildings)}")
        if regularized_buildings is not None:
            print(f"   Regularized buildings: {len(regularized_buildings)}")
            print(f"   Average area (original): {original_buildings.geometry.area.mean():.2f}")
            print(f"   Average area (regularized): {regularized_buildings.geometry.area.mean():.2f}")
