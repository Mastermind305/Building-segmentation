"""
Building Extraction & Regularization Pipeline - Gradio UI

This application provides a complete pipeline for extracting building footprints from satellite imagery:
1. Semantic segmentation using DeepLabV3
2. Raster-to-vector conversion (mask to shapefile)
3. Building polygon regularization
4. Interactive visualization with satellite base maps

Author: Building Segmentation Team
Dependencies: torch, gradio, rasterio, geopandas, folium, buildingregulariser (optional)
"""

import torch
import numpy as np
from PIL import Image
import gradio as gr
import rasterio
from rasterio.features import shapes
from pyproj import Transformer
from deemodel import prepare_model
from trygen import get_transform
import folium
import os
import geopandas as gpd
from shapely.geometry import shape
import time
import tempfile
import zipfile

# Set PROJ_LIB path
proj_lib_path = r"C:\Users\TCS\anaconda3\envs\newtorchenv\Library\share\proj"
os.environ['PROJ_LIB'] = proj_lib_path

# Check if buildingregulariser is available
try:
    from buildingregulariser import regularize_geodataframe
    REGULARIZER_AVAILABLE = True
except ImportError:
    REGULARIZER_AVAILABLE = False

# ---------------------- Helper: Extract Lat/Lon ----------------------
def get_latlon_from_tif(tif_path):
    """Extract center latitude/longitude from GeoTIFF metadata."""
    try:
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            crs = src.crs

        if crs is None:
            return "❌ No CRS found in file", None, None

        # Compute center
        easting = (bounds.left + bounds.right) / 2
        northing = (bounds.bottom + bounds.top) / 2

        # Convert to EPSG:4326 (lat/lon)
        epsg = crs.to_epsg() if crs.to_epsg() else 32644  # fallback for UTM zone 44N
        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)

        return f"📍 Approximate Center: Latitude {lat:.6f}, Longitude {lon:.6f}", lat, lon

    except Exception as e:
        return f"⚠️ Error reading GeoTIFF metadata: {e}", None, None

# ---------------------- Mask Generation Function ----------------------
def predict_and_save_mask(image_path, model, save_dir=None):
    """Generate binary mask from image using DeepLabV3 model."""
    if save_dir is None:
        save_dir = tempfile.mkdtemp()

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Read image using rasterio
    with rasterio.open(image_path) as src:
        meta = src.meta.copy()
        img_array = src.read([1, 2, 3])  # (3, H, W)
        img_rgb = np.transpose(img_array, (1, 2, 0))  # (H, W, 3)

    # Preprocess for model
    transform = get_transform()
    img_tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device).float()

    # Predict
    with torch.no_grad():
        output = model(img_tensor)["out"]
        pred = torch.sigmoid(output)
        pred_mask = (pred > 0.5).float().cpu().numpy().squeeze()

    # Convert mask to 0-1 for GeoTIFF
    pred_mask_01 = pred_mask.astype(np.uint8)

    # Save GeoTIFF mask
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(save_dir, f"{base_name}_mask.tif")

    meta.update({"count": 1, "dtype": "uint8"})
    with rasterio.open(mask_path, "w", **meta) as dst:
        dst.write(pred_mask_01, 1)

    # Create visualization overlay
    pred_mask_255 = (pred_mask * 255).astype(np.uint8)
    overlay_img = create_mask_overlay(img_rgb, pred_mask_255)

    return mask_path, Image.fromarray(img_rgb), Image.fromarray(overlay_img), pred_mask_255


def create_mask_overlay(image, mask_255, color=(255, 0, 0), alpha=0.4):
    """Create overlay of mask on image."""
    image = image.astype(np.uint8).copy()
    overlay = image.copy()

    # Create color overlay
    mask_binary = mask_255 > 0
    color_layer = np.zeros_like(image, dtype=np.uint8)
    color_layer[:] = color

    # Blend
    blended = (image * (1 - alpha) + color_layer * alpha).astype(np.uint8)
    overlay[mask_binary] = blended[mask_binary]

    return overlay


# ---------------------- Mask to Shapefile Conversion ----------------------
def mask_to_shapefile(mask_tif_path, output_dir=None, min_area=50):
    """Convert binary mask to shapefile."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(mask_tif_path) as src:
        mask_array = src.read(1)
        transform = src.transform
        crs = src.crs

        # Convert raster to vector polygons
        geometries = []
        values = []

        for geom, value in shapes(mask_array, transform=transform):
            if value > 0:
                geom_shape = shape(geom)
                if geom_shape.area > min_area:
                    geometries.append(geom_shape)
                    values.append(value)

    if len(geometries) == 0:
        return None, "No polygons found after filtering"

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {'value': values, 'area': [geom.area for geom in geometries]},
        geometry=geometries,
        crs=crs
    )

    # Save shapefile
    base_name = os.path.splitext(os.path.basename(mask_tif_path))[0]
    shp_path = os.path.join(output_dir, f"{base_name}.shp")

    try:
        gdf.to_file(shp_path, driver='ESRI Shapefile')
    except Exception:
        # Try with WKT CRS
        try:
            if crs is not None:
                crs_wkt = crs.to_wkt()
                gdf = gpd.GeoDataFrame(
                    {'value': values, 'area': [geom.area for geom in geometries]},
                    geometry=geometries,
                    crs=crs_wkt
                )
                gdf.to_file(shp_path, driver='ESRI Shapefile')
        except Exception:
            # Save without CRS
            gdf = gpd.GeoDataFrame(
                {'value': values, 'area': [geom.area for geom in geometries]},
                geometry=geometries
            )
            gdf.to_file(shp_path, driver='ESRI Shapefile')

    info = f"✅ Found {len(gdf)} buildings\n📐 Total area: {gdf['area'].sum():.2f} sq units"
    return gdf, shp_path, info


# ---------------------- Building Regularization ----------------------
def regularize_buildings(gdf, simplify_tolerance=2.0):
    """Regularize building polygons."""
    if not REGULARIZER_AVAILABLE:
        return gdf, "⚠️ buildingregulariser not installed - skipping regularization"

    start = time.time()
    regularized_gdf = regularize_geodataframe(gdf, simplify_tolerance=simplify_tolerance)
    elapsed = time.time() - start

    info = f"✅ Regularized {len(gdf)} polygons in {elapsed:.2f}s"
    return regularized_gdf, info


# ---------------------- Create Shapefile ZIP for Download ----------------------
def create_shapefile_zip(shp_path):
    """Create a ZIP file containing all shapefile components."""
    if shp_path is None:
        return None

    base_path = shp_path.replace('.shp', '')
    zip_path = base_path + '.zip'

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            file_path = base_path + ext
            if os.path.exists(file_path):
                zipf.write(file_path, os.path.basename(file_path))

    return zip_path

# ---------------------- Model Setup ----------------------
def load_model():
    """Load the trained DeepLabV3 model."""
    model = prepare_model(num_classes=1)  # Binary segmentation (building vs background)
    checkpoint = torch.load('best_checkpoint.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# ---------------------- Create Interactive Map with Buildings ----------------------
def create_building_map(original_gdf, regularized_gdf, lat, lon, output_path=None):
    """Create interactive Folium map with building overlays."""
    if output_path is None:
        output_path = os.path.join(tempfile.mkdtemp(), "map.html")

    # Create map with satellite tiles
    m = folium.Map(
        location=[lat, lon],
        zoom_start=16,
        max_zoom=20,
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
    )

    # Add original buildings layer
    if original_gdf is not None and len(original_gdf) > 0:
        original_gdf.to_crs(epsg=4326).explore(
            m=m,
            name="Detected Buildings",
            style_kwds={
                "color": "yellow",
                "weight": 2,
                "fillOpacity": 0.1,
                "fillColor": "yellow"
            },
        )

    # Add regularized buildings layer
    if regularized_gdf is not None and len(regularized_gdf) > 0:
        regularized_gdf.to_crs(epsg=4326).explore(
            m=m,
            name="Regularized Buildings",
            style_kwds={
                "fillColor": "red",
                "color": "blue",
                "weight": 3,
                "fillOpacity": 0.4
            },
        )

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add marker at center
    folium.Marker([lat, lon], tooltip="Image Center", icon=folium.Icon(color='red')).add_to(m)

    # Save map
    m.save(output_path)
    return output_path


# ---------------------- Complete Pipeline Function ----------------------
def process_building_extraction(
    file_obj,
    min_area=50,
    simplify_tolerance=2.0,
    enable_regularization=True
):
    """
    Complete pipeline: Image -> Mask -> Shapefile -> Regularization -> Map
    """
    global MODEL

    input_path = file_obj.name
    temp_dir = tempfile.mkdtemp()

    # Initialize outputs
    original_img = None
    overlay_img = None
    mask_download = None
    shapefile_download = None
    regularized_download = None
    map_html = None
    status_info = ""

    try:
        # Check if it's a GeoTIFF
        if not input_path.lower().endswith((".tif", ".tiff")):
            return None, None, None, None, None, None, "❌ Please upload a GeoTIFF (.tif) file"

        # Step 1: Extract coordinates
        status_info += "📍 STEP 1: Extracting coordinates...\n"
        latlon_msg, lat, lon = get_latlon_from_tif(input_path)
        status_info += latlon_msg + "\n\n"

        # Step 2: Generate mask
        status_info += "🔮 STEP 2: Running segmentation model...\n"
        mask_path, original_img, overlay_img, mask_array = predict_and_save_mask(input_path, MODEL, temp_dir)
        status_info += f"✅ Mask saved: {os.path.basename(mask_path)}\n\n"
        mask_download = mask_path

        # Step 3: Convert to shapefile
        status_info += "🗺️ STEP 3: Converting mask to shapefile...\n"
        result = mask_to_shapefile(mask_path, temp_dir, min_area)

        if result[0] is None:
            status_info += "❌ " + result[1]
            return original_img, overlay_img, mask_download, None, None, None, status_info

        original_gdf, shp_path, shp_info = result
        status_info += shp_info + "\n\n"

        # Create ZIP for original shapefile
        shapefile_download = create_shapefile_zip(shp_path)

        # Step 4: Regularization (optional)
        regularized_gdf = None
        if enable_regularization:
            status_info += "🏗️ STEP 4: Regularizing building polygons...\n"
            regularized_gdf, reg_info = regularize_buildings(original_gdf, simplify_tolerance)
            status_info += reg_info + "\n\n"

            # Save regularized shapefile
            if regularized_gdf is not None and REGULARIZER_AVAILABLE:
                reg_shp_path = os.path.join(temp_dir, os.path.basename(shp_path).replace('.shp', '_regularized.shp'))
                regularized_gdf.to_file(reg_shp_path, driver="ESRI Shapefile")
                regularized_download = create_shapefile_zip(reg_shp_path)

        # Step 5: Create interactive map
        if lat and lon:
            status_info += "🗺️ STEP 5: Creating interactive map...\n"
            map_html = create_building_map(original_gdf, regularized_gdf, lat, lon, os.path.join(temp_dir, "map.html"))
            status_info += "✅ Interactive map created!\n\n"

        status_info += "="*50 + "\n"
        status_info += "✅ PIPELINE COMPLETE!\n"
        status_info += "="*50 + "\n\n"

        # Summary statistics
        status_info += "📊 SUMMARY:\n"
        status_info += f"  • Buildings detected: {len(original_gdf)}\n"
        status_info += f"  • Total area: {original_gdf['area'].sum():.2f} sq units\n"
        status_info += f"  • Average area: {original_gdf['area'].mean():.2f} sq units\n"

        if regularized_gdf is not None and REGULARIZER_AVAILABLE:
            status_info += f"  • Regularized buildings: {len(regularized_gdf)}\n"

    except Exception as e:
        status_info += f"\n❌ ERROR: {str(e)}"
        import traceback
        status_info += f"\n{traceback.format_exc()}"

    return original_img, overlay_img, mask_download, shapefile_download, regularized_download, map_html, status_info


# ---------------------- Gradio Interface ----------------------
def create_gradio_interface():
    """Create the Gradio interface."""

    with gr.Blocks(title="Building Extraction Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏗️ Building Extraction & Regularization Pipeline

            Upload a GeoTIFF satellite image to automatically:
            1. **Detect buildings** using DeepLabV3 segmentation
            2. **Convert to shapefile** (vector polygons)
            3. **Regularize building shapes** (optional)
            4. **Visualize on interactive map** with satellite imagery

            ---
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Input & Settings")

                file_input = gr.File(
                    label="Upload GeoTIFF Image",
                    file_types=[".tif", ".tiff"]
                )

                min_area = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=50,
                    step=10,
                    label="Minimum Building Area (sq units)",
                    info="Filter out small detections"
                )

                simplify_tolerance = gr.Slider(
                    minimum=0.5,
                    maximum=5.0,
                    value=2.0,
                    step=0.5,
                    label="Regularization Tolerance",
                    info="Higher = more simplified shapes"
                )

                enable_regularization = gr.Checkbox(
                    label="Enable Building Regularization",
                    value=REGULARIZER_AVAILABLE,
                    info="Straighten building edges" if REGULARIZER_AVAILABLE else "⚠️ buildingregulariser not installed"
                )

                process_btn = gr.Button("🚀 Process Image", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")

                status_output = gr.Textbox(
                    label="Processing Status",
                    lines=12,
                    max_lines=15,
                    show_label=True
                )

        gr.Markdown("---")

        with gr.Row():
            original_output = gr.Image(label="Original Image", type="pil")
            overlay_output = gr.Image(label="Segmentation Overlay", type="pil")

        gr.Markdown("### 📥 Downloads")

        with gr.Row():
            mask_output = gr.File(label="Binary Mask (GeoTIFF)")
            shapefile_output = gr.File(label="Buildings Shapefile (ZIP)")
            regularized_output = gr.File(label="Regularized Shapefile (ZIP)")

        gr.Markdown("### 🗺️ Interactive Map")

        map_output = gr.HTML(label="Building Map")

        # Connect the process button
        process_btn.click(
            fn=process_building_extraction,
            inputs=[file_input, min_area, simplify_tolerance, enable_regularization],
            outputs=[
                original_output,
                overlay_output,
                mask_output,
                shapefile_output,
                regularized_output,
                map_output,
                status_output
            ]
        )

        gr.Markdown(
            """
            ---
            ### 💡 Tips:
            - **Minimum Area**: Increase to remove small noise detections
            - **Regularization**: Makes building shapes more rectangular and cleaner
            - **Downloads**: All shapefiles include .shp, .shx, .dbf, .prj files in ZIP format
            - **Map**: Toggle between original and regularized buildings in the layer control
            """
        )

    return demo


# ---------------------- Run ----------------------
if __name__ == "__main__":
    print("Loading model...")
    MODEL = load_model()
    print("✅ Model loaded successfully!")

    demo = create_gradio_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
