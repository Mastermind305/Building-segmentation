
# 🏠 Drone Image–Based Building Segmentation

**High‑resolution UAV imagery → clean building footprints**  
Single‑class (binary) building segmentation using **DeepLabV3 with a ResNet‑50 encoder**, optimized for sharp boundaries and fast inference.

---

## ✨ Overview
This project performs **accurate building footprint extraction** from drone/orthophoto imagery. It is designed for practical GIS and remote‑sensing workflows where speed, precision, and reproducibility matter.

**Key highlights**
- 🎯 **Task**: Binary segmentation (Building = 1, Background = 0)
- 🧠 **Model**: DeepLabV3 + ResNet‑50 (ImageNet pretrained)
- 🛰️ **Input**: High‑resolution UAV / orthophoto imagery
- ✂️ **Patch‑based pipeline** for large images
- ⚡ **Fast inference** with crisp building boundaries

---

## 🎬 Demo
A short demo showing model predictions on drone imagery:

https://github.com/user-attachments/assets/fa810eb9-35e9-4035-9034-00f416296fee


---

## 📁 Project Structure

```
.
├── best_checkpoint.pth          # Best trained DeepLabV3 model weights
├── deemodel.py                  # DeepLabV3 (ResNet50) model definition
├── train4.py                    # Training script
├── testopti.py                  # Inference / prediction script
├── topolygons.py               # Convert raster predictions to vector polygons (GeoJSON/Shapefile)
├── ui.py                       # End-to-end UI for full segmentation pipeline
├── generator2.py                # Data generator utilities
├── trygen.py                    # Alternative data generator
├── utils.py                     # Helper & utility functions
├── check2.py                    # Debug / validation helpers
│
├── project.ipynb                # Patch creation notebook (run FIRST)
├── requirements_pip.txt         # Python dependencies
│
├── images/
│   └── test/
│       └── images               # Place full test images here (optional)
│
└── output_merged/               # Auto‑generated after patch creation
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── masks/
        ├── train/
        ├── val/
        └── test/
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Mastermind305/Building-segmentation.git
cd Building-segmentation
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements_pip.txt
```
> 💡 Recommended: use a virtual environment or Conda environment for clean dependency management.

---

## 🧩 Data Preparation (Patch Creation)

Large orthophotos cannot be fed directly into the network. This project uses a **patch‑based strategy**.

### 3️⃣ Create image & mask patches

1. Open **`project.ipynb`** in:
   - Jupyter Notebook
   - VS Code
   - or Google Colab

2. At the **top of the notebook**, update paths to:
   - 🛰️ Your **orthophoto image** (GeoTIFF or raster)
   - 🗺️ Your **building footprint shapefile (.shp)**

3. Run **only the cells up to and including**:
   **👉 “Create Patches” section**

This step will automatically:
- ✂️ Split large orthophotos into smaller patches (e.g., **513 × 513**)
- 🏷️ Rasterize building footprints into binary masks
- 🔀 Split data into **train / validation / test** sets
- 📦 Save everything under `output_merged/`

📂 Output structure:
```
output_merged/
├── images/{train,val,test}/
└── masks/{train,val,test}/
```

⚠️ **Important**: Do **NOT** run the entire notebook.  
Training and inference are handled by separate Python scripts.

---

## 🧠 Model Training

### 4️⃣ Train DeepLabV3

Once patches are created:
```bash
python train4.py
```

During training:
- DeepLabV3 with **ResNet‑50 encoder** is initialized
- ImageNet pretrained weights are used
- Best model is automatically saved as:
  ```
  best_checkpoint.pth
  ```

---

## 🔍 Inference & Prediction

### 5️⃣ Run inference on test data
```bash
python testopti.py
```

This will:
- Load `best_checkpoint.pth`
- Run predictions on test images
- Save **binary raster building masks**

---

## 🗺️ Raster to Vector Conversion (Building Footprints)

Deep learning models produce **raster masks**, but GIS workflows require **vector polygons**.

### 6️⃣ Convert raster masks to polygons
```bash
python topolygons.py
```

This script:
- Takes predicted raster masks
- Cleans noise and small artifacts
- Converts connected building regions into **vector polygons**
- Exports results as GIS-friendly formats (e.g., Shapefile / GeoJSON)

✔️ Output polygons can be directly used in:
- QGIS / ArcGIS
- Urban mapping pipelines
- Spatial analysis & reporting

---

## 🧭 End-to-End Pipeline UI

For users who prefer a **single-click workflow**, the project provides a full pipeline interface.

### 7️⃣ Run the complete pipeline
```bash
python ui.py
```

The UI enables:
- Selection of a patch
- Model inference
- Raster-to-vector conversion
- Regularization of building footprint 

🎯 Ideal for:
- Non-technical users
- GIS analysts
- Rapid demonstrations & deployment


## 📌 Use Cases

- 🏙️ Urban planning & building inventory
- 🛰️ Drone‑based land‑cover mapping
- 🗺️ GIS automation workflows
- 🧪 Research in remote sensing & computer vision

---

## 🚀 Why DeepLabV3?

- Atrous (dilated) convolutions → **large receptive field**
- Excellent **boundary preservation**
- Strong performance on **high‑resolution imagery**
- Proven architecture for semantic segmentation tasks

---


## 🤝 Acknowledgements

- PyTorch & TorchVision
- DeepLabV3 architecture
- UAV / Remote Sensing research community

---



