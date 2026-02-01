# Building-segmentation


# Drone-Image-Based-Landcover-Segmentation  
**Single-Class Building Segmentation using DeepLabV3 (ResNet50)**

Accurate binary building footprint extraction from high-resolution drone/UAV imagery using **DeepLabV3** 

- Task: Binary segmentation (building = 1, everything else = 0)  
- Model: DeepLabV3 with ResNet50 encoder (pretrained on ImageNet)  
- Fast inference, excellent boundary precision for building extraction

###  Demo Video







## Project Structure

```bash
$ tree -L 3
.
├── best_checkpoint.pth          # Best trained DeepLabV3 model
├── check2.py
├── deemodel.py                  # DeepLabV3 model definition
├── generator2.py
├── images
│   └── test
│       └── images               # Place your test images here for inference
├── output_merged                # Auto-generated after running project.ipynb
│   ├── images                   # Predicted building masks
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── masks                    # Ground-truth binary masks (building = 1 or 255)
│       ├── test
│       ├── train
│       └── val
├── project.ipynb                # ← Run this first to create patches .Just until during create preatches udating paths
├── requirements_pip.txt
├── testopti.py                  # Inference / generate predictions
├── train4.py                    # Training script
├── trygen.py                     #data generator file
└── utils.py

```

## How to Use (Step-by-Step)



### 1. Clone the repository
```bash
git clone https://github.com/Mastermind305/Building-segmentation.git
```
### 2. Install dependencies
```bash
pip install -r requirements_pip.txt
```
### 3. Create patches

- Open `project.ipynb` (using Jupyter Notebook, VS Code, or Google Colab)
- Update the paths at the top of the notebook to point to:
  - Your file containing orthophoto
  - Your file containing .shp file
- Run **only the cells up to and including the "Create Patches" section**  
  → This will automatically:
  - Cut large orthophoto and masks into smaller patches (e.g., 513×513)
  - Split them into train/validation/test sets
  - Save everything inside the `output_merged/` folder as follows:
    - `output_merged/images/train/`, `val/`, `test/`
    - `output_merged/masks/train/`, `val/`, `test/`

**Do not run the entire notebook yet** — training and testing are done separately with `train4.py` and `testopti.py`.

### 4. Train the model
```bash
python train4.py
```

### 5. Generate predictions
```bash
python testopti.py
```

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



