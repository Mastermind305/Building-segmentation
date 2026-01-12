# Building-segmentation


# Drone-Image-Based-Landcover-Segmentation  
**Single-Class Building Segmentation using DeepLabV3 (ResNet50)**

Accurate binary building footprint extraction from high-resolution drone/UAV imagery using **DeepLabV3** 

- Task: Binary segmentation (building = 1, everything else = 0)  
- Model: DeepLabV3 with ResNet50 encoder (pretrained on ImageNet)  
- Fast inference, excellent boundary precision for building extraction

###  Demo Video





https://github.com/user-attachments/assets/fa810eb9-35e9-4035-9034-00f416296fee



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

