# Building-segmentation


# Drone-Image-Based-Landcover-Segmentation  
**Single-Class Building Segmentation using DeepLabV3 (ResNet50)**

Accurate binary building footprint extraction from high-resolution drone/UAV imagery using **DeepLabV3** (not DeepLabV3+).

- Task: Binary segmentation (building = 1, everything else = 0)  
- Model: DeepLabV3 with ResNet50 encoder (pretrained on ImageNet)  
- Fast inference, excellent boundary precision for building extraction

## Project Structure

Drone-Image-Based-Landcover-Segmentation/
├── best_checkpoint.pth # Best trained DeepLabV3 model
├── check2.py # Script for checking results (optional)
├── deemodel.py # DeepLabV3 model definition
├── generator2.py # Not required for typical use
├── project.ipynb # Must run this first for creating patches (follow the instructions in this notebook)
├── requirements_pip.txt # List of dependencies to install
├── testopti.py # Script for testing and prediction
├── train4.py # Script for training the model
├── trygen.py # Data generator file (used for training)
├── utils.py # Utility functions (e.g., image transformations)
├── output_merged/ # Folder containing merged output after patching
│ ├── images/ # Folder structure for training images (output from patching)
│ │ ├── test/ # Test images (for evaluation)
│ │ ├── train/ # Training images
│ │ └── val/ # Validation images
│ └── masks/ # Folder structure for training masks (output from patching)
│ ├── test/ # Mask files for test images
│ ├── train/ # Mask files for training images
│ └── val/ # Mask files for validation images
├── images/ # Original training images folder structure
│ ├── test/ # Test images (for inference)
│ ├── train/ # Training images
│ └── val/ # Validation images
└── masks/ # Original masks folder structure
├── test/ # Mask files for test images
├── train/ # Mask files for training images
└── val/ # Mask files for validation images
