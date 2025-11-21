# Building-segmentation


# Drone-Image-Based-Landcover-Segmentation  
**Single-Class Building Segmentation using DeepLabV3 (ResNet50)**

Accurate binary building footprint extraction from high-resolution drone/UAV imagery using **DeepLabV3** 

- Task: Binary segmentation (building = 1, everything else = 0)  
- Model: DeepLabV3 with ResNet50 encoder (pretrained on ImageNet)  
- Fast inference, excellent boundary precision for building extraction

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

How to Use (Step-by-Step)




