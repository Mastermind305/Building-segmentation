# Building-segmentation


# Drone-Image-Based-Landcover-Segmentation  
**Single-Class Building Segmentation using DeepLabV3 (ResNet50)**

Accurate binary building footprint extraction from high-resolution drone/UAV imagery using **DeepLabV3** (not DeepLabV3+).

- Task: Binary segmentation (building = 1, everything else = 0)  
- Model: DeepLabV3 with ResNet50 encoder (pretrained on ImageNet)  
- Fast inference, excellent boundary precision for building extraction

## Project Structure

Drone-Image-Based-Landcover-Segmentation/
├── best_checkpoint.pth          # Best trained DeepLabV3 model
├── check2.py
├── deemodel.py                  # DeepLabV3 model definition
├── generator2.py                # not required
├── project.ipynb                # ← MUST RUN FIRST for creating patches mange path and run until create patches function
├── requirements_pip.txt
├── testopti.py                  # Testing & prediction script
├── train4.py                    # Training script
├── trygen.py                     #this is the  data generator file
├── utils.py
├── images/                      # THIS IS FOR TESTING IMAGES ONLY FOR TEST IMAGES FOR INFERENCE
│   └── test/
│                         # Place your TEST IMAGES HERE
└──├ output_merged/               # Auto-created after patching
       ├── images/                  # FOR TRAINING IMAGES FOLDER STRUCTURE
│               ├── test/
│               ├── train/
│               └── val/
       └── masks/                   # FOR TRAINING MASK FOLDER STRUCTURE
              ├── test/
              ├── train/
              └── val/
