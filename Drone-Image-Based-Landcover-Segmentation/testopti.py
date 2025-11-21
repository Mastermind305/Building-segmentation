


import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import rasterio
import cv2

from trygen import NepalDataset, get_transform
from deemodel import prepare_model  # your DeepLabV3+ model



def test_model(model, dataloader, save_dir="predictions", device=None):
    save_dir_255 = os.path.join(save_dir, "mask_0_255")
    overlay_dir = os.path.join(save_dir, "overlay")
    os.makedirs(save_dir_255, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (images, paths) in enumerate(tqdm(dataloader, desc="Testing")):
            images = images.to(device)

            outputs = model(images)["out"]
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()  

            for b in range(images.size(0)):
                pred_mask = preds[b].cpu().numpy().squeeze() 
                pred_mask_255 = (pred_mask * 255).astype(np.uint8)

                img_path = paths[b]
                base_name = os.path.splitext(os.path.basename(img_path))[0]

               
                with rasterio.open(img_path) as src:
                    meta = src.meta.copy()
                    img_array = src.read([1, 2, 3])  
                    img_array = np.transpose(img_array, (1, 2, 0))  

                meta.update({
                    "count": 1,
                    "dtype": "uint8"
                })
                out_path_255 = os.path.join(save_dir_255, f"{base_name}_255.tif")
                with rasterio.open(out_path_255, "w", **meta) as dst:
                    dst.write(pred_mask_255, 1)

                overlay = create_overlay(img_array, pred_mask_255)
                overlay_path = os.path.join(overlay_dir, f"{base_name}_overlay.png")
                cv2.imwrite(overlay_path, overlay)

    print(f"✅ Saved 0–255 masks in: {save_dir_255}")
    print(f"✅ Saved overlay images in: {overlay_dir}")



def create_overlay(image, mask_255, color=(255, 0, 0), alpha=0.6):
  
    image = image.astype(np.uint8)
    mask_color = np.zeros_like(image)
    mask_color[:, :] = color
    mask_binary = (mask_255 == 255).astype(np.uint8)[..., np.newaxis]

    overlay = cv2.addWeighted(image, 1.0, mask_color * mask_binary, alpha, 0)
    return overlay



if __name__ == "__main__":
    data_path = r"P:\Drone-Image-Based-Landcover-Segmentation"  #  Update as needed

    test_dataset = NepalDataset(
        data_path,
        subset="test",
        transform=get_transform(),
        test_mode=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = prepare_model(num_classes=1)
    # model.load_state_dict(torch.load("best_checkpoint.pth", map_location="cpu"))
    checkpoint = torch.load("best_checkpoint.pth", map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])


    test_model(
        model,
        test_loader,
        save_dir="predictions2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
