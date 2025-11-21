

# import os
# import copy
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# from trygen import NepalDataset, get_transform
# from deemodel import prepare_model  # DeepLabV3 model
# from torchmetrics.classification import JaccardIndex
# from torch.utils.tensorboard import SummaryWriter   # NEW


# # ---- Custom BCE + Dice Loss ----
# class BCEDiceLoss(nn.Module):
#     def __init__(self, bce_weight=0.5):
#         super(BCEDiceLoss, self).__init__()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.bce_weight = bce_weight

#     def forward(self, inputs, targets):
#         # BCE loss
#         bce_loss = self.bce(inputs, targets)

#         # Dice loss
#         inputs = torch.sigmoid(inputs)  # convert logits -> probabilities
#         smooth = 1.0
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         intersection = (inputs * targets).sum()
#         dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

#         return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


# def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, patience=15, device=None):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # TensorBoard writer
#     writer = SummaryWriter("runs/landcover_experiment")

#     # Use BCE + Dice Loss
#     criterion = BCEDiceLoss(bce_weight=0.5)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     jaccard_metric = JaccardIndex(task='binary').to(device)

#     best_loss = float('inf')
#     early_stop_counter = 0
#     best_weights = copy.deepcopy(model.state_dict())

#     train_losses, val_losses = [], []
#     train_ious, val_ious = [], []

#     for epoch in range(num_epochs):
#         print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

#         # ---- Training ----
#         model.train()
#         running_loss, running_iou = 0.0, 0.0
#         for images, masks, *_ in tqdm(train_loader, desc="Train", leave=False):
#             images, masks = images.to(device), masks.to(device)

#             # Ensure float32
#             images = images.float()
#             masks = masks.float()

#             optimizer.zero_grad()
#             outputs = model(images)['out']
#             loss = criterion(outputs, masks)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             with torch.no_grad():
#                 iou = jaccard_metric(torch.sigmoid(outputs), masks.int())
#                 running_iou += iou.item()

#         epoch_loss = running_loss / len(train_loader)
#         epoch_iou = running_iou / len(train_loader)
#         train_losses.append(epoch_loss)
#         train_ious.append(epoch_iou)
#         print(f"Train Loss: {epoch_loss:.4f}, Train IoU: {epoch_iou:.4f}")

#         # ---- Validation ----
#         model.eval()
#         val_loss_sum, val_iou_sum = 0.0, 0.0
#         with torch.no_grad():
#             for images, masks, *_ in tqdm(val_loader, desc="Val", leave=False):
#                 images, masks = images.to(device), masks.to(device)
#                 images = images.float()
#                 masks = masks.float()

#                 outputs = model(images)['out']
#                 val_loss = criterion(outputs, masks)
#                 val_loss_sum += val_loss.item()

#                 iou = jaccard_metric(torch.sigmoid(outputs), masks.int())
#                 val_iou_sum += iou.item()

#         val_epoch_loss = val_loss_sum / len(val_loader)
#         val_epoch_iou = val_iou_sum / len(val_loader)
#         val_losses.append(val_epoch_loss)
#         val_ious.append(val_epoch_iou)
#         print(f"Val Loss: {val_epoch_loss:.4f}, Val IoU: {val_epoch_iou:.4f}")

#         # ---- Log to TensorBoard ----
#         writer.add_scalar("Loss/train", epoch_loss, epoch+1)
#         writer.add_scalar("Loss/val", val_epoch_loss, epoch+1)
#         writer.add_scalar("IoU/train", epoch_iou, epoch+1)
#         writer.add_scalar("IoU/val", val_epoch_iou, epoch+1)

#         # ---- Early stopping ----
#         if val_epoch_loss < best_loss:
#             best_loss = val_epoch_loss
#             best_weights = copy.deepcopy(model.state_dict())
#             torch.save(model.state_dict(), "best_modelnew1.pth")
#             early_stop_counter = 0
#         else:
#             early_stop_counter += 1

#         if early_stop_counter >= patience:
#             print(f"Early stopping triggered after {epoch+1} epochs.")
#             break

#     writer.close()  # close TB writer

#     model.load_state_dict(best_weights)
#     print("Training finished.")

#     # ---- Plot Loss ----
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label="Train Loss")
#     plt.plot(val_losses, label="Val Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Train vs Validation Loss")
#     plt.legend()
#     plt.show()

#     # ---- Plot IoU ----
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_ious, label="Train IoU")
#     plt.plot(val_ious, label="Val IoU")
#     plt.xlabel("Epoch")
#     plt.ylabel("Jaccard Index (IoU)")
#     plt.title("Train vs Validation IoU")
#     plt.legend()
#     plt.show()

#     return model, train_losses, val_losses, train_ious, val_ious


# def main():
#     data_path = r"P:\Drone-Image-Based-Landcover-Segmentation\output_merged"

#     transform = get_transform()

#     train_dataset = NepalDataset(data_path, subset='train', transform=transform, augment=True)
#     val_dataset = NepalDataset(data_path, subset='val', transform=transform, augment=False)

#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

#     model = prepare_model(num_classes=1)  # Binary segmentation

#     train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, patience=15)


# if __name__ == "__main__":
#     main()


# import os
# import copy
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from tqdm import tqdm

# from trygen import NepalDataset, get_transform
# from deemodel import prepare_model  # DeepLabV3 model
# from torchmetrics.classification import JaccardIndex
# from torch.utils.tensorboard import SummaryWriter


# # ---- Custom BCE + Dice Loss ----
# class BCEDiceLoss(nn.Module):
#     def __init__(self, bce_weight=0.5):
#         super(BCEDiceLoss, self).__init__()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.bce_weight = bce_weight

#     def forward(self, inputs, targets):
#         bce_loss = self.bce(inputs, targets)
#         inputs = torch.sigmoid(inputs)
#         smooth = 1.0
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         intersection = (inputs * targets).sum()
#         dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#         return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


# def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, patience=15, device=None):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     writer = SummaryWriter("runs/landcover_experiment")

#     criterion = BCEDiceLoss(bce_weight=0.5)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     jaccard_metric = JaccardIndex(task='binary').to(device)

#     best_loss = float('inf')
#     early_stop_counter = 0
#     best_weights = copy.deepcopy(model.state_dict())

#     for epoch in range(num_epochs):
#         print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

#         # ---- Training ----
#         model.train()
#         running_loss, running_iou = 0.0, 0.0
#         for images, masks, *_ in tqdm(train_loader, desc="Train", leave=False):
#             images, masks = images.to(device), masks.to(device)
#             images, masks = images.float(), masks.float()

#             optimizer.zero_grad()
#             outputs = model(images)['out']
#             loss = criterion(outputs, masks)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             with torch.no_grad():
#                 iou = jaccard_metric(torch.sigmoid(outputs), masks.int())
#                 running_iou += iou.item()

#         epoch_loss = running_loss / len(train_loader)
#         epoch_iou = running_iou / len(train_loader)
#         print(f"Train Loss: {epoch_loss:.4f}, Train IoU: {epoch_iou:.4f}")

#         # ---- Validation ----
#         model.eval()
#         val_loss_sum, val_iou_sum = 0.0, 0.0
#         with torch.no_grad():
#             for images, masks, *_ in tqdm(val_loader, desc="Val", leave=False):
#                 images, masks = images.to(device), masks.to(device)
#                 images, masks = images.float(), masks.float()

#                 outputs = model(images)['out']
#                 val_loss = criterion(outputs, masks)
#                 val_loss_sum += val_loss.item()

#                 iou = jaccard_metric(torch.sigmoid(outputs), masks.int())
#                 val_iou_sum += iou.item()

#         val_epoch_loss = val_loss_sum / len(val_loader)
#         val_epoch_iou = val_iou_sum / len(val_loader)
#         print(f"Val Loss: {val_epoch_loss:.4f}, Val IoU: {val_epoch_iou:.4f}")

#         # ---- TensorBoard logging ----
#         writer.add_scalar("Loss/train", epoch_loss, epoch + 1)
#         writer.add_scalar("Loss/val", val_epoch_loss, epoch + 1)
#         writer.add_scalar("IoU/train", epoch_iou, epoch + 1)
#         writer.add_scalar("IoU/val", val_epoch_iou, epoch + 1)

#         # ---- Early stopping ----
#         if val_epoch_loss < best_loss:
#             best_loss = val_epoch_loss
#             best_weights = copy.deepcopy(model.state_dict())
#             torch.save(model.state_dict(), "best_modelnew1.pth")
#             early_stop_counter = 0
#         else:
#             early_stop_counter += 1

#         if early_stop_counter >= patience:
#             print(f"Early stopping triggered after {epoch+1} epochs.")
#             break

#     writer.close()
#     model.load_state_dict(best_weights)
#     print("Training completed successfully.")


# def main():
#     data_path = r"P:\Drone-Image-Based-Landcover-Segmentation\output_merged"
#     transform = get_transform()

#     train_dataset = NepalDataset(data_path, subset='train', transform=transform, augment=True)
#     val_dataset = NepalDataset(data_path, subset='val', transform=transform, augment=False)

#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

#     model = prepare_model(num_classes=1)
#     train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, patience=15)


# if __name__ == "__main__":
#     main()

# import os
# import copy
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from tqdm import tqdm

# from trygen import NepalDataset, get_transform
# from deemodel import prepare_model  # DeepLabV3 model
# from torchmetrics.classification import JaccardIndex
# from torch.utils.tensorboard import SummaryWriter


# # ---- Custom BCE + Dice Loss ----
# class BCEDiceLoss(nn.Module):
#     def __init__(self, bce_weight=0.5):
#         super(BCEDiceLoss, self).__init__()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.bce_weight = bce_weight

#     def forward(self, inputs, targets):
#         bce_loss = self.bce(inputs, targets)
#         inputs = torch.sigmoid(inputs)
#         smooth = 1.0
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         intersection = (inputs * targets).sum()
#         dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
#         return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


# def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, patience=15, device=None):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     writer = SummaryWriter("runs/landcover_experiment")

#     criterion = BCEDiceLoss(bce_weight=0.5)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     jaccard_metric = JaccardIndex(task='binary').to(device)

#     best_loss = float('inf')
#     early_stop_counter = 0
#     best_weights = copy.deepcopy(model.state_dict())

#     for epoch in range(num_epochs):
#         print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

#         # ---- Training ----
#         model.train()
#         running_loss = 0.0
#         for images, masks, *_ in tqdm(train_loader, desc="Train", leave=False):
#             images, masks = images.to(device), masks.to(device)
#             images, masks = images.float(), masks.float()

#             optimizer.zero_grad()
#             outputs = model(images)['out']
#             loss = criterion(outputs, masks)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             with torch.no_grad():
#                 preds = torch.sigmoid(outputs)
#                 jaccard_metric.update(preds, masks.int())

#         epoch_loss = running_loss / len(train_loader)
#         epoch_iou = jaccard_metric.compute().item()
#         print(f"Train Loss: {epoch_loss:.4f}, Train IoU: {epoch_iou:.4f}")
#         jaccard_metric.reset()   # reset after training

#         # ---- Validation ----
#         model.eval()
#         val_loss_sum = 0.0
#         with torch.no_grad():
#             for images, masks, *_ in tqdm(val_loader, desc="Val", leave=False):
#                 images, masks = images.to(device), masks.to(device)
#                 images, masks = images.float(), masks.float()

#                 outputs = model(images)['out']
#                 val_loss = criterion(outputs, masks)
#                 val_loss_sum += val_loss.item()

#                 preds = torch.sigmoid(outputs)
#                 jaccard_metric.update(preds, masks.int())

#         val_epoch_loss = val_loss_sum / len(val_loader)
#         val_epoch_iou = jaccard_metric.compute().item()
#         print(f"Val Loss: {val_epoch_loss:.4f}, Val IoU: {val_epoch_iou:.4f}")
#         jaccard_metric.reset()   # reset after validation

#         # ---- TensorBoard logging ----
#         writer.add_scalar("Loss/train", epoch_loss, epoch + 1)
#         writer.add_scalar("Loss/val", val_epoch_loss, epoch + 1)
#         writer.add_scalar("IoU/train", epoch_iou, epoch + 1)
#         writer.add_scalar("IoU/val", val_epoch_iou, epoch + 1)

#         # ---- Early stopping ----
#         if val_epoch_loss < best_loss:
#             best_loss = val_epoch_loss
#             best_weights = copy.deepcopy(model.state_dict())
#             torch.save(model.state_dict(), "best_modelnew2.pth")
#             early_stop_counter = 0
#         else:
#             early_stop_counter += 1

#         if early_stop_counter >= patience:
#             print(f"Early stopping triggered after {epoch+1} epochs.")
#             break

#     writer.close()
#     model.load_state_dict(best_weights)
#     print("Training completed successfully.")


# def main():
#     data_path = r"P:\Drone-Image-Based-Landcover-Segmentation\output_merged"
#     transform = get_transform()

#     train_dataset = NepalDataset(data_path, subset='train', transform=transform, augment=True)
#     val_dataset = NepalDataset(data_path, subset='val', transform=transform, augment=False)

#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

#     model = prepare_model(num_classes=1)
#     train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, patience=15)


# if __name__ == "__main__":
#     main()


import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from trygen import NepalDataset, get_transform
from deemodel import prepare_model  # DeepLabV3 model
from torchmetrics.classification import JaccardIndex
from torch.utils.tensorboard import SummaryWriter


# ---- Custom BCE + Dice Loss ----
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)
        smooth = 1.0
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, patience=15, device=None, checkpoint_path="best_checkpoint.pth"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    writer = SummaryWriter("runs/landcover_experiment")

    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    jaccard_metric = JaccardIndex(task='binary').to(device)

    best_loss = float('inf')
    early_stop_counter = 0
    best_weights = copy.deepcopy(model.state_dict())

    start_epoch = 0

    # Optionally, load checkpoint if exists to resume training
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}' to resume training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch} with best val loss {best_loss:.4f}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        # ---- Training ----
        model.train()
        running_loss = 0.0
        for images, masks, *_ in tqdm(train_loader, desc="Train", leave=False):
            images, masks = images.to(device), masks.to(device)
            images, masks = images.float(), masks.float()

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                jaccard_metric.update(preds, masks.int())

        epoch_loss = running_loss / len(train_loader)
        epoch_iou = jaccard_metric.compute().item()
        print(f"Train Loss: {epoch_loss:.4f}, Train IoU: {epoch_iou:.4f}")
        jaccard_metric.reset()   # reset after training

        # ---- Validation ----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, masks, *_ in tqdm(val_loader, desc="Val", leave=False):
                images, masks = images.to(device), masks.to(device)
                images, masks = images.float(), masks.float()

                outputs = model(images)['out']
                val_loss = criterion(outputs, masks)
                val_loss_sum += val_loss.item()

                preds = torch.sigmoid(outputs)
                jaccard_metric.update(preds, masks.int())

        val_epoch_loss = val_loss_sum / len(val_loader)
        val_epoch_iou = jaccard_metric.compute().item()
        print(f"Val Loss: {val_epoch_loss:.4f}, Val IoU: {val_epoch_iou:.4f}")
        jaccard_metric.reset()   # reset after validation

        # ---- TensorBoard logging ----
        writer.add_scalar("Loss/train", epoch_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_epoch_loss, epoch + 1)
        writer.add_scalar("IoU/train", epoch_iou, epoch + 1)
        writer.add_scalar("IoU/val", val_epoch_iou, epoch + 1)

        # ---- Early stopping and checkpoint saving ----
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_weights = copy.deepcopy(model.state_dict())

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_epoch_loss,
                'best_loss': best_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1} with val loss: {val_epoch_loss:.4f}")

            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    writer.close()
    model.load_state_dict(best_weights)
    print("Training completed successfully.")


def main():
    data_path = r"P:\Drone-Image-Based-Landcover-Segmentation\output_merged"
    transform = get_transform()

    train_dataset = NepalDataset(data_path, subset='train', transform=transform, augment=True)
    val_dataset = NepalDataset(data_path, subset='val', transform=transform, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,drop_last=True)

    model = prepare_model(num_classes=1)
    train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4, patience=15)


if __name__ == "__main__":
    main()
