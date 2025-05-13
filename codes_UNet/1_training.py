import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import csv
import tifffile as tiff
import sys
sys.path.append(".")
from UNet import UNet

# Hyperparameters
lr = 0.001  # Learning rate
batch_size = 16  # Batch size
num_epochs = 100  # Number of epochs

# Dataset class
class CropsSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
        self.mask_filenames = os.listdir(mask_dir)  # Load all mask filenames
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get image file path
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)

        # Extract base name from the TIFF file (without extension)
        base_name = os.path.splitext(image_filename)[0]
        # Find the corresponding mask file
        mask_filename = next(
            (f for f in self.mask_filenames if base_name in f and f.endswith("_mask.png")),
            None
        )
        if mask_filename is None:
            raise FileNotFoundError(f"No matching mask found for image {image_filename}")

        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Load image and mask
        image = tiff.imread(image_path).astype(np.float32)
        image[:, :, :3] /= 255.0
        image[:, :, 3] = (image[:, :, 3] + 1) / 2  # Normalize NDVI
        image = torch.from_numpy(image).permute(2, 0, 1)  # Convert to (C, H, W)

        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask, dtype=np.float32) / 255.0  # Normalize mask
        mask = (mask > 0).astype(np.float32)  # Ensure binary
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image, mask, image_filename


# Paths
image_dir = "D:/data_02/mpg23_classes/thesis/ciat_guate/unet_resources/attempt_05/toto/train1/tiff_images"
mask_dir = "D:/data_02/mpg23_classes/thesis/ciat_guate/unet_resources/attempt_05/toto/train1/mask"
output_dir = "D:/data_02/mpg23_classes/thesis/ciat_guate/unet_resources/attempt_05/toto/train1"
os.makedirs(output_dir, exist_ok=True)

# Transformations
transform = transforms.Compose([
    #transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = CropsSegmentationDataset(image_dir, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# U-Net model
in_channels = 4
out_channels = 1
model = UNet(in_channels, out_channels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Metrics calculation
def calculate_metrics(predictions, targets, epsilon=1e-6):
    predictions = (predictions > 0.5).float()
    targets = targets.float()

    intersection = torch.logical_and(predictions, targets).sum().float()
    union = torch.logical_or(predictions, targets).sum().float()
    iou = intersection / (union + epsilon)

    dice = (2 * intersection) / (predictions.sum() + targets.sum() + epsilon)

    accuracy = (predictions == targets).sum().float() / torch.numel(targets)

    return iou.item(), dice.item(), accuracy.item()


# Logging setup
log_path = os.path.join(output_dir, "training_f1_att5.csv")
with open(log_path, "w", newline="") as log_file:
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Learning Rate: " + str(lr)])
    csv_writer.writerow(["Batch Size: " + str(batch_size)])
    #csv_writer.writerow("Epoch/tLoss/tAccuracy/tIoU/tDice/n")
    csv_writer.writerow(["Epoch", "Loss", "Accuracy", "IoU", "Dice"])

# Training loop
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    total_iou, total_dice, total_accuracy = 0.0, 0.0, 0.0

    for inputs, masks, filenames in dataloader:
        inputs, masks = inputs.to(device), masks.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        predictions = torch.sigmoid(outputs)
        iou, dice, accuracy = calculate_metrics(predictions, masks)
        total_iou += iou
        total_dice += dice
        total_accuracy += accuracy

    avg_loss = epoch_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)

    # Log results in row-column format
    with open(log_path, "a", newline="") as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow([epoch + 1, avg_loss, avg_accuracy, avg_iou, avg_dice])

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")

# Save trained model
model_path = os.path.join(output_dir, "crops_segmentation_unet_train_f1_att5.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
