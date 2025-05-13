import os
import torch
import csv
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
import sys
sys.path.append(".")
from UNet import UNet  # Assuming your model architecture is defined in a file called UNet.py
#this Unet file must be in the same path than this script.py
# Print the current working directory to verify it's correct
print("Current working directory:", os.getcwd())
# Ensure dataloader is in the same directory and import it
import dataloader
from dataloader import CornFieldsSegmentationDataset # Correct import

# Print dataloader to verify it is correctly loaded
print(dataloader)

# Define transformation for preprocessing
transform = transforms.Compose([
    transforms.ToTensor()
])

# Define dataset directories (adjust for your actual directories)
image_dir = "D:/data_02/mpg23_classes/thesis/ciat_guate/unet_resources/attempt_05/toto/val1/val_dataset"
mask_dir = "D:/data_02/mpg23_classes/thesis/ciat_guate/unet_resources/attempt_05/toto/val1/mask"
csv_file_path = "D:/data_02/mpg23_classes/thesis/ciat_guate/unet_resources/attempt_05/toto/val1/val1_report03.csv"

# Create datasets for images and masks
dataset = CornFieldsSegmentationDataset(image_dir,mask_dir, transform=transform)
#mask_dataset = SegmentationMaskDataset(mask_dir, transform=transform)
batch_size = 16
data_loader = DataLoader(dataset, batch_size, shuffle=False)
# Initialize U-Net model
model = UNet(in_channels = 4,  # 4 bands (NIRGB + NDVI)
out_channels = 1)  # Binary segmentation (crop fields or background)
# Load trained model
model.load_state_dict(torch.load("D:/data_02/mpg23_classes/thesis/ciat_guate/unet_resources/attempt_05/toto/val1/crops_segmentation_unet_train_f1_att5.pth",weights_only=True))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Metrics initialization
all_preds = []
all_labels = []

# val1 loop
criterion = nn.BCELoss()
total_loss = 0.0
num_batches = len(data_loader)

with torch.no_grad():
    for inputs, masks, files in data_loader:
        inputs = inputs.to(device)
        masks = masks.to(device) # Remove singleton channel dimension
        # Debug: Print original shapes
        print(f"Original inputs shape: {inputs.shape}")
        print(f"Original masks shape: {masks.shape}")
              
        # Get predictions from the model
        outputs = model(inputs)
        print(f"Raw model output shape: {outputs.shape}")

        # Ensure outputs and masks have the same size
        masks = masks.squeeze(3).squeeze(1).unsqueeze(1)
        # Debug: Print reshaped masks
        print(f"Reshaped masks shape: {masks.shape}")
        print(f"Resized outputs shape: {outputs.shape}, Masks shape: {masks.shape}, Type of Masks: {type(masks)}")
        
        # Ensure shapes match
        assert outputs.shape == masks.shape, \
            f"Shape mismatch: Resized output {outputs.shape}, Masks {masks.shape}"
        
        # Apply sigmoid activation to outputs before computing loss
        outputs_sigmoid = torch.sigmoid(outputs)  # Ensure outputs are in the range [0, 1]

        # Calculate loss
        loss = criterion(outputs_sigmoid, masks)
        total_loss += loss.item()
        print(f"Outputs shape: {outputs.shape}, Masks shape: {masks.shape}")

        # Convert predictions and masks to numpy arrays for metrics calculation
        preds = outputs_sigmoid.cpu().numpy()  # (B, H, W) or (Batch, Height, Width)
        preds_flat = (preds > 0.5).astype(np.uint8).flatten()
        masks_flat = masks.cpu().numpy().flatten()

        all_preds.extend(preds_flat)
        all_labels.extend(masks_flat)

# Calculate average loss
average_loss = total_loss / num_batches
print(f'val1 Loss: {average_loss:.4f}')

# Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
kappa = cohen_kappa_score(all_labels, all_preds)

# Compute confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)


tp = conf_matrix[1, 1]
fp = conf_matrix[0, 1]
fn = conf_matrix[1, 0]
tn = conf_matrix[0, 0]

re_conf_matrix = [[tp, fp],
                  [fn, tn]]

# Print the metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Cohen Kappa Score: {kappa:.4f}')

with open(csv_file_path, "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write header
    csvwriter.writerow(["Metric", "Value"])
    # Write metrics
    csvwriter.writerow(["Accuracy", accuracy])
    csvwriter.writerow(["Precision", precision])
    csvwriter.writerow(["Recall", recall])
    csvwriter.writerow(["F1 Score", f1])
    csvwriter.writerow(["Kappa coef.", kappa])
    # Save confusion matrix as rows
    csvwriter.writerow([])
    csvwriter.writerow(["Confusion Matrix"])
    csvwriter.writerow(["", "Predicted positive", "Predicted negative"])  # Column headers
    csvwriter.writerow(["Actual positive"] + re_conf_matrix[0])  # Row for Actual 0
    csvwriter.writerow(["Actual negative"] + re_conf_matrix[1])  # Row for Actual 1
    #csvwriter.writerow([])
    csvwriter.writerow(["","True Positive", "False Positive"])
    csvwriter.writerow(["","False Negative", "True Negative"])


print(f"val1 metrics saved to {csv_file_path}")