import os
import cv2
import numpy as np
import torch
import tifffile as tiff
import torchvision.transforms as transforms
import sys
sys.path.append(".")
from UNet import UNet  # Assuming your model architecture is defined in a file called UNet.py
#this Unet file must be in the same path than this script.py

# Define dataset directories
test_data_dir = "D:/data_02/mpg23_classes/thesis/ciat_guate/unet_resources/attempt_05/toto/test1/tiff_images"

# Create a transform to preprocess the images
transform = transforms.Compose([transforms.ToTensor()])

# Initialize U-Net model
model = UNet(in_channels=4, out_channels=1)  # Modify based on your model architecture
model.load_state_dict(torch.load("D:/data_02/mpg23_classes/thesis/ciat_guate/unet_resources/attempt_05/toto/test1/crops_segmentation_unet_train_f1_att5.pth", weights_only=True))  # Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Create a list to store predicted masks
predicted_masks = []

# Debug: List all files in the test_data_dir
print("Files in test_data_dir:", os.listdir(test_data_dir))

# Iterate over the images in the test dataset
for img_name in os.listdir(test_data_dir):
    img_path = os.path.join(test_data_dir, img_name)
     # Only process TIFF files
    if img_name.lower().endswith('.tif'):
        print(f"Processing {img_name}...")  # Log the current image being processed

        try:
            # Load the TIFF image (with 4 bands: NIR, Green, Blue, NDVI)
            image = tiff.imread(img_path).astype(np.float32)
        
            # Normalize the bands: We assume the NDVI values are between -1 and 1, normalize to [0, 1]
            image[:, :, :3] /= 255.0  # Normalize RGB bands (0 to 255 range)
            image[:, :, 3] = (image[:, :, 3] + 1) / 2  # Normalize NDVI (-1 to 1 range -> 0 to 1 range)
        
            # Convert the image to a tensor with shape (C, H, W) where C = 4
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device) # Shape (1, 4, H, W)

            with torch.no_grad():
                # Pass the image tensor through the model
                output = model(image_tensor)

                # Apply sigmoid to get probabilities, then threshold to get the binary mask
                predicted_mask = torch.sigmoid(output).squeeze(0).cpu().numpy() # Convert from tensor to numpy
            
                # Thresholding to create binary mask
                binary_mask = (predicted_mask > 0.5).astype(np.uint8)
            
                # Append the binary mask to the list
                predicted_masks.append((img_name, binary_mask))
            
        except Exception as e:
            print(f"Failed to process {img_name}: {e}")
     # Load the TIFF image (with 4 bands: NIR, Green, Blue, NDVI)
    #image = tiff.imread(img_path).astype(np.float32)

# Check if predicted_masks is populated
if not predicted_masks:
    print("No masks were generated. Check if all files are properly processed.")

else:
# Create a directory to save the predicted masks
    output_dir = "D:/data_02/mpg23_classes/thesis/ciat_guate/unet_resources/attempt_05/toto/test1/predicted_mask"
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the predicted masks and save each mask as an image file
    for image_name, mask in predicted_masks:

        # Extract the base name (without extension)
        base_name = os.path.splitext(image_name)[0]

        # Construct the file name for the mask (e.g., image_1_pred_mask.png, image_2_pred_mask.png, etc.)
        mask_file = os.path.join(output_dir, f'{base_name}_pred_mask.png')
        
        # Threshold the predicted mask to generate binary mask
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        # Remove the singleton dimension
        #normalized_mask = normalized_mask.squeeze() this work
        
        if len(binary_mask.shape) == 3:
            # In case the mask has a channel dimension, we remove it to make it 2D
            binary_mask = binary_mask.squeeze()
        # Save the mask as an image file using OpenCV
        cv2.imwrite(mask_file, binary_mask)
        print(f"Saved mask for {img_name} as {mask_file}")  # Log the saved mask file

    print(f"Predicted masks saved to {output_dir}")