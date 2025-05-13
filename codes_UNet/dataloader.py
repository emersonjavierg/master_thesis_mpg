import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os

class CornFieldsSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

        print(f"Found {len(self.image_filenames)} images in {image_dir}")
        print(f"Found {len(self.mask_filenames)} masks in {mask_dir}")

        # Ensure the number of images and masks are the same
        assert len(self.image_filenames) == len(self.mask_filenames), \
            f"Mismatch in number of images and masks: {len(self.image_filenames)} images, {len(self.mask_filenames)} masks"

        # Check filenames for direct correspondence
        for img_file, mask_file in zip(self.image_filenames, self.mask_filenames):
            img_name_without_ext = os.path.splitext(img_file)[0]
            mask_name_without_ext = os.path.splitext(mask_file)[0]
            mask_name_without_suffix = mask_name_without_ext.replace('_mask', '')

            assert img_name_without_ext == mask_name_without_suffix, \
                f"Image and mask filenames do not match: {img_name_without_ext} != {mask_name_without_suffix}"
            
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])
        
        # Read the 4-band TIFF image
        image = tiff.imread(image_path).astype(np.float32)  # Assuming 4 bands (e.g., RGB and NDVI)
        # Normalize the image
        image[:, :, :3] /= 255.0  # Normalize bands to [0, 1]
        image[:, :, 3] = (image[:, :, 3] + 1) / 2  # Normalize NDVI to [0, 1]
        # Convert to Tensor format (C, H, W) where C = 4, H = height, W = width
        image = torch.from_numpy(image).permute(2, 0, 1)  # Shape: (4, H, W)

        # Load the mask as a binary image (assuming binary segmentation)
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        #mask = mask.squeeze(0)  # Remove singleton dimension for BCELoss compatibility
        #mask = transforms.Resize((512, 512))(mask)  # Match output size
        mask = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)  # Binary tensor
        mask = np.array(mask, dtype=np.float32) / 255.0
        #mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)  # Add channel dimension
        #mask = mask.point(lambda p: 255 if p > 0 else 0)  # Binary thresholding
        #mask = mask.convert("1")  # Ensure it is a binary mask
        
        # If transform is provided, apply to both image and mask
        if self.transform:
            #image = self.transform(image)
            mask = self.transform(mask)

        # Convert the mask to a binary tensor
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)  # Add channel dimension

        return image, mask, self.image_filenames[idx]
    
class SegmentationMaskDataset(Dataset):
     def __init__(self, mask_dir, transform=None):
         self.mask_dir = mask_dir
         self.mask_paths = [os.path.join(mask_dir, mask) for mask in os.listdir(mask_dir) if mask.endswith('.png')] #List all mask file paths in the directory.
         self.transform = transform

     def __len__(self): # Return the number of masks in the dataset.
         return len(self.mask_paths)

     def __getitem__(self, idx): # Get a mask by its index.
         mask_path = self.mask_paths[idx]
         mask = Image.open(mask_path).convert('L')  # Open and convert the mask to grayscale.
         if self.transform:
             mask = self.transform(mask) # Apply transformations if provided.
         return mask


