import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from project.preprocessing import min_max_normalization, z_score_normalization

import cv2

class MRIDataset(Dataset):
    def __init__ (self, cases, masks, transform=None, target_size=(256, 256), normalize=False, rpn_mode=False):
        self.cases = cases
        self.masks = masks
        self.transform = transform
        self.target_size = target_size
        self.normalize_type = normalize
        self.resize = transforms.Resize(self.target_size)
        self.rpn_mode = rpn_mode
        
        
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case_path = self.cases[idx]
        mask_path = self.masks[idx]
        
        case_img = nib.load(case_path).get_fdata()
        mask_img = nib.load(mask_path).get_fdata()
        
        print(f"MRI DATASET - Raw MRI scan shape: {case_img.shape}")
        print(f"MRI DATASET - Raw mask shape: {mask_img.shape}")
        
        case_img = self.preprocess(case_img)
        mask_img = self.preprocess(mask_img)
        
        print(f"MRI DATASET - Preprocessed MRI scan shape: {case_img.shape}")
        print(f"MRI DATASET - Preprocessed mask shape: {mask_img.shape}")
        
        case_img = np.transpose(case_img, (2, 0, 1))  # Shape: (num_slices, height, width)
        mask_img = np.transpose(mask_img, (2, 0, 1)) 
        
        case_img = torch.from_numpy(case_img).unsqueeze(1)  # Add channel dimension at axis 1
        mask_img = torch.from_numpy(mask_img).unsqueeze(1)

        
        print(f"MRI DATASET - Preprocessed MRI scan shape (with channel): {case_img.shape}")
        print(f"MRI DATASET - Preprocessed mask shape (with channel): {mask_img.shape}")
        
        if self.rpn_mode:
            case_img, mask_img = self.apply_rpn_transform(case_img, mask_img)

        
        if self.transform:
            case_img = self.transform(case_img)
            mask_img = self.transform(mask_img)
            
            print(f"MRI DATASET - Transformed MRI scan shape: {case_img.shape}")
            print(f"MRI DATASET - Transformed mask shape: {mask_img.shape}")
            
        print(f"MRI DATASET - Final MRI scan shape: {case_img.shape}")
        print(f"MRI DATASET - Final mask shape: {mask_img.shape}")
        
        cmb_counts = np.count_nonzero(mask_img.numpy())
            
        return case_img, mask_img, case_path, cmb_counts
    
    def preprocess(self, img):
        img_resized = self.resize_image(img, self.target_size)
        img_preprocessed = self.normalize(img_resized)
        
        return img_preprocessed
    
    def normalize(self, img):
        if self.normalize_type == 'minmax':
            return min_max_normalization(img)
        elif self.normalize_type == 'zscore':
            return z_score_normalization(img)
        else:
            return img
    
    def resize_image(self, img, target_size):
        current_size = img.shape
        zoom_factors = [target_size[0] / current_size[0], target_size[1] / current_size[1], 1]
        img_resized = zoom(img, zoom_factors, order=1)  
        
        return img_resized
    
    def apply_rpn_transform(self, case_img, mask_img):
        image_slices = []
        box_slices = []

        num_slices = case_img.shape[0]

        for i in range(num_slices):
            mask_slice = mask_img[i, 0, :, :] 
            boxes = self.extract_bounding_boxes(mask_slice.numpy())  
            img_slice = case_img[i, 0, :, :]  

            if len(boxes) == 0:
                boxes = [[-1, -1, -1, -1]]  

            img_tensor = img_slice.unsqueeze(0).float()
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

            if boxes_tensor.shape[0] == 1:
                image_slices.append(img_tensor.unsqueeze(0))
                box_slices.append(boxes_tensor.unsqueeze(0))
            else:
                for box in boxes_tensor:
                    image_slices.append(img_tensor.unsqueeze(0)) 
                    box_slices.append(box.unsqueeze(0).unsqueeze(0))

        image = torch.stack(image_slices)
        boxes = torch.stack(box_slices)
        
        print(f"MRI DATASET - RPN Transformed MRI scan shape: {image.shape}")
        print(f"MRI DATASET - RPN Transformed mask shape: {boxes.shape}")

        return image, boxes
    
    def extract_bounding_boxes(self, mask):
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        boxes = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x + w, y + h])
        return boxes
