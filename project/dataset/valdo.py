import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import cv2

class VALDODataset(Dataset):
    def __init__(self, cases, masks, transform, normalization=None):
        self.cases = cases
        self.masks = masks
        self.transform = transform
        self.cmb_counts = self.count_cmb_per_image(self.masks)
        self.normalization = normalization

        assert len(self.cases) == len(
            self.masks), "Cases and masks must have the same length"

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        try:
            case = self.cases[idx]
            mask = self.masks[idx]
        
            slices, masks = self.transform(mri_image_path=case, segmentation_mask_path=mask)
            if slices is None or masks is None:
                raise ValueError(f"Transform returned None for {case} and {mask}")
            
            return slices, masks, case, self.cmb_counts[idx]
        
        except Exception as e:
            print(f'Error loading image: {e}')
            return None, None, None, None

    def extract_bounding_boxes(self, mask):
        # Extract bounding boxes from mask
        boxes = []
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x + w, y + h])
        return boxes

    def count_cmb_per_image(self, segmented_images):
        cmb_counts = []
        for img_path in segmented_images:
            img = nib.load(img_path)
            data = img.get_fdata()
            slice_cmb_counts = [self.extract_bounding_boxes(
                (data[:, :, i] > 0).astype(np.uint8)) for i in range(data.shape[2])]
            total_cmb_count = sum(len(contours)
                                  for contours in slice_cmb_counts)
            cmb_counts.append(total_cmb_count)
        return cmb_counts

    def locate_case_by_mri(self, case_name): # Find a specific case using name search
        # Enumerate to all masks name
        for idx, case in enumerate(self.cases):
            if case_name in case:
                # Return case details
                return self.__getitem__(idx)
        # If no match found
        return None

    def locate_case_by_name(self, case_name): # Find a specific case using name search
        # Enumerate to all masks name
        for idx, case in enumerate(self.masks):
            if case_name in case:
                # Return case details
                return self.__getitem__(idx)
        # If no match found
        return None
