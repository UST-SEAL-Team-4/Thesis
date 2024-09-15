import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    
    def get_cropped_locations(self, img, x_min, y_min, x_max, y_max):
        """
        Display the specified location across all slices of the MRI.

        :param idx: Index of the MRI scan in the dataset.
        :param x_min: Minimum x coordinate of the location.
        :param y_min: Minimum y coordinate of the location.
        :param x_max: Maximum x coordinate of the location.
        :param y_max: Maximum y coordinate of the location.
        """
        try:
            # img_path = self.img_paths[idx]

            # # Load 3D image
            # img = nib.load(img_path).get_fdata()
            # # img = (img / np.max(img) * 255).astype(np.uint8)
            # img, targets, img_path, cmb_count = self.__getitem__(idx=idx)
            cropped_slices = []

            # Iterate through each slice and crop to the specified region
            # print(img.shape)
            if img.dim() == 5:
                # If 5D, assume shape is [num_slices, channels, height, width]
                for i in range(img.__len__()):
                    img_slice = img[i]
                    cropped_slice = img_slice[0, 0, y_min:y_max, x_min:x_max]
                    cropped_slices.append(torch.Tensor(cropped_slice))
                
                # # for i in range(img.__len__()):
                # #     returned.append(cropped_slices)
                # combined_slices = np.hstack(cropped_slices)

                # current_width = combined_slices.shape[1]
                # pad_width = max_width - current_width
                # padded_image = np.pad(combined_slices, ((0, 0), (0, pad_width)), mode='constant')

                # plt.imshow(padded_image, cmap='gray')
                # plt.title(f'Slice {i}')
                # plt.show()
                return torch.stack(cropped_slices, dim=0).unsqueeze(0).unsqueeze(0)

            elif img.dim() == 4:
                # If 4D, assume shape is [num_slices, height, width]
                for i in range(img.shape[0]):
                    img_slice = img[i, 0]  # Extract the 2D slice (assuming single channel)
                    cropped_slice = img_slice[y_min:y_max, x_min:x_max]
                    cropped_slices.append(torch.Tensor(cropped_slice))

                # # for i in range(img.__len__()):
                # #     returned.append(cropped_slices)
                # combined_slices = np.hstack(cropped_slices)
                # current_width = combined_slices.shape[1]
                # pad_width = max_width - current_width
                # padded_image = np.pad(combined_slices, ((0, 0), (0, pad_width)), mode='constant')

                # plt.imshow(padded_image, cmap='gray')
                # plt.title(f'Slice {i}')
                # plt.show()
                return torch.stack(cropped_slices, dim=0).unsqueeze(0)

            else:
                raise ValueError("Unsupported tensor dimension. Expected 3D or 4D tensor.")


        except Exception as e:
            print(f"Error processing: {e}")
            raise
    