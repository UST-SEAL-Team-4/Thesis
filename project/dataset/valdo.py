import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import cv2
from project.preprocessing import z_score_normalization, min_max_normalization

class VALDODataset(Dataset):
    def __init__(self, img_paths, ann_paths, transform=None, normalization=None):
        self.img_paths = img_paths
        self.ann_paths = ann_paths
        self.transform = transform
        self.cmb_counts = self.count_cmb_per_image(self.ann_paths)
        self.normalization = normalization

        assert len(self.img_paths) == len(
            self.ann_paths), "Mismatch between number of images and annotations"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.img_paths[idx]
            ann_path = self.ann_paths[idx]
            cmb_count = self.cmb_counts[idx]

            # Load 3D image
            img = nib.load(img_path).get_fdata()
            
            if self.normalization == 'z_score':
                img = z_score_normalization(img)
            elif self.normalization == 'min_max':
                img = min_max_normalization(img)
                
            img = (img * 255).astype(np.uint8)

            # Load 3D annotation
            ann = nib.load(ann_path).get_fdata()
            ann = (ann > 0).astype(np.uint8)  # Ensure mask is binary

            slices = []
            targets = []

            for i in range(img.shape[2]):
                img_slice = img[:, :, i]
                ann_slice = ann[:, :, i]

                # Convert single-channel to three-channel
                img_slice = cv2.merge([img_slice] * 3)
                boxes = self.extract_bounding_boxes(ann_slice)

                if boxes:
                    augmented = self.transform(
                        image=img_slice, bboxes=boxes, labels=[1]*len(boxes))
                    img_slice = augmented['image']
                    boxes = augmented['bboxes']
                    labels = augmented['labels']
                else:
                    augmented = self.transform(
                        image=img_slice, bboxes=[], labels=[])
                    img_slice = augmented['image']
                    boxes = augmented['bboxes']
                    labels = augmented['labels']

                target = {
                    'boxes': torch.tensor(boxes, dtype=torch.float32),
                    'labels': torch.tensor(labels, dtype=torch.int64)
                }

                slices.append(img_slice)
                targets.append(target)

            return slices, targets, img_path, cmb_count

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            raise

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