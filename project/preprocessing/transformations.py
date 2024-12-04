import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import torch
import nibabel as nib
import cv2
import torch
import torch.nn as nn

class PatchTruther(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        assert patch_size == int(patch_size), 'Patch size is not a perfect square'
        self.i2p = nn.Unfold(kernel_size = int(patch_size), stride = int(patch_size))

    def forward(self, img):
        '''
        Find section with the highest confidence_score and return as patch
        '''

        assert len(img.shape) == 3, 'Image must be of dim 3'

        patches = self.i2p(img)
        patches = patches.permute(1, 0)
        out = torch.any(patches, dim=-1)
        return out.unsqueeze(0)

def get_transform(height, width, p, rpn_mode=False):
    if rpn_mode is False:
        return A.Compose(
            [
                A.Resize(
                    height=height, 
                    width=width, 
                    p=p, 
                    always_apply=True
                ),
                ToTensorV2(p=p)
            ],
            is_check_shapes=False
        )
    else:
        return A.Compose(
            [
                A.Resize(
                    height=height, 
                    width=width, 
                    p=p, 
                    always_apply=True
                ),
                ToTensorV2(p=p)
            ],
            p=p,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                min_area=0,
                min_visibility=0,
                label_fields=['labels']
            )
        )

class NiftiToTensorTransform:
    def __init__(self, target_shape=(512,512), patch_size=2, in_channels=1, rpn_mode=False, normalization=None):
        self.target_shape = target_shape
        self.in_channels = in_channels
        self.transform = get_transform(
            height=target_shape[0],
            width=target_shape[1],
            p=1.0,
            rpn_mode=rpn_mode
        )
        self.rpn_mode = rpn_mode
        self.normalization = normalization
        self.i2p = PatchTruther(patch_size)
        
    def convert_to_binary_mask(self, segmentation_mask):
        binary_mask = (segmentation_mask > 0).astype(np.uint8)
        return binary_mask

    def extract_bounding_boxes(self, mask):
        # Extract bounding boxes from mask
        boxes = []
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x-50, y-50, x + w+50, y + h+50])
        return boxes
    
    def normalize_slice(self, slice):
        # mean, std = self.normalization
        # slice = (slice - mean) / std
        min, max = self.normalization
        slice = (slice - min) / (max - min)
        
        return slice

    def __call__(self, mri_image_path, segmentation_mask_path):
        try:
            mri_image = nib.load(mri_image_path).get_fdata()
            segmentation_mask = nib.load(segmentation_mask_path).get_fdata()
            
            if self.normalization is not None:
                mri_image = np.stack([self.normalize_slice(mri_image[:, :, i]) for i in range(mri_image.shape[2])], axis=-1)
                
            segmentation_mask = self.convert_to_binary_mask(segmentation_mask) # USE THIS
            
            image_slices = []
            mask_slices = []
            anno_slices = []
            
            if self.rpn_mode == False:
                for i in range(mri_image.shape[2]):
                    augmented = self.transform(
                        image=mri_image[:, :, i], 
                        mask=segmentation_mask[:, :, i]
                    )
                
                    image_slices.append(augmented['image'].unsqueeze(0)) 
                    mask_slices.append(augmented['mask'].unsqueeze(0))   
    
                image = torch.stack(image_slices) 
                mask = torch.stack(mask_slices)  
            
                if image.shape[1] != 1 or mask.shape[1] != 1:
                    raise ValueError("Unexpected number of slices in the MRI image or segmentation mask.")

                return image, mask

            else: # RPN transformation
                for i in range(mri_image.shape[2]):
                    boxes = self.extract_bounding_boxes(segmentation_mask[:, :, i])
                    if boxes:
                        augmented = self.transform(
                            image=mri_image[:, :, i],
                            mask=segmentation_mask[:, :, i],
                            bboxes=boxes,
                            labels=[1]*len(boxes)
                        )
                        img_slice = augmented['image']
                        anno_slice = augmented['mask']
                        boxes = torch.tensor(augmented['bboxes'])
                        labels = augmented['labels']
                    else:
                        augmented = self.transform(
                            image=mri_image[:, :, i],
                            mask=segmentation_mask[:, :, i],
                            bboxes=[],
                            labels=[]
                        )
                        img_slice = augmented['image']
                        anno_slice = augmented['mask']
                        boxes = torch.tensor([0] * 4, dtype=torch.float32).unsqueeze(0)
                        labels = augmented['labels']

                    if boxes.shape[0] == 1:
                        image_slices.append(img_slice.unsqueeze(0))
                        boxes = torch.clamp(boxes, min=0, max=self.target_shape[0])
                        mask_slices.append(boxes.unsqueeze(0))
                        anno_slices.append(anno_slice.unsqueeze(0))
                    else: # if there are more than one bbox coordinates for a slice
                        # print('MULTIPLE BOXES FOUND')
                        # print(boxes)
                        image_slices.append(img_slice.unsqueeze(0))
                        anno_slices.append(anno_slice.unsqueeze(0))

                        found_boxes = []
                        for bbox in boxes:
                            found_boxes.append(bbox)

                        found_boxes = torch.stack(found_boxes)
                        mask_slices.append(found_boxes)

                image = torch.stack(image_slices) 
                mask = mask_slices
                assert image.shape[0] == len(mask)
                annot = torch.stack(anno_slices).float()

                truthsarr = []
                for i in annot:
                    truthsarr.append(self.i2p(i))
                truths = torch.stack(truthsarr)

                assert image.shape[1] == 1, "Unexpected number of slices in the MRI image or segmentation mask."
                # if image.shape[1] != 1 or mask.shape[1] != 1:
                #     raise ValueError("Unexpected number of slices in the MRI image or segmentation mask.")

                return image, mask
        
        except Exception as e:
            print(f"Error in __call__ with {mri_image} and {segmentation_mask}: {e}")
            return None, None

class NEWNiftiToTensorTransform:
    def __init__(self, target_shape=(512,512), patch_size=2, in_channels=1, rpn_mode=False, normalization=None):
        self.target_shape = target_shape
        self.in_channels = in_channels
        self.transform = get_transform(
            height=target_shape[0],
            width=target_shape[1],
            p=1.0,
        )
        self.rpn_mode = rpn_mode
        self.normalization = normalization
        self.i2p = PatchTruther(patch_size)
        
    def convert_to_binary_mask(self, segmentation_mask):
        binary_mask = (segmentation_mask > 0).astype(np.uint8)
        return binary_mask

    def extract_bounding_boxes(self, mask):
        # Extract bounding boxes from mask
        boxes = []
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x-50, y-50, x + w+50, y + h+50])
        return boxes
    
    def normalize_slice(self, slice):
        # mean, std = self.normalization
        # slice = (slice - mean) / std
        min, max = self.normalization
        slice = (slice - min) / (max - min)
        
        return slice

    def __call__(self, mri_image_path, segmentation_mask_path):
        try:
            mri_image = nib.load(mri_image_path).get_fdata()
            segmentation_mask = nib.load(segmentation_mask_path).get_fdata()
            
            if self.normalization is not None:
                mri_image = np.stack([self.normalize_slice(mri_image[:, :, i]) for i in range(mri_image.shape[2])], axis=-1)
                
            segmentation_mask = self.convert_to_binary_mask(segmentation_mask) # USE THIS
            
            image_slices = []
            mask_slices = []
            
            if self.rpn_mode == False:
                for i in range(mri_image.shape[2]):
                    augmented = self.transform(
                        image=mri_image[:, :, i], 
                        mask=segmentation_mask[:, :, i]
                    )
                
                    image_slices.append(augmented['image'].unsqueeze(0)) 
                    mask_slices.append(augmented['mask'].unsqueeze(0))   
    
                image = torch.stack(image_slices) 
                mask = torch.stack(mask_slices)  
            
                if image.shape[1] != 1 or mask.shape[1] != 1:
                    raise ValueError("Unexpected number of slices in the MRI image or segmentation mask.")

                return image, mask

            else: # RPN transformation
                for i in range(mri_image.shape[2]):
                    augmented = self.transform(
                        image=mri_image[:, :, i],
                        mask=segmentation_mask[:, :, i]
                    )

                    image_slices.append(augmented['image'].unsqueeze(0))
                    mask_slices.append(augmented['mask'].unsqueeze(0))

                image = torch.stack(image_slices)
                mask = torch.stack(mask_slices).float()

                truthsarr = []
                for i in mask:
                    truthsarr.append(self.i2p(i))
                truths = torch.stack(truthsarr)

                return image, truths
        
        except Exception as e:
            print(f"Error in __call__ with {mri_image} and {segmentation_mask}: {e}")
            return None, None
