import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import torch
import nibabel as nib
import cv2

def get_transform(height, width, p, rpn_mode):
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
            # p=p,
            # bbox_params=A.BboxParams(
            #     format='pascal_voc',
            #     min_area=0,
            #     min_visibility=0,
            #     label_fields=['labels']
            # )
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
    def __init__(self, target_shape=(512,512), in_channels=1, rpn_mode=False):
        self.target_shape = target_shape
        self.in_channels = in_channels
        self.transform = get_transform(
            height=target_shape[0],
            width=target_shape[1],
            p=1.0,
            rpn_mode=rpn_mode
        )
        self.rpn_mode = rpn_mode
        
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
            boxes.append([x, y, x + w, y + h])
        return boxes

    def __call__(self, mri_image_path, segmentation_mask_path):
        try:
            mri_image = nib.load(mri_image_path).get_fdata()
            segmentation_mask = nib.load(segmentation_mask_path).get_fdata()
            
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

            else:
                for i in range(mri_image.shape[2]):
                    boxes = self.extract_bounding_boxes(segmentation_mask[:, :, i])
                    if boxes:
                        augmented = self.transform(
                            image=mri_image[:, :, i],
                            bboxes=boxes,
                            labels=[1]*len(boxes)
                        )
                        img_slice = augmented['image']
                        boxes = torch.tensor(augmented['bboxes'])
                        labels = augmented['labels']
                    else:
                        augmented = self.transform(
                            image=mri_image[:, :, i],
                            bboxes=[],
                            labels=[]
                        )
                        img_slice = augmented['image']
                        boxes = torch.tensor([-1] * 4, dtype=torch.float32).unsqueeze(0)
                        labels = augmented['labels']

                    image_slices.append(img_slice.unsqueeze(0))
                    mask_slices.append(boxes)

                image = torch.stack(image_slices) 
                mask = torch.stack(mask_slices)  
            
                if image.shape[1] != 1 or mask.shape[1] != 1:
                    raise ValueError("Unexpected number of slices in the MRI image or segmentation mask.")

                return image, mask
        
        except Exception as e:
            print(f"Error in __call__ with {mri_image} and {segmentation_mask}: {e}")
            return None, None
