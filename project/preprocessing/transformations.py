import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import torch
import nibabel as nib

def get_transform(height, width, p):
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
    
class NiftiToTensorTransform:
    def __init__(self, target_shape=(512,512), in_channels=1):
        self.target_shape = target_shape
        self.in_channels = in_channels
        self.transform = get_transform(
            height=target_shape[0],
            width=target_shape[1],
            p=1.0
        )
        
    def convert_to_binary_mask(self, segmentation_mask):
        binary_mask = (segmentation_mask > 0).astype(np.uint8)
        return binary_mask
    
    def __call__(self, mri_image_path, segmentation_mask_path):
        try:
            mri_image = nib.load(mri_image_path).get_fdata()
            segmentation_mask = nib.load(segmentation_mask_path).get_fdata()
            
            segmentation_mask = self.convert_to_binary_mask(segmentation_mask)
            
            image_slices = []
            mask_slices = []
            
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
        
        except Exception as e:
            print(f"Error in __call__ with {mri_image} and {segmentation_mask}: {e}")
            return None, None