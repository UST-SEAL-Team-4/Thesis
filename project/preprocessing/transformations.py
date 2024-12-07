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

        assert out.shape[0] == (img.shape[1]/self.patch_size)**2, f'Output does not match number of required regions: {out.shape} vs needed {(img.shape[1]/self.patch_size)**2}'

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
        self.patch_size = patch_size
        self.i2p = PatchTruther(patch_size)

        assert target_shape[0]/patch_size == int(target_shape[0]/patch_size)
        
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
            boxes.append([x-10, y-10, x + w+10, y + h+10])
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

                    cur_mask = self.i2p(anno_slice.unsqueeze(0).float())
                    assert cur_mask.shape[1] == (self.target_shape[0]/self.patch_size)**2, 'Shape of cur_mask not divided properly with needed amount of regions'
                    # patch_dim = int(self.target_shape[0]/self.patch_size)
                    # cur_mask = cur_mask.view(-1, patch_dim, patch_dim).flip(dims=(1,)).flatten(1)

                    # print(f'CURMASK SHAPE: {cur_mask.shape}') # SHAPE [1, 100]

                    # gridN = self.target_shape[0]/self.patch_size
                    # assert gridN == int(gridN)
                    # gridN = int(gridN)
                    # cur_mask = cur_mask.view(-1, gridN, gridN).flip(dims=(1,))

                    dimdim = int(self.target_shape[0]/self.patch_size)**2
                    base_regions = torch.zeros(dimdim, 4)

                    if boxes.shape[0] == 1:
                        image_slices.append(img_slice.unsqueeze(0))
                        boxes = torch.clamp(boxes, min=0, max=self.target_shape[0])
                        # mask_slices.append(boxes.unsqueeze(0))
                        # concat data to proper position here
                        anno_slices.append(anno_slice.unsqueeze(0))

                        # print(f'ONLY ONE: {boxes.shape}') # SHAPE [1, 4]
                        # print(f'X and Y: {boxes.squeeze()[0], boxes.squeeze()[1]} for {boxes}')
                        # print(f'ALL?: {all(x == 0 for x in boxes.squeeze())} for {boxes}')
                        x = (boxes.squeeze()[0] + boxes.squeeze()[2])/2
                        y = (boxes.squeeze()[1] + boxes.squeeze()[3])/2
                        num_patches = int(self.target_shape[0]/self.patch_size)
                        index = int(y/self.patch_size)*num_patches + int(x/self.patch_size)

                        x_base = self.patch_size * int(x/self.patch_size)
                        y_base = self.patch_size * int(y/self.patch_size)
                        boxes = boxes.squeeze()
                        boxes[[0, 2]] -= x_base-1
                        boxes[[1, 3]] -= y_base-1

                        base_regions[index] = boxes
                        base_regions = base_regions.permute(1, 0)

                        final_mask = torch.cat([cur_mask, base_regions])
                        mask_slices.append(final_mask)

                    else: # if there are more than one bbox coordinates for a slice
                        # print('MULTIPLE BOXES FOUND')
                        # print(boxes)
                        image_slices.append(img_slice.unsqueeze(0))
                        anno_slices.append(anno_slice.unsqueeze(0))

                        found_boxes = []
                        for bbox in boxes:
                            # concat data to proper position here
                            # print(f'MULTIPLE BOX SHAPE: {bbox.shape}') # SHAPE [4]
                            # found_boxes.append(bbox.unsqueeze(0))
                            x = (bbox[0] + bbox[2])/2
                            y = (bbox[1] + bbox[3])/2
                            num_patches = int(self.target_shape[0]/self.patch_size)
                            index = int(y/self.patch_size)*num_patches + int(x/self.patch_size)

                            x_base = self.patch_size * int(x/self.patch_size)
                            y_base = self.patch_size * int(y/self.patch_size)
                            bbox[[0, 2]] -= x_base-1
                            bbox[[1, 3]] -= y_base-1

                            base_regions[index] = bbox

                        base_regions = base_regions.permute(1, 0)
                        final_mask = torch.cat([cur_mask, base_regions])
                        mask_slices.append(final_mask)

                        # found_boxes = torch.stack(found_boxes)
                        # mask_slices.append(found_boxes)

                image = torch.stack(image_slices) 
                mask = torch.stack(mask_slices)
                assert image.shape[0] == len(mask)
                annot = torch.stack(anno_slices).float()

                # truthsarr = []
                # for i in annot:
                #     truthsarr.append(self.i2p(i))
                # truths = torch.stack(truthsarr)

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
