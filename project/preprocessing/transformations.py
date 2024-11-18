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
    
def pad_to_square(image):
    image = image.numpy(force=True)
    height, width = image.shape[:2]
    square_size = max(height, width)

    # mean_value = float(np.mean(image))

    # print(mean_value)

    pad_top = (square_size - height) // 2
    pad_bottom = square_size - height - pad_top
    pad_left = (square_size - width) // 2
    pad_right = square_size - width - pad_left

    padded_image = cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # padded_image = np.pad(
    #     image,
    #     pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),  # Padding for (H, W, C)
    #     mode='reflect'
    # )
    return padded_image

class NiftiToTensorTransform:
    def __init__(self, target_shape=(512,512), in_channels=1, rpn_mode=False, normalization=None):
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
                        boxes = torch.tensor([0] * 4, dtype=torch.float32).unsqueeze(0)
                        labels = augmented['labels']

                    if boxes.shape[0] == 1:
                        image_slices.append(img_slice.unsqueeze(0))
                        boxes = torch.clamp(boxes, min=0, max=self.target_shape[0])
                        mask_slices.append(boxes.unsqueeze(0))
                    else: # if there are more than one bbox coordinates for a slice
                        # print('MULTIPLE BOXES FOUND')
                        # print(boxes)
                        image_slices.append(img_slice.unsqueeze(0))
                        max_x = boxes[0, 0]
                        max_y = boxes[0, 1]
                        max_w = boxes[0, 2]
                        max_h = boxes[0, 3]
                        for i in boxes[1:]:
                            x, y, w, h = i
                            if x < max_x:
                                max_x = x
                            if y < max_y:
                                max_y = y
                            if w > max_w:
                                max_w = w
                            if h > max_h:
                                max_h = h
                            # mask_slices.append(i.unsqueeze(0).unsqueeze(0))

                        bbox = torch.tensor([max_x, max_y, max_w, max_h])
                        bbox = torch.clamp(bbox, min=0, max=self.target_shape[0])
                        # print('============== FINAL BOX')
                        # print(bbox)
                        mask_slices.append(bbox.unsqueeze(0).unsqueeze(0))

                image = torch.stack(image_slices) 
                mask = torch.stack(mask_slices)  
            
                if image.shape[1] != 1 or mask.shape[1] != 1:
                    raise ValueError("Unexpected number of slices in the MRI image or segmentation mask.")

                return image, mask
        
        except Exception as e:
            print(f"Error in __call__ with {mri_image} and {segmentation_mask}: {e}")
            return None, None
