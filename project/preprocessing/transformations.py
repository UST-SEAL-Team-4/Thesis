import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

def get_transform(height, width, p):
    return A.Compose(
        [
            A.Resize(height=height, width=width, p=p),
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