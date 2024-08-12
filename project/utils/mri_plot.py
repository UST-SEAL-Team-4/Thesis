import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_mri_slice(file_path, slice_index=30, get_middle_slice=False):
    img = nib.load(file_path)
    data = img.get_fdata()
    
    num_slices = data.shape[2]
    if get_middle_slice is True:
        slice_index = num_slices // 2
    
    plt.figure(figsize=(10, 5))
    plt.imshow(data[:, :, slice_index], cmap='gray')
    plt.title(f'Axial Slice (Slice {slice_index})')
    plt.axis('off')
    plt.show()
    
def plot_all_slices(img_path):
    img = nib.load(img_path).get_fdata()

    num_cols = 5
    num_rows = (img.shape[2] + num_cols - 1) // num_cols

    _, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))

    for i in range(img.shape[2]):
        ax = axes[i // num_cols, i % num_cols]

        mri_slice = img[:, :, i]

        ax.imshow(mri_slice, cmap='gray')
        ax.set_title(f'Slice {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# needs update. maybe create another function for getting the bounding boxes of the cmb masks and apply it here
def plot_all_slices_with_bboxes(img_path, ann_path):
    img = nib.load(img_path).get_fdata()
    ann = nib.load(ann_path).get_fdata()

    num_cols = 5
    num_rows = (img.shape[2] + num_cols - 1) // num_cols

    _, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))

    for i in range(img.shape[2]):
        ax = axes[i // num_cols, i % num_cols]

        mri_slice = img[:, :, i]
        segmented_slice = ann[:, :, i]

        contours, _ = cv2.findContours(segmented_slice.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ax.imshow(mri_slice, cmap='gray')

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            rect = plt.Rectangle((x, y), w, h, linewidth=2,
                                 edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        ax.set_title(f'Slice {i+1} - CMBs: {len(contours)}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()