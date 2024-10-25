import nibabel as nib
import numpy as np

def compute_statistics(cases):
    '''Min-max normalization'''
    global_min = np.inf
    global_max = -np.inf
    
    for case in cases:
        image = nib.load(case).get_fdata() 
        
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            print(f"stopping because of {case}")
            raise ValueError
        
        local_min = np.min(image)
        local_max = np.max(image)
        
        if local_min < global_min:
            global_min = local_min
        if local_max > global_max:
            global_max = local_max

    return global_min, global_max

    '''Z-score normalization'''
    # num_voxels = np.float64(0)
    # sum_voxels = np.float64(0)
    # sum_squares = np.float64(0)
    
    # for case_path in cases:
    #     mri_image = nib.load(case_path).get_fdata() 
        
    #     if np.any(np.isnan(mri_image)) or np.any(np.isinf(mri_image)):
    #         print(f"stopping because of {case_path}")
    #         raise ValueError
        
    #     num_voxels += np.prod(mri_image.shape)
    #     sum_voxels += np.sum(mri_image)
    #     sum_squares += np.sum(np.square(mri_image))

    # mean = sum_voxels / num_voxels
    # std = np.sqrt(sum_squares / num_voxels - (mean ** 2))
    
    # return mean, std