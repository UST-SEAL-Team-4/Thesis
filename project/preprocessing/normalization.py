import numpy as np

def z_score_normalization(image):
    """
    Apply z-score normalization to the image by transforming its 
    pixel intensity value to have mean of 0 and standard deviation of 1
    
    Parameters:
    -----------
    image: numpy.ndarray
    
    Returns:
    -----------
    numpy.ndarray
    """
    
    std_dev = np.std(image)
    
    if std_dev == 0:
        return np.zeros_like(image)
    
    return (image - np.mean(image)) / std_dev

def min_max_normalization(image):
    """
    Apply min-max normalization to the image by transforming its 
    pixel intensity value to a specified range like [0, 1].
    
    Parameters:
    -----------
    image: numpy.ndarray
    
    Returns:
    -----------
    numpy.ndarray
    """
    
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val == min_val:
        return np.zeros_like(image)
    
    return (image - min_val) / (max_val - min_val)