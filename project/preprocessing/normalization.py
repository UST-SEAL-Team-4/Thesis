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
    return (image - np.mean(image)) / np.std(image)

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
    return (image - np.min(image)) / (np.max(image) - np.min(image))