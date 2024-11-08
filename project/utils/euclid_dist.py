import numpy as np

def get_euclid_dist(t1, t2):
    t1 = np.array(t1)
    t2 = np.array(t2)
    
    return np.sqrt(((t1-t2)**2).sum())