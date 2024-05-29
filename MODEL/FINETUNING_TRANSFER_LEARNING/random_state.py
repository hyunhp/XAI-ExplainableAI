import torch
import numpy as np
import random
import argparse

def set_seed(seed:int=42):
    '''
    ## Example usage:
    random_seed = 42  (Choose any integer value as the random seed)
        
        - argparse
        1. RandomNumber : Random Numbe to set enviroment stable.
        
        - output
        1. Random seed staiblized
    '''
    # Set the random seed for PyTorch on CPU and GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for Python's built-in random module
    random.seed(seed)
    
    print(f'RANDOM STATE {seed} IS STAIBLIZED....')