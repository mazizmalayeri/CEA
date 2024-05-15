import numpy as np
import random
import os
import torch
from sklearn.model_selection import train_test_split
import sklearn

def set_all_seeds(seed):

  """
  Set random seeds for reproducibility across multiple libraries and devices.

  Parameters:
  -----------
  seed: int
      The random seed to use for setting all random number generators.

  Example:
  --------
  >>> set_all_seeds(42)
  """
  
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  

def split_data(in_features_np, in_label_np, train_size=0.7, random_state=0):

    """
    Split input data into training, validation, and test sets.

    Parameters:
    -----------
    in_features_np: np.ndarray
        Input features as a NumPy array.
    in_label_np: np.ndarray
        Input labels as a NumPy array.
    train_size: float, optional (default=0.7)
        Proportion of the data to include in the training set.
    random_state: int, optional (default=0)
        Seed used by the random number generator for reproducibility.
    handle_imbalance_data: bool, optional (default=False)
        Whether to handle data imbalance by oversampling the minority classes in the training set.

    Returns:
    --------
    tuple
        A tuple containing two dictionaries:
        - X: A dictionary with keys 'train', 'val', and 'test', each containing NumPy arrays of features.
        - y: A dictionary with keys 'train', 'val', and 'test', each containing NumPy arrays of labels.
    """
        
    X, y = {}, {}
    X['train'], X['test'], y['train'], y['test'] = train_test_split(in_features_np, in_label_np, stratify=in_label_np, train_size=train_size, random_state=random_state)
    X['test'], X['val'], y['test'], y['val'] = train_test_split(X['test'], y['test'], stratify=y['test'], train_size=0.5, random_state=random_state)
    
    return X, y
    

def normalization(X, y, device):

    """
    Normalize input data.

    Parameters:
    -----------
    X: dict
        A dictionary containing NumPy arrays of input features for 'train', 'val', and 'test' splits.
    y: dict
        A dictionary containing NumPy arrays of input labels for 'train', 'val', and 'test' splits.
    device: torch.device
        The device (CPU or GPU) on which to store the normalized tensors.

    Returns:
    --------
    tuple
        A tuple containing three elements:
        - X_normalized: A dictionary with keys 'train', 'val', and 'test', each containing normalized input features as PyTorch tensors.
        - y: A dictionary with keys 'train', 'val', and 'test', each containing input labels as PyTorch tensors.
        - preprocess: The preprocessing transformation applied to the data (for possible future inverse transformation).
    """
    
    preprocess = sklearn.preprocessing.StandardScaler().fit(X['train'])
    
    X = {
        k: torch.tensor(preprocess.transform(v), device=device)
        for k, v in X.items()}
        
    y = {k: torch.tensor(v, device=device) for k, v in y.items()}
    
    return X, y, preprocess