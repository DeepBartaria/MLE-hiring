import os
import random
import numpy as np

def set_seed(seed: int = 42) -> None:
    """
    Set deterministic random seed for reproducibility across the application.
    This ensures identical runs on the same dataset.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
