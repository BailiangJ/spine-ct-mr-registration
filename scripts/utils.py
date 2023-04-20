import os
import random
from typing import Sequence

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'Random seed set as {seed}')


def worker_init_fn(worker_id):
    """Check https://github.com/Project-MONAI/MONAI/issues/1068."""
    worker_info = torch.utils.data.get_worker_info()
    try:
        worker_info.dataset.transform.set_random_state(worker_info.seed %
                                                       (2**32))
    except AttributeError:
        pass
