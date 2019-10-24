import os
import random
import glob

import pandas as pd
import numpy as np
import torch

from src.utils.logger import log

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)