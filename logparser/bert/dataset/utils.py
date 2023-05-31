import os
import random

import numpy as np
import torch


def save_parameters(options, filename):
    with open(filename, "w+") as f:
        for key in options.keys():
            f.write(f"{key}: {options[key]}\n")


# https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
