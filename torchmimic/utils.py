import os
import shutil
import torch

import numpy as np
from torch.nn.utils.rnn import pad_sequence


def pad_colalte(batch):
    xx, yy, lens = zip(*batch)
    x = pad_sequence(xx, batch_first=True, padding_value=-np.inf)
    y = torch.stack(yy, dim=0)

    mask = (x == -np.inf)[:, :, 0]
    return x, y, lens, mask


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print(f"Experiment dir: {path}")

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, "scripts")):
            os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)
