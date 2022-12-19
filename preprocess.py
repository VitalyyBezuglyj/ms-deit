from datasets import load_dataset
import os, io
import numpy as np
from PIL import Image
dataset = load_dataset('Maysee/tiny-imagenet')
for split, data in dataset.items():
    # def fn(batch):
    #     batch['image'] = np.array(batch['image']) 
    #     if len(batch['image'].shape) == 2:
    #         batch['image'] = np.repeat(batch['image'][:, :, None], 3, axis=-1) 
    #     return batch
    # data.map(fn)
    data.to_csv(
        os.path.join("src", "data", f"d_{split}.csv"), index = None
    )