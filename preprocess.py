from datasets import load_dataset
import os, io
import numpy as np
from PIL import Image
dataset = load_dataset('Maysee/tiny-imagenet')
for split, data in dataset.items():
    # def fn(batch):
    #     if len(batch['image'].size) == 2:
    #         batch['image'] = np.array(batch['image'].convert('RGB'))
    #     else: 
    #         batch['image'] = np.array(batch['image']) 

    #     return batch
    data = data.filter(lambda batch: len(np.array(batch['image']).shape) == 3)
    data.to_csv(
        os.path.join("src", "data", f"d_{split}.csv"), index = None
    )