# testing if the torch data loader plays nice

import torch.utils.data
import numpy as np
import torchvision

def image_to_numpy(img):
    img = np.array(img, dtype=np.int32)
    img = img[...,None]  # Make image [28, 28, 1]
    return img

path = 'outputs/LSUN_Bedroom/16x16'

full_dset = torchvision.datasets.ImageFolder(root=path, transform=image_to_numpy)

print(full_dset)

train_val_test_splits = np.array([0.8,0.1,0.1])
n_imgs = len(full_dset)

img_splits = (train_val_test_splits*n_imgs).round()
img_splits[2] = n_imgs - img_splits[0:2].sum()

train_set, val_set, test_set = torch.utils.data.random_split(full_dset, img_splits, 
                                                   generator=torch.Generator().manual_seed(42))


print(f"Train {len(train_set)}, Val {len(val_set)}, Test {len(test_set)}")
