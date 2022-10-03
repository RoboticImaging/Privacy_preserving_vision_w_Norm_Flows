# testing if the torch data loader plays nice

import torch.utils.data
import numpy as np
import torchvision

def image_to_numpy(img):
    img = np.array(img, dtype=np.int32)
    img = img[...,None]  # Make image [28, 28, 1]
    return img

path = 'outputs/LSUN_Bedroom/16x16'

overall_transform =  torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                                     image_to_numpy])

full_dset = torchvision.datasets.ImageFolder(root=path, transform=overall_transform)

print(full_dset)

train_val_test_splits = np.array([0.8,0.1,0.1])
n_imgs = len(full_dset)

img_splits = (train_val_test_splits*n_imgs).round().astype(int)
img_splits[2] = n_imgs - img_splits[0:2].sum()

# print(img_splits)

train_set, val_set, test_set = torch.utils.data.random_split(full_dset, list(img_splits), 
                                                   generator=torch.Generator().manual_seed(42))


print(f"Train {len(train_set)}, Val {len(val_set)}, Test {len(test_set)}")


# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

train_exmp_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=False, drop_last=False, collate_fn=numpy_collate)
# Actual data loaders for training, validation, and testing
train_data_loader = torch.utils.data.DataLoader(train_set,
                                    batch_size=32,
                                    shuffle=True,
                                    drop_last=True,
                                    collate_fn=numpy_collate,
                                    num_workers=8,
                                    persistent_workers=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4, collate_fn=numpy_collate)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4, collate_fn=numpy_collate)



exmp_imgs, _ = next(iter(train_exmp_loader))

print(exmp_imgs.shape)
