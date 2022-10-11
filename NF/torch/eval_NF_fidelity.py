
from NFEvaluator import NFEvaluator
import torchvision
from torchvision import transforms
import torch
from train import discretize
import numpy as np


if __name__ == "__main__":
    n_pix = 32
    DATASET_PATH = f"data/LSUN_Bedroom/{n_pix}x{n_pix}"

    overall_transform =  torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                                     transforms.ToTensor(),
                                                     discretize])
    full_dset = torchvision.datasets.ImageFolder(root=DATASET_PATH, transform=overall_transform)
    train_val_test_splits = np.array([0.8,0.1,0.1])
    n_imgs = len(full_dset)

    img_splits = (train_val_test_splits*n_imgs).round().astype(int)
    img_splits[2] = n_imgs - img_splits[0:2].sum()

    train_set, val_set, test_set = torch.utils.data.random_split(full_dset, list(img_splits), 
                                                    generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=False, drop_last=False)
 
    eval = NFEvaluator(n_pix, 'bedroomFlow_multiscale', train_loader)
    eval.standard_interp()