
from NFEvaluator import NFEvaluator
import torchvision
from torchvision import transforms
import torch
from train import discretize
import numpy as np


if __name__ == "__main__":
    n_pix = 64
    DATASET_PATH = f"data/LSUN_Bedroom/{n_pix}x{n_pix}"

    if n_pix == 32:
        model_name = 'bedroomFlow_multiscale'
    elif n_pix == 64:
        model_name = 'bedroomFlow_multiscale_complex'
    else:
        raise NotImplementedError()

    overall_transform =  torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                                     transforms.ToTensor(),
                                                     discretize])
    full_dset = torchvision.datasets.ImageFolder(root=DATASET_PATH, transform=overall_transform)
    if model_name == 'bedroomFlow_multiscale':
        train_val_test_splits = np.array([0.8,0.1,0.1])
    elif model_name == 'bedroomFlow_multiscale_complex':
        train_val_test_splits = np.array([0.4,0.1,0.1])
    else:
        raise NotImplementedError(f"{model_name} doen't have dsets defined")
    n_imgs = len(full_dset)

    img_splits = (train_val_test_splits*n_imgs).round().astype(int)
    img_splits[2] = n_imgs - img_splits[0:2].sum()

    train_set, val_set, test_set = torch.utils.data.random_split(full_dset, list(img_splits), 
                                                    generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=False, drop_last=False)
 
    eval = NFEvaluator(n_pix, model_name, train_loader)
    # eval.standard_interp()
    # eval.show_random_samples()
    # eval.interp_inside_out()
    # eval.interp_inside_out_rand_dir()
    # eval.interp_inside_out_zoomed()
    # eval.interp_inside_out_rand_dir_zoomed()
    # eval.hist_of_training_imgs()
    eval.dist_of_noise_and_inverted()