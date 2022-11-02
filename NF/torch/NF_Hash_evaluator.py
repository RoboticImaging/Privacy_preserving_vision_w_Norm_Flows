

from NFEvaluator import NFEvaluator


class NFHashEvaluator:
    def __init__(self, hasher, n_pix, model_name, train_loader, 
                    ckpt_pth = 'saved_models/bedroom_flows/', 
                    save_pth='outputs'):
        # load the model 
        self.n_pix = n_pix
        self.hasher = hasher
        ckpt_pth = os.path.join(ckpt_pth, f'{n_pix}x{n_pix}')
        self.model_params = NFEvaluator.get_model_dict(model_name)
        self.model = self._read_model(model_name, ckpt_pth, n_pix)
        
        # setup figure saving path
        self.output_save_path = os.path.join(save_pth, f'{n_pix}x{n_pix}', model_name)
        if not os.path.exists(self.output_save_path):
            os.mkdir(self.output_save_path)

        # get example images
        self.train_loader = train_loader
        self.exmp_imgs, _ = next(iter(train_loader))




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
 
    eval = NFHashEvaluator(n_pix, model_name, train_loader)