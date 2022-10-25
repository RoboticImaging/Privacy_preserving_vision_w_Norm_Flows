
from layers import *
from ImageFlow import ImageFlow
import torchvision
from torchvision import transforms

def create_multiscale_flow(height, width, n_vardq=4, n_coupling_pre_split=2, n_coupling_post_split=4):
    flow_layers = []
    
    vardeq_layers = [CouplingLayer(network=GatedConvNet(c_in=2, c_out=2, c_hidden=16),
                                   mask=create_checkerboard_mask(h=height, w=width, invert=(i%2==1)),
                                   c_in=1) for i in range(n_vardq)]
    flow_layers += [VariationalDequantization(vardeq_layers)]
    
    flow_layers += [CouplingLayer(network=GatedConvNet(c_in=1, c_hidden=32),
                                  mask=create_checkerboard_mask(h=height, w=width, invert=(i%2==1)),
                                  c_in=1) for i in range(n_coupling_pre_split)]
    flow_layers += [SqueezeFlow()]
    for i in range(n_coupling_pre_split):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=4, c_hidden=48),
                                      mask=create_channel_mask(c_in=4, invert=(i%2==1)),
                                      c_in=4)]
    flow_layers += [SplitFlow(),
                    SqueezeFlow()]
    for i in range(n_coupling_post_split):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_in=8, c_hidden=64),
                                      mask=create_channel_mask(c_in=8, invert=(i%2==1)),
                                      c_in=8)]

    device = torch.device("cpu")
    # device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    flow_model = ImageFlow(flow_layers).to(device)
    return flow_model




def train_flow(flow, train_set, val_set, model_name="MNISTFlow", n_epochs=200, from_version=None):
    extra_args = {}
    if from_version is not None:
        if os.path.exists(os.path.join(CHECKPOINT_PATH, model_name,'lightning_logs')):
            if from_version == -1:
            # use latest version
                version_list = os.listdir(os.path.join(CHECKPOINT_PATH, model_name,'lightning_logs'))

                version_list.sort(key = lambda x: int(x.split('_')[-1]))

                ckpt_name = os.listdir(os.path.join(CHECKPOINT_PATH, model_name,'lightning_logs',version_list[-1],'checkpoints'))[0]
                ckpt_path = os.path.join(CHECKPOINT_PATH, model_name,'lightning_logs',version_list[-1],'checkpoints', ckpt_name)
            else:
                # use version indicated
                raise NotImplementedError
            extra_args = {"resume_from_checkpoint": ckpt_path}
        
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_name), 
                         gpus=1 if torch.cuda.is_available() else 0, 
                         max_epochs=n_epochs, 
                         gradient_clip_val=1.0,
                         callbacks=[ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_bpd"),
                                    LearningRateMonitor("epoch")],
                         check_val_every_n_epoch=5,
                         **extra_args)
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    train_data_loader = data.DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
    val_loader = data.DataLoader(val_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    result = None
    
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, model_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        ckpt = torch.load(pretrained_filename, map_location=device)
        flow.load_state_dict(ckpt['state_dict'])
        result = ckpt.get("result", None)
    else:
        print("Start training", model_name)
        trainer.fit(flow, train_data_loader, val_loader)
    
    # Test best model on validation and test set if no result has been found
    # Testing can be expensive due to the importance sampling.
    if result is None:
        val_result = trainer.test(flow, val_loader, verbose=False)
        start_time = time.time()
        test_result = trainer.test(flow, test_loader, verbose=False)
        duration = time.time() - start_time
        result = {"test": test_result, "val": val_result, "time": duration / len(test_loader) / flow.import_samples}
    
    return flow, result


# Convert images from 0-1 to 0-255 (integers)
def discretize(sample):
    return (sample * 255).to(torch.int32)

if __name__ == "__main__":

    n_pix = 64
    n_epochs = 300

    DATASET_PATH = f"data/LSUN_Bedroom/{n_pix}x{n_pix}"
    CHECKPOINT_PATH = f"saved_models/bedroom_flows/{n_pix}x{n_pix}/"

    overall_transform =  torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                                     transforms.ToTensor(),
                                                     discretize])
    full_dset = torchvision.datasets.ImageFolder(root=DATASET_PATH, transform=overall_transform)
    train_val_test_splits = np.array([0.4,0.1,0.1])
    n_imgs = len(full_dset)

    img_splits = (train_val_test_splits*n_imgs).round().astype(int)
    img_splits[2] = n_imgs - img_splits[0:2].sum()


    train_set, val_set, test_set = torch.utils.data.random_split(full_dset, list(img_splits), 
                                                    generator=torch.Generator().manual_seed(42))


    model, result = train_flow(create_multiscale_flow(n_pix,n_pix, n_vardq=6, n_coupling_pre_split=4,n_coupling_post_split=7), train_set,val_set, model_name="bedroomFlow_multiscale_complex", from_version=-1, n_epochs=n_epochs)