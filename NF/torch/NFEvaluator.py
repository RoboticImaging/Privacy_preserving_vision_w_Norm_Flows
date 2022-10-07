import os
from train import create_multiscale_flow
import torch

class NFEvaluator:
    def __init__(self, n_pix, model_name, train_loader, 
                    ckpt_pth = 'saved_models/bedroom_flows/32x32/', 
                    save_pth='outputs'):
        # load the model 
        self.model = self._read_model(model_name, ckpt_pth, n_pix)
        
        # setup figure saving path

        # get example images
        self.exmp_imgs, _ = next(iter(train_loader))


    def run_all_eval(self):
        self.get_results()
        # etc. etc.

    def get_results(self):
        # compute results as in UvA tute
        pass
    
    def standard_interp(self, save=False):
        # interp between some samples 
        pass

    def interp_inside_out(self, save=False):
        # interp out from mean to some fixed distance out through training data
        pass
    
    def interp_inside_out_rand_dir(self, save=False):
        # pick a random direction and interp out from it for a fixed distance
        pass
    
    def show_random_samples(self):
        # create some images from the model
        pass
    
    def hist_of_training_imgs(self):
        pass

    def _read_model(self, model_name, ckpt_path, n_pix):
        flow = create_multiscale_flow(n_pix,n_pix)
        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        pretrained_filename = os.path.join(ckpt_path, model_name,"model.ckpt")
        if os.path.isfile(pretrained_filename):
            print("Found pretrained model, loading...")
            ckpt = torch.load(pretrained_filename, map_location=device)
            flow.load_state_dict(ckpt['state_dict'])
        return flow

    def show_imgs(imgs, title=None, row_size=4):
        # Form a grid of pictures (we use max. 8 columns)
        num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
        is_int = imgs.dtype==torch.int32 if isinstance(imgs, torch.Tensor) else imgs[0].dtype==torch.int32
        nrow = min(num_imgs, row_size)
        ncol = int(math.ceil(num_imgs/nrow))
        imgs = torchvision.utils.make_grid(imgs, nrow=nrow, pad_value=128 if is_int else 0.5)
        np_imgs = imgs.cpu().numpy()
        # Plot the grid
        plt.figure(figsize=(1.5*nrow, 1.5*ncol))
        plt.imshow(np.transpose(np_imgs, (1,2,0)), interpolation='nearest')
        plt.axis('off')
        if title is not None:
            plt.title(title)

    