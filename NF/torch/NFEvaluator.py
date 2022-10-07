import os
from train import create_multiscale_flow
import torch
import math
import torchvision
import matplotlib.pyplot as plt
import numpy as np

class NFEvaluator:
    def __init__(self, n_pix, model_name, train_loader, 
                    ckpt_pth = 'saved_models/bedroom_flows/', 
                    save_pth='outputs'):
        # load the model 
        ckpt_pth = os.path.join(ckpt_pth, f'{n_pix}x{n_pix}')
        self.model = self._read_model(model_name, ckpt_pth, n_pix)
        
        # setup figure saving path
        self.output_save_path = os.path.join(save_pth, f'{n_pix}x{n_pix}', model_name)
        if not os.path.exists(self.output_save_path):
            os.mkdir(self.output_save_path)

        # get example images
        self.exmp_imgs, _ = next(iter(train_loader))


    def run_all_eval(self):
        self.get_results()
        # etc. etc.

    def get_results(self):
        # compute results as in UvA tute
        pass
    
    def standard_interp(self, save=True):
        # interp between some samples 
        n_step = 8
        for i in range(2):
            interp_imgs = self._interpolate(self.exmp_imgs[2*i], self.exmp_imgs[2*i+1], n_step)
            NFEvaluator._show_imgs(interp_imgs)
            if save:
                plt.savefig(os.path.join(self.output_save_path, f"standard_interp_{i}.png"))

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

    @torch.no_grad()
    def _interpolate(self, img1, img2, num_steps=8):
        """
        Inputs:
            img1, img2 - Image tensors of shape [1, 28, 28]. Images between which should be interpolated.
            num_steps - Number of interpolation steps. 8 interpolation steps mean 6 intermediate pictures besides img1 and img2
        """
        imgs = torch.stack([img1, img2], dim=0).to(self.model.device)
        z, _ = self.model.encode(imgs)
        alpha = torch.linspace(0, 1, steps=num_steps, device=z.device).view(-1, 1, 1, 1)
        interpolations = z[0:1] * alpha + z[1:2] * (1 - alpha)
        interp_imgs = self.model.sample(interpolations.shape[:1] + imgs.shape[1:], z_init=interpolations)
        return interp_imgs

    def _show_imgs(imgs, title=None, row_size=8):
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

    