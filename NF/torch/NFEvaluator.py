import os
from train import create_multiscale_flow
import torch
import math
import torchvision
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cpu")
# device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")

class NFEvaluator:
    def __init__(self, n_pix, model_name, train_loader, 
                    ckpt_pth = 'saved_models/bedroom_flows/', 
                    save_pth='outputs'):
        # load the model 
        self.n_pix = n_pix
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


    def get_model_dict(model_name):
        if model_name == 'bedroomFlow_multiscale':
            model_set = {
                'n_vardq' : 4,
                'n_coupling_pre_split' : 2, 
                'n_coupling_post_split' : 4
                }
        elif model_name == 'bedroomFlow_multiscale_complex':
            model_set = {
                'n_vardq' : 6,
                'n_coupling_pre_split' : 4, 
                'n_coupling_post_split' : 7
                }
        else:
            raise NotImplementedError(f"{model_name} doen't have dsets defined")
        
        return model_set

    def run_all_eval(self):
        self.get_results()
        # etc. etc.

    def get_results(self):
        # compute results as in UvA tute
        pass
    
    def standard_interp(self, save=True, n_imgs = 6):
        # interp between some samples 
        n_step = 8
        for i in range(n_imgs):
            interp_imgs = self._interpolate(self.exmp_imgs[2*i], self.exmp_imgs[2*i+1], n_step)
            self._show_imgs(interp_imgs)
            if save:
                margin = 0.00001
                plt.subplots_adjust(left=margin, bottom=margin, right=1-margin, top=1-margin)
                plt.savefig(os.path.join(self.output_save_path, f"standard_interp_{i}.png"))
                # plt.savefig(os.path.join(self.output_save_path, f"standard_interp_{i}.png"), bbox_inches='tight')

    def interp_inside_out(self, save=True, n_times = 3, num_steps = 15):
        # interp out from mean to some fixed distance out through training data
        
        for i in range(n_times):
            imgs = torch.stack([self.exmp_imgs[i], self.exmp_imgs[i]], dim=0).to(self.model.device)
            z, _ = self.model.encode(imgs)

            zVals = torch.zeros_like(z)
            zVals[0,:] = z[1,:]/torch.norm(z[1,:])

            alpha = torch.linspace(0, self.n_pix*2, steps=num_steps, device=z.device).view(-1, 1, 1, 1)
            interpolations = zVals[0:1] * alpha + zVals[1:2] * (1 - alpha)

            interp_imgs = self.model.sample(interpolations.shape[:1] + imgs.shape[1:], z_init=interpolations)
            self._show_imgs(interp_imgs, row_size=num_steps)
            if save:
                plt.savefig(os.path.join(self.output_save_path, f"inside_out_{i}.png"))

    def interp_inside_out_rand_dir(self, save=True, n_times = 3, num_steps = 15):
        # pick a random direction and interp out from it for a fixed distance
        for i in range(n_times):
            imgs = torch.stack([self.exmp_imgs[i], self.exmp_imgs[i]], dim=0).to(self.model.device)
            z, _ = self.model.encode(imgs)


            zVals = torch.zeros_like(z)
            rand = torch.randn_like(z[1,:])
            zVals[0,:] = rand/torch.norm(rand)

            alpha = torch.linspace(0, 2*self.n_pix, steps=num_steps, device=z.device).view(-1, 1, 1, 1)
            interpolations = zVals[0:1] * alpha + zVals[1:2] * (1 - alpha)

            interp_imgs = self.model.sample(interpolations.shape[:1] + imgs.shape[1:], z_init=interpolations)
            print(interp_imgs[0,0,:,:])
            self._show_imgs(interp_imgs, row_size=num_steps)
            if save:
                plt.savefig(os.path.join(self.output_save_path, f"inside_out_rand_{i}.png"))
    
    def show_random_samples(self, n_imgs = 16, save=True):
        # create some images from the model
        samples = self.model.sample(img_shape=[n_imgs, 8,self.n_pix//4,self.n_pix//4])
        self._show_imgs(samples.cpu())
        if save:
            plt.savefig(os.path.join(self.output_save_path, f"random_sample.png"))
    
    def hist_of_training_imgs(self, save=True, hundreds_of_imgs=2):
        # the histogram of the distance from the origin of the training images
        im_stack = []
        for i in range (hundreds_of_imgs):
            exmp_imgs, _ = next(iter(self.train_loader))
            im_stack += [*exmp_imgs[0:100]]

            imgs = torch.Tensor(np.stack(im_stack, axis=0)).to(device)
            zz, _ = self.model.encode(imgs)
            if i == 0:
                z = zz
            else:
                z = torch.cat([z,zz], dim=0)


        dists = NFEvaluator._get_images_distance(z)

        fig = plt.figure()
        fig.patch.set_facecolor('w')
        plt.hist(dists)
        plt.ylabel('Count')
        plt.xlabel('Distance from origin')
        plt.title('Encoding the training images to gaussian space')
        if save:
            plt.savefig(os.path.join(self.output_save_path, f"training_hist.png"))

    def dist_of_noise_and_inverted(self, save=True, hundreds_of_imgs=10):
        # see how far the inverted version of images are
        im_stack = []
        for i in range (hundreds_of_imgs):
            exmp_imgs, _ = next(iter(self.train_loader))
            im_stack += [*exmp_imgs[0:20]]

            imgs = torch.Tensor(np.stack(im_stack, axis=0)).to(device)
            zz, _ = self.model.encode(imgs)
            zzz, _ = self.model.encode(NFEvaluator._invert_image(imgs))

            rand_imgs = torch.rand_like(imgs)*255
            zzzz, _ = self.model.encode(rand_imgs)
            if i == 0:
                z_train = zz
                z_inverted = zzz
                z_rand = zzzz
            else:
                z_train = torch.cat([z_train,zz], dim=0)
                z_inverted = torch.cat([z_inverted, zzz], dim=0)
                z_rand = torch.cat([z_rand, zzzz], dim=0)

        plt.figure()
        self._show_imgs(NFEvaluator._invert_image(imgs))
        plt.savefig(os.path.join(self.output_save_path, 'example_inverted_image.png'))

        train_dists = NFEvaluator._get_images_distance(z_train)
        invert_dists = NFEvaluator._get_images_distance(z_inverted)
        rand_dists = NFEvaluator._get_images_distance(z_rand)

        fig = plt.figure()
        fig.patch.set_facecolor('w')
        plt.hist(train_dists, label='Training')
        plt.hist(invert_dists, label='Inverted')
        plt.hist(rand_dists, label='Noise')
        plt.legend()
        plt.ylabel('Count')
        plt.xlabel('Distance from origin')
        if save:
            plt.savefig(os.path.join(self.output_save_path, f"hist.png"))

    
    def _invert_image(imgs):
        return (255-imgs).to(torch.int32)

    def _read_model(self, model_name, ckpt_path, n_pix):
        flow = create_multiscale_flow(n_pix,n_pix, **self.model_params)
        device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
        pretrained_filename = os.path.join(ckpt_path, model_name,"model.ckpt")
        if os.path.isfile(pretrained_filename):
            print("Found pretrained model, loading...")
            ckpt = torch.load(pretrained_filename, map_location=device)
            flow.load_state_dict(ckpt['state_dict'])
        return flow

    def _get_images_distance(z):
        return np.linalg.norm(z.reshape([z.shape[0], -1]).cpu().detach().numpy(), axis=1)

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

    def _show_imgs(self,imgs, title=None, row_size=8):
        # Form a grid of pictures (we use max. 8 columns)
        num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
        is_int = imgs.dtype==torch.int32 if isinstance(imgs, torch.Tensor) else imgs[0].dtype==torch.int32
        nrow = min(num_imgs, row_size)
        ncol = int(math.ceil(num_imgs/nrow))
        imgs = torchvision.utils.make_grid(imgs, nrow=nrow, pad_value=128 if is_int else 0.5)
        np_imgs = imgs.cpu().numpy()
        # Plot the grid
        plt.figure(figsize=(1.5*nrow, 1.5*ncol))
        plt.imshow(np.transpose(np_imgs, (1,2,0)), interpolation='nearest', vmin=0, vmax=255)
        plt.axis('off')
        if title is not None:
            plt.title(title)

    