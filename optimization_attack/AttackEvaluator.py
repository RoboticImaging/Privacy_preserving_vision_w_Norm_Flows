from sre_constants import SUCCESS
import numpy as np

import cv2
import os

import sys
import time

import pickle

import OptimizationAttacker
import matplotlib.pyplot as plt

sys.path.append('./hashes')
from ImageHasher import CircleHasher, LineHasher



class AttackEvaluator:
    def __init__(self, datatset_path):
        self.datatset_path = datatset_path
        self.save_path = 'outputs\\optimization_attack'

    def eval_image_space_attack(self):
        img = cv2.imread(os.path.join(self.datatset_path, "00768.png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img.shape[1]//8 , img.shape[0]//8), interpolation=cv2.INTER_AREA)

        self.show_img(img)
        plt.savefig(os.path.join(self.save_path, 'image_space_attack', 'true_img.png'))

        # with open(os.path.join(self.save_path, 'image_space_attack', 'true_img'), 'wb') as f:
        #     pickle.dump(img, f)


        # ch = CircleHasher(img.shape, 50, False, r_bnd = (5,20))
        ch = LineHasher(img.shape, 50, False)

        start_time = time.time()
        true_hash = ch.compute_features(img)

        img_smol = cv2.resize(img, (img.shape[1]//2 , img.shape[0]//2), interpolation=cv2.INTER_AREA)
        downsampled_img = cv2.resize(img_smol, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)

        self.show_img(downsampled_img)
        plt.savefig(os.path.join(self.save_path, 'image_space_attack', 'starting_img.png'))

        # with open(os.path.join(self.save_path, 'image_space_attack', 'starting_img'), 'wb') as f:
        #     pickle.dump(downsampled_img, f)

        attacker = OptimizationAttacker.ImageSpaceAttack(ch)
        final_img, is_done, n_fev = attacker.attack(true_hash, downsampled_img)
        print(type(final_img))
        self.show_img(final_img)
        plt.title(f"SUCCESS ={is_done}, used {n_fev} function evals")
        plt.savefig(os.path.join(self.save_path, 'image_space_attack', 'final_img.png'))
        
        end_time = time.time()
        print(f"Elapsed time is {end_time-start_time:.2f}s")

        # with open(os.path.join(self.save_path, 'image_space_attack', 'final_img'), 'wb') as f:
        #     pickle.dump(final_img, f)

    
    def show_img(self, img):
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')


if __name__ == "__main__":
    dset = 'data_cleaned/Digeto_seq_2_subset'
    ae = AttackEvaluator(dset)
    ae.eval_image_space_attack()
