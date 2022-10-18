import numpy as np

import cv2
import os

import sys

import OptimizationAttacker
import matplotlib.pyplot as plt

sys.path.append('./hashes')
from ImageHasher import CircleHasher



class AttackEvaluator:
    def __init__(self, datatset_path):
        self.datatset_path = datatset_path
        self.save_path = 'outputs\\optimization_attack'

    def eval_image_space_attack(self):
        img = cv2.imread(os.path.join(self.datatset_path, "00001.png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img.shape[1]//8 , img.shape[0]//8))

        self.show_img(img)
        plt.savefig(os.path.join(self.save_path, 'image_space_attack', 'starting_img.png'))

        ch = CircleHasher(img.shape, 100, False)

        true_hash = ch.compute_features(img)

        img_smol = cv2.resize(img, (img.shape[1]//2 , img.shape[0]//2))
        downsampled_img = cv2.resize(img_smol, (img.shape[1], img.shape[0]))

        attacker = OptimizationAttacker.ImageSpaceAttack(ch)
        final_img = attacker.attack(true_hash, downsampled_img)
        print(type(final_img))
        self.show_img(final_img)
        plt.savefig(os.path.join(self.save_path, 'image_space_attack', 'final_img.png'))

    
    def show_img(self, img):
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')


if __name__ == "__main__":
    dset = 'data_cleaned/Digeto_seq_2_subset'
    ae = AttackEvaluator(dset)
    ae.eval_image_space_attack()
