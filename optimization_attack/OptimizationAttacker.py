
import numpy as np



class OptimizationAttacker:
    def __init__(self, image_hasher):
        self.image_hasher = image_hasher

    def attack(self, target_hash, starting_img):
        raise NotImplementedError

    def hash_diff(hash1, hash2):
        if not (hash1.shape == hash2.shape):
            raise ValueError(f"Hashes need same shape: {hash1.shape} != {hash2.shape}")
        return (hash1 - hash2).flatten()

    def hash_norm(hash1, hash2):
        return np.linalg.norm(OptimizationAttacker.hash_diff(hash1, hash2))


class ImageSpaceAttack(OptimizationAttacker):
    # attack by trying to optimize the image directly
    def __init__(self, image_hasher):
        super().__init__(image_hasher)

    def attack(self, target_hash, starting_img):
        opt_fn = lambda img_flat : OptimizationAttacker.hash_diff(self.get_hash_of_est(img_flat), target_hash)
        return opt_fn(starting_img)

    def get_hash_of_est(self, img_flat):
        img = np.reshape(img_flat, self.image_hasher.get_image_size())
        return self.image_hasher.compute_features(img_flat)

class NFSpaceAttack(OptimizationAttacker):
    # attack using a trained Normalising Flow
    def __init__(self, image_hasher, norm_flow):
        super().__init__(image_hasher)
        self.norm_flow = norm_flow

    def attack(self, target_hash, starting_img):
        pass


import sys
sys.path.append('./hashes')

if __name__ == "__main__":
    import cv2
    import os
    from ImageHasher import CircleHasher

    dset = 'data_cleaned\Digeto_seq_2_subset'
    # print(os.path.join(dset, "00001.png"))
    img = cv2.imread(os.path.join(dset, "00001.png"), cv2.IMREAD_GRAYSCALE)

    ch = CircleHasher(img.shape, 10, False)
    print(ch.compute_features(img))

    print(img.shape, (img.shape[0]//4 , img.shape[1]//4))
    img_smol = cv2.resize(img, (img.shape[1]//4 , img.shape[0]//4))
    downsampled_img = cv2.resize(img_smol, (img.shape[1], img.shape[0]))
    print(downsampled_img.shape)
    print(ch.compute_features(downsampled_img))


    # print(ch.compute_features(img))


