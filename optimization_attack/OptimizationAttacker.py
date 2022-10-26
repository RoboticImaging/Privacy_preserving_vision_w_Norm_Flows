
import numpy as np
import scipy.optimize


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
    
    def flatten_img(self, img):
        return img.flatten()
    
    def unflatten_img(self, img_flat):
        return np.reshape(img_flat, self.image_hasher.get_image_size())

    def get_hash_of_est(self, img_flat):
        img = self.unflatten_img(img_flat)
        return self.image_hasher.compute_features(img)


class ImageSpaceAttack(OptimizationAttacker):
    # attack by trying to optimize the image directly
    def __init__(self, image_hasher):
        super().__init__(image_hasher)

    def attack(self, target_hash, starting_img):
        opt_fn = lambda img_flat : OptimizationAttacker.hash_diff(self.get_hash_of_est(img_flat), target_hash)

        # print(starting_img.flatten())
        # print(np.all(np.reshape(starting_img.flatten(), self.image_hasher.get_image_size()) == starting_img))

        result = scipy.optimize.least_squares(opt_fn , starting_img.flatten(), verbose=2, ftol=1e-2, xtol=1e-5, bounds=(0,255))

        return self.unflatten_img(result.x), result.success, result.nfev


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
    img = cv2.resize(img, (img.shape[1]//8 , img.shape[0]//8))

    ch = CircleHasher(img.shape, 10, False)

    true_hash = ch.compute_features(img)
    print(true_hash)

    print(img.shape, (img.shape[0]//4 , img.shape[1]//4))
    img_smol = cv2.resize(img, (img.shape[1]//4 , img.shape[0]//4))
    downsampled_img = cv2.resize(img_smol, (img.shape[1], img.shape[0]))
    print(downsampled_img.shape)
    print(ch.compute_features(downsampled_img))


    attacker = ImageSpaceAttack(ch)
    final_img = attacker.attack(true_hash, downsampled_img)
    print(final_img)

    # print(ch.compute_features(img))


