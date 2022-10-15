
import numpy as np



class OptimizationAttacker:
    def __init__(self, image_hasher):
        pass

    def attack(self, target_hash, starting_img):
        pass

    def hash_norm(hash1, hash2):
        if not (hash1.shape == hash2.shape):
            raise ValueError(f"Hashes need same shape: {hash1.shape} != {hash2.shape}")
        return np.linalg.norm(hash1 - hash2)


class ImageSpaceAttack(OptimizationAttacker):
    # attack by trying to optimize the image directly
    pass

class NFSpaceAttack(OptimizationAttacker):
    # attack using a trained Normalising Flow
    pass