
import numpy as np
from Circle import Circle

import scipy

class ImageHash:
    def __init__(self):
        pass

    def compute_features(self, img):
        # output a n_features x feature_size matrix
        # FIXED

        # FILL: generate x,y to sample
        # FILL: sample and take max min as needed

        pass

    def get_xy_to_sample():
        raise NotImplementedError()



class CircleHash(ImageHash):
    def __init__(self):
        super().__init__()
        self.circles = self.get_circle_params()
    
    def get_circle_params(img_shape, n_features, r_bnd = [20,50]):
        # given an image shape, return the centre and radii of the circles
        assert(img_shape.shape == [1,2])
        circs = []
        for feature_idx in range(n_features):
            circs.append(Circle(np.random.rand(1,2)*img_shape, r_bnd[0] + np.random.rand()*r_bnd[1]))
        
        return circs

    def compute_features(self, img):
        for circle in self.circles:
            pass

class LineHash(ImageHash):
    def __init__(self):
        super().__init__()


