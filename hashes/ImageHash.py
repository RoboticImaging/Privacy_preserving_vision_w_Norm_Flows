
import numpy as np
from Circle import Circle

import scipy.interpolate

class ImageHash:
    def __init__(self, img_size):
        img_size = np.array(img_size)
        if img_size.shape == [1,2]:
            img_size = img_size.T
        self.img_size = img_size
        print(self.img_size)

    def compute_features(self, img):
        # output a n_features x feature_size matrix
        # FIXED

        # FILL: generate x,y to sample
        # FILL: sample and take max min as needed

        pass

    def get_xy_to_sample():
        raise NotImplementedError()



class CircleHash(ImageHash):
    def __init__(self, img_size, n_features):
        super().__init__(img_size)
        self.n_features = n_features
        self.circles = self.get_circle_params(n_features)
    
    def get_circle_params(self, n_features, r_bnd = [20,50]):
        # given an image shape, return the centre and radii of the circles
        circs = []
        for feature_idx in range(n_features):
            circs.append(Circle(np.random.rand(2,)*self.img_size, r_bnd[0] + np.random.rand()*(r_bnd[1]-r_bnd[0])))
        return circs

    def compute_features(self, img):
        spline_interp = scipy.interpolate.RectBivariateSpline(range(self.img_size[0]),
                                                range(self.img_size[1]),img,kx=1,ky=1)
        features = np.zeros([self.n_features,2])
        analog_operations = [np.max, np.min]
        for i, circle in enumerate(self.circles):
            samp = circle.get_xy_samples(100)
            curve = spline_interp.ev(samp[:,0],samp[:,1])

            # apply each analog operation to the interpolated curve
            for op_idx in range(analog_operations):
                features[i,op_idx] = analog_operations[op_idx](curve)   
                
        return features

class LineHash(ImageHash):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    import cv2
    img = cv2.imread('data_cleaned/mono.png', cv2.IMREAD_GRAYSCALE)
    img = img[0:128,:]
    print(img.shape)
    circ_hash = CircleHash(img.shape, 10)
    # for c in circ_hash.circles:
    #     print(c)
    print(circ_hash.compute_features(img))