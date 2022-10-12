
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
        self.analog_ops = [np.max, np.min]

    def compute_features(self, img):
        spline_interp = scipy.interpolate.RectBivariateSpline(range(self.img_size[0]),
                                                range(self.img_size[1]),img,kx=1,ky=1)
        gridded_interp = scipy.interpolate.RegularGridInterpolator(range(self.img_size[0]), range(self.img_size[1]))                                        
        features = np.zeros([self.n_features,len(self.analog_ops)])
        for i, obj in enumerate(self.objects):
            samp = obj.get_xy_samples(100)
            curve = spline_interp.ev(samp[:,0],samp[:,1])
            

            # apply each analog operation to the interpolated curve
            for op_idx in range(len(self.analog_ops)):
                features[i,op_idx] = self.analog_ops[op_idx](curve)   

        print("out of bounds",spline_interp.ev(-1,-1))
        return features

    def get_xy_to_sample(self):
        raise NotImplementedError()



class CircleHash(ImageHash):
    def __init__(self, img_size, n_features, is_random=True):
        super().__init__(img_size)
        self.n_features = n_features
        self.objects = self.get_circle_params(n_features)
        self.is_random = is_random
    
    def get_circle_params(self, n_features, r_bnd = [20,50]):
        # given an image shape, return the centre and radii of the circles
        circs = []
        for _ in range(n_features):
            circs.append(Circle(np.random.rand(2,)*self.img_size, r_bnd[0] + np.random.rand()*(r_bnd[1]-r_bnd[0])))
        return circs

    def compute_features(self, img):
        # if is randomised, reshuffle the circles
        if self.is_random:
            self.objects = self.get_circle_params(self.n_features)
        
        super().compute_features(img)


class LineHash(ImageHash):
    def __init__(self):
        super().__init__()


if __name__ == "__main__":
    import cv2
    img = cv2.imread('data_cleaned/mono.png', cv2.IMREAD_GRAYSCALE)
    img = img[0:128,:]
    print(img.shape)
    circ_hash = CircleHash(img.shape, 10)
    print(circ_hash.compute_features(img))