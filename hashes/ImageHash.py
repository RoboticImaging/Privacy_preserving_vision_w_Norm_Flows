
import numpy as np
from Circle import Circle
from Line import Line

import scipy.interpolate

class ImageHasher:
    def __init__(self, img_size):
        img_size = np.array(img_size)
        if img_size.shape == [1,2]:
            img_size = img_size.T
        self.img_size = img_size
        self.analog_ops = [np.nanmax, np.nanmin]

    def compute_features(self, img):
        # spline_interp = scipy.interpolate.RectBivariateSpline(range(self.img_size[0]),
        #                                         range(self.img_size[1]),img,kx=1,ky=1)
        gridded_interp = scipy.interpolate.RegularGridInterpolator((list(range(self.img_size[0])), list(range(self.img_size[1]))), img, bounds_error=False)                                        
        features = np.zeros([self.n_features,len(self.analog_ops)])
        for i, obj in enumerate(self.objects):
            samp = obj.get_xy_samples(5)
            print(samp)
            curve = gridded_interp(samp)
            print(curve)

            # apply each analog operation to the interpolated curve
            for op_idx in range(len(self.analog_ops)):
                features[i,op_idx] = self.analog_ops[op_idx](curve)   

        return features

    def get_xy_to_sample(self):
        raise NotImplementedError()
    
    def get_image_size(self):
        return self.img_size
    



class CircleHasher(ImageHasher):
    def __init__(self, img_size, n_features, is_random=True):
        super().__init__(img_size)
        self.n_features = n_features
        self.objects = self.get_circle_params(n_features)
        self.is_random = is_random
    
    def get_circle_params(self, n_features, r_bnd = [20,50]):
        # given an image shape, return the centre and radii of the circles
        circs = []
        for _ in range(n_features):
            circs.append(Circle(np.random.rand(2,)*self.img_size, r_bnd[0] + np.random.rand()*(r_bnd[1]-r_bnd[0]), self.img_size))
        return circs

    def compute_features(self, img):
        # if is randomised, reshuffle the circles
        if self.is_random:
            self.objects = self.get_circle_params(self.n_features)
        
        return super().compute_features(img)


class LineHasher(ImageHasher):
    def __init__(self, img_size, n_features, is_random=True):
        super().__init__(img_size)
        self.n_features = n_features
        self.is_random = is_random
        self.objects = self.get_line_params()

    def get_line_params(self):
        # given an image shape, return the centre and radii of the circles
        lines = []
        for _ in range(self.n_features):
            lines.append(Line(np.random.rand(2,)*self.img_size, np.random.rand()*2*np.pi, self.img_size))
        return lines

    def compute_features(self, img):
        # if is randomised, reshuffle the circles
        if self.is_random:
            self.objects = self.get_line_params()
        
        return super().compute_features(img)

if __name__ == "__main__":
    np.random.seed(1)
    import cv2
    img = cv2.imread('data_cleaned/mono.png', cv2.IMREAD_GRAYSCALE)
    img = img[0:128,:]

    # line_hasher = LineHasher(img.shape, 5, False)
    # print(line_hasher.compute_features(img))
    # print(line_hasher.compute_features(img))

    circ_hasher = CircleHasher(img.shape, 3, False)
    print(circ_hasher.compute_features(img))
    print(circ_hasher.compute_features(img))