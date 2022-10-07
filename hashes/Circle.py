import numpy as np


class Circle:
    def __init__(self, centre, radius):
        self.centre = centre
        self.radius = radius

    def get_xy_samples(self, n_samp=100):
        theta = np.linspace(0, 2*np.pi, n_samp+1)[0:-1]
        x = self.centre[0] + self.radius*np.cos(theta)
        y = self.centre[1] + self.radius*np.sin(theta)
        return np.concatenate([x,y])