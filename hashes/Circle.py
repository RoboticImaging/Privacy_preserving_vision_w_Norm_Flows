import numpy as np


class Circle:
    def __init__(self, centre, radius, img_size):
        assert(centre.size == 2)
        self.centre = centre
        self.radius = radius
        self.img_size = img_size

    def get_xy_samples(self, n_samp=100):
        theta = np.linspace(0, 2*np.pi, n_samp+1)[0:-1]
        x = self.centre[0] + self.radius*np.cos(theta)
        y = self.centre[1] + self.radius*np.sin(theta)


        x = np.where((x < 0) | (x > self.img_size[0]-1), np.nan, x)
        y = np.where((y < 0) | (y > self.img_size[1]-1), np.nan, y)
        return np.concatenate([[x],[y]]).T
    
    def __str__(self):
        return f"Centre: ({self.centre[0]:.2f},{self.centre[1]:.2f}), r= {self.radius:.2f}"


if __name__ == "__main__":
    c = Circle(np.array([0,0]),1.)
    print(c.get_xy_samples(5))