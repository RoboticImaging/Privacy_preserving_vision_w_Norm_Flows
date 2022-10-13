import numpy as np
import matplotlib.pyplot as plt


class Line:
    def __init__(self, point, orientation, img_size):
        assert(point.size == 2)
        assert(not np.isclose(orientation % np.pi/2, 0))
        self.point = point
        self.orientation = orientation
        self.img_size = img_size

        axesDir = np.array([[1, 0],
                            [1, 0],
                            [0, 1],
                            [0, 1]]).T
        axesPts = np.array([[1,1],
                            [1, img_size[0]],
                            [1, 1],
                            [img_size[1], 1]]).T # ones used for axes intercepts since this is the edge of the image

        self.direction_vec = np.array([np.cos(self.orientation), np.sin(self.orientation)])

        n_sides = 4 # of a rectangle

        tVals = np.zeros([n_sides,1])
        for i in range(0,n_sides):
            matrix = np.concatenate([axesDir[:,i][:, np.newaxis], -self.direction_vec[:, np.newaxis]], axis=1)
            st = np.linalg.inv(matrix)@((self.point - axesPts[:,i])[:,np.newaxis])
            tVals[i] = st[1]
        self.t_bound = [np.nanmax(np.where(tVals<0, tVals, np.nan)), np.nanmin(np.where(tVals>0, tVals, np.nan))]
        print(self.t_bound)

        print(self.point + self.t_bound[0]*self.direction_vec)
        print(self.point + self.t_bound[1]*self.direction_vec)

    def show_line_on_img(self, img):
        plt.figure()
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        start = self.point + self.t_bound[0]*self.direction_vec
        end = self.point + self.t_bound[1]*self.direction_vec
        plt.plot([start[0], end[0]], [start[1], end[1]])
        plt.savefig('verify_lines.png')

    def get_xy_samples(self, n_samp=100):
        tVals = np.linspace(self.t_bound[0], self.t_bound[1], n_samp)[:,np.newaxis]
        return self.point + tVals*self.direction_vec


    def __str__(self):
        return f"point: ({self.point[0]:.2f},{self.point[1]:.2f}), orientation= {self.orientation:.2f}"


if __name__ == "__main__":
    import cv2
    img = cv2.imread('data_cleaned/mono.png', cv2.IMREAD_GRAYSCALE)
    img = img[0:128,:]
    print(img.shape)
    l = Line(np.array([20,40]), np.pi/4, img.shape)
    # l.show_line_on_img(img)
    print(l.get_xy_samples(5))