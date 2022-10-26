import matplotlib.pyplot as plt

import numpy as np

class InsetBox:
    # given a set of corners, convert to the rows and cols that bound the box
    def __init__(self, corner_1, corner_2, name=None):
        self.r = np.array([corner_1[0], corner_2[0]])
        self.c = np.array([corner_1[1], corner_2[1]])
        
        self.r.sort()
        self.c.sort()
        print(self)

        self.name = name if name is not None else ''

    def rows(self):
        return slice(int(self.r[0]), int(self.r[1]))

    def cols(self):
        return slice(int(self.c[0]), int(self.c[1]))
    
    def downsize(self, scale_factor):
        self.r = self.r.astype(float)/scale_factor
        self.c = self.c.astype(float)/scale_factor
        print("after", self)

    def __str__(self) -> str:
        return f'r = {self.r}, c = {self.c}'


if __name__ == "__main__":
    ib = InsetBox([2,5], [6,7])
    print(ib.rows(), ib.cols())

    img = np.zeros([50,50])
    print(img[ib.rows(), ib.cols()])
    print(img[2:6, 5:7])

    print(ib.name)