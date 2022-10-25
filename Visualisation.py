import matplotlib.pyplot as plt

import numpy as np

class InsetBox:
    # given a set of corners, convert to the rows and cols that bound the box
    def __init__(self, corner_1, corner_2, name=None):
        self.r = [corner_1[0], corner_2[0]]
        self.c = [corner_1[1], corner_2[1]]
        
        self.r.sort()
        self.c.sort()

        self.name = name if name is not None else ''

    def rows(self):
        return slice(self.r[0], self.r[1])

    def cols(self):
        return slice(self.c[0], self.c[1])



if __name__ == "__main__":
    ib = InsetBox([2,5], [6,7])
    print(ib.rows(), ib.cols())

    img = np.zeros([50,50])
    print(img[ib.rows(), ib.cols()])
    print(img[2:6, 5:7])

    print(ib.name)