import unittest

from hashes.Circle import Circle
import numpy as np

class TestCircleOutputs(unittest.TestCase):
    def test_output_size(self):
        c = Circle(np.array([0,0]),1.)
        self.assert_(c.get_xy_samples(100).shape == (100,2))
        self.assert_(c.get_xy_samples(200).shape == (200,2))

    def test_radius(self):
        pass

    def test_circle_coverage(self):
        pass


if __name__ == '__main__':
    unittest.main()
