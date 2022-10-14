import unittest

from hashes.Circle import Circle
import numpy as np

class TestCircleOutputs(unittest.TestCase):
    def test_output_size(self):
        c = Circle(np.array([0,0]),1., [256,256])
        self.assert_(c.get_xy_samples(100).shape == (100,2))
        self.assert_(c.get_xy_samples(200).shape == (200,2))

    def test_radius(self):            
        # moved circ
        c = Circle(np.array([10,20]),2.5, [256,256])
        
        samples = c.get_xy_samples(100)

        for sampleIdx in range(samples.shape[0]):
            np.testing.assert_almost_equal(np.sqrt((samples[sampleIdx, 0]-10)**2 + (samples[sampleIdx, 1]-20)**2), 2.5)

    def test_circle_coverage(self):
        x = 10.
        y = 20.
        r = 3.

        c = Circle(np.array([x,y]),r, [256,256])
        samples = c.get_xy_samples(4)

        np.testing.assert_almost_equal(samples[0,:], np.array([x+r,y]))
        np.testing.assert_almost_equal(samples[1,:], np.array([x,y+r]))
        np.testing.assert_almost_equal(samples[2,:], np.array([x-r,y]))
        np.testing.assert_almost_equal(samples[3,:], np.array([x,y-r]))



if __name__ == '__main__':
    unittest.main()
