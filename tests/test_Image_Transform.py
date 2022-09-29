import unittest

from ImageTransform import Image_Transform
import numpy as np

class TestImageTransformCrop(unittest.TestCase):
    def test_output_size(self):
        # pick some odd and even sizes and make sure the ouput is always the size 
        # of the smallest dimension

        img = np.zeros([50,50,3])
        crop = Image_Transform.centre_crop(img)
        self.assertEqual(crop.shape, (50,50,3))
        
        img = np.zeros([50,55,3])
        crop = Image_Transform.centre_crop(img)
        self.assertEqual(crop.shape, (50,50,3))

        img = np.zeros([55,50,3])
        crop = Image_Transform.centre_crop(img)
        self.assertEqual(crop.shape, (50,50,3))
        
        img = np.zeros([55,53,3])
        crop = Image_Transform.centre_crop(img)
        self.assertEqual(crop.shape, (52,52,3))

    def test_image_subset(self):
        # img = np.array([[1,2,3],
        #                 [4,5,6],
        #                 [7,8,9]])
        img = np.array([[1,2],
                        [4,5]])
        img = np.stack([img,img,img],axis=2)
        crop = Image_Transform.centre_crop(img)
        self.assert_(np.array_equal(crop, img))




if __name__ == '__main__':
    unittest.main()
