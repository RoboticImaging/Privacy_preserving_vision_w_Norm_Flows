import unittest

from data_manipulation.ImageTransform import Image_Transform
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
        np.testing.assert_array_equal(crop, img)

        
        img = np.array([[1,2,3,4],
                        [5,6,7,8]])
        target = np.array([[2,3],
                           [6,7]])
        img = np.stack([img,img,img],axis=2)
        target = np.stack([target,target,target],axis=2)
        crop = Image_Transform.centre_crop(img)
        np.testing.assert_array_equal(crop, target)

        
        img = np.array([[1,2,3,4],
                        [5,6,7,8],
                        [9,10,11,12]])
        target = np.array([[2,3],
                           [6,7]])
        img = np.stack([img,img,img],axis=2)
        target = np.stack([target,target,target],axis=2)
        crop = Image_Transform.centre_crop(img)
        np.testing.assert_array_equal(crop, target)
        
        img = np.array([[1,2,3],
                        [5,6,7],
                        [9,10,11]])
        target = np.array([[1,2],
                           [5,6]])
        img = np.stack([img,img,img],axis=2)
        target = np.stack([target,target,target],axis=2)
        crop = Image_Transform.centre_crop(img)
        np.testing.assert_array_equal(crop, target)


class TestImageTransformCenter(unittest.TestCase):
    def test_output_size(self):
        img = np.array([[1,2],
                        [4,5]])
        img = np.stack([img,img,img],axis=2)
        c = Image_Transform.calc_img_center(img)
        np.testing.assert_array_equal(c, (1,1))
        
        img = np.array([[1,2,3],
                        [4,5,6]])
        img = np.stack([img,img,img],axis=2)
        c = Image_Transform.calc_img_center(img)
        np.testing.assert_array_equal(c, (1,1))
        
        img = np.array([[1,2,3,4],
                        [4,5,6,7]])
        img = np.stack([img,img,img],axis=2)
        c = Image_Transform.calc_img_center(img)
        np.testing.assert_array_equal(c, (1,2))


class TestImageTransformMonochrome(unittest.TestCase):
    def test_output(self):
        img = np.array([[1,2,3,4],
                        [4,5,6,7]], dtype=np.uint8)
        imgStack = np.stack([img,img,img],axis=2)
        output = Image_Transform.color2monochrome(imgStack)
        np.testing.assert_array_equal(output, img)


if __name__ == '__main__':
    unittest.main()
