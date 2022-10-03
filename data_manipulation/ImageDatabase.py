

import os
import cv2
import numpy as np
from ImageTransform import Image_Transform

class Image_Database:
    # reads images recursively 
    def __init__(self, input_path, output_path):
        assert(os.path.exists(input_path))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.imgIdx = 0
        self.input_path = input_path
        self.output_path = output_path
    
    def transform_and_save(self, list_of_transforms, output_ftype = 'png'):
        imgIdx = 0
        for subdir, dirs, files in os.walk(self.input_path):
            for file in files:
                img = np.array(cv2.imread(os.path.join(subdir, file)))
                for transform in list_of_transforms:
                    img = transform(img)
                cv2.imwrite(os.path.join(self.output_path, f"{imgIdx}.{output_ftype}"), img)
                imgIdx += 1

                if imgIdx % 1e4 == 0:
                    print(f"Transformed {imgIdx} images")
                



if __name__ == "__main__":
    id = Image_Database('../data/Extracted', '../outputs/LSUN_Bedroom/16x16')

    new_size = 16

    id.transform_and_save([
        Image_Transform.centre_crop,
        Image_Transform.color2monochrome,
        lambda img: Image_Transform.resize(img,[new_size, new_size])
    ])

