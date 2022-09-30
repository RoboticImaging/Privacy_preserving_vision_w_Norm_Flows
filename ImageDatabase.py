

import os

class Image_Database:
    # reads images recursively 
    def __init__(self, input_path, output_path):
        assert(os.path.exists(input_path))

        self.imgIdx = 0
        self.input_path = input_path
        self.output_path = output_path
    
    def transform_and_save(self, list_of_transforms):
        for subdir, dirs, files in os.walk(self.input_path):
            for file in files:
                img = np.array(cv2.imread(os.path.join(subdir, file)))
                for transform in list_of_transforms:
                    



if __name__ == "__main__":
    id = Image_Database('data\Extracted', 'outputs\LSUN')
    id.transform_and_save([])

