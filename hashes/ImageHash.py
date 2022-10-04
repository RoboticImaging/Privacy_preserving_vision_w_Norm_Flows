

class ImageHash:
    def __init__(self):
        pass

    def compute_features(img):
        # output a n_features x feature_size matrix
        # FIXED

        # FILL: generate x,y to sample
        # FILL: sample and take max min as needed
        pass

    def get_xy_to_sample():
        raise NotImplementedError()



class CircleHash(ImageHash):
    def __init__(self):
        super().__init__()
        self.circle_centres, self.circle_radii = self.get_circle_params()
    
    def get_circle_params(img_shape, n_features):
        # given an image shape, return the centre and radii of the circles
        pass


class LineHash(ImageHash):
    def __init__(self):
        super().__init__()


class FixedCircleHash(CircleHash):
    def __init__(self):
        super().__init__()
        # FILL: fix the circles and radii 
        

class RandomCircleHash(CircleHash):
    def __init__(self):
        super().__init__()
    
    def get_xy_to_sample():
        # shuffle the circles 
        # generate x,y to sample
        pass
