
import numpy as np

class Image_Transform:
    def centre_crop(img):
        center = Image_Transform.calc_img_center(img)
        dim = min(center)
        x = center[1] + dim*np.array([-1,1])
        y = center[0] + dim*np.array([-1,1])

        return img[y[0]:y[1], x[0]:x[1],:]
    
    def calc_img_center(img):
        return np.array(img.shape[:2])//2



if __name__ == "__main__":
    import cv2

    fname = "data/test.webp"

    output = "outputs"

    img = np.array(cv2.imread(fname))
    

    cv2.imwrite(output+"original.png", img)
    cv2.imwrite(output+"center.png", Image_Transform.centre_crop(img))
