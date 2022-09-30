
import numpy as np
import cv2

class Image_Transform:
    def centre_crop(img):
        center = Image_Transform.calc_img_center(img)
        dim = min(center)
        x = center[1] + dim*np.array([-1,1])
        y = center[0] + dim*np.array([-1,1])

        return img[y[0]:y[1], x[0]:x[1],:]
    
    def calc_img_center(img):
        return np.array(img.shape[:2])//2
    
    def color2monochrome(img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    def resize(img, new_size):
        return cv2.resize(img, new_size)



if __name__ == "__main__":
    fname = "data/test.webp"

    output = "outputs/"

    img = np.array(cv2.imread(fname))
    crop = Image_Transform.centre_crop(img)
    mono = Image_Transform.color2monochrome(crop)
    downsize = Image_Transform.resize(mono,[64,64])

    cv2.imwrite(output+"original.png", img)
    cv2.imwrite(output+"center.png", crop)

    cv2.imwrite(output+"mono.png", mono)

    cv2.imwrite(output+"downsize.png", downsize)