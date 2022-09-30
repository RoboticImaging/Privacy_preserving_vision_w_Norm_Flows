
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



if __name__ == "__main__":
    fname = "data/test.webp"

    output = "outputs/"

    img = np.array(cv2.imread(fname))
    crop = Image_Transform.centre_crop(img)
    print(crop.dtype)
    mono = Image_Transform.color2monochrome(crop)
    print(mono.shape)

    cv2.imwrite(output+"original.png", img)
    cv2.imwrite(output+"center.png", crop)

    cv2.imwrite(output+"mono.png", mono)

    img = np.array([[1,2,3,4],
                    [4,5,6,7]], dtype=np.uint8)
    imgStack = np.stack([img,img,img],axis=2)
    mono = Image_Transform.color2monochrome(imgStack)