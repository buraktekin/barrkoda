import cv2
import matplotlib.pyplot as plt
import numpy as np

class Detector:
    def __init__(self, frame):
        self.frame = frame
        self.scale = 800
        self.image = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
        self.image_out = frame # will be fixed

    def scale_image(self):
        self.scale = 800.0 / self.image.shape[1]
        self.image = cv2.resize(self.image,
            (int(self.image.shape[1] * self.scale), 
            int(self.image.shape[0] * self.scale)))
        return self.image
        
    def morphological_transformation(self):
        # blackhat transformation
        kernel = np.ones((1, 3), np.uint8)
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_BLACKHAT, kernel, anchor=(1, 0))
        thresh, self.image = cv2.threshold(self.image, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((1, 5), np.uint8) # set the kernel for convolution
        # dilation
        self.image = cv2.morphologyEx(
            self.image,
            cv2.MORPH_DILATE, 
            kernel, 
            anchor=(2, 0), 
            iterations=2
        )
        # closing
        self.image = cv2.morphologyEx(
            self.image,
            cv2.MORPH_CLOSE, 
            kernel, 
            anchor=(2, 0), 
            iterations=2
        )
        # Opening
        kernel = np.ones((21, 35), np.uint8)
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel, iterations=1)
        return self.image

    def set_contours(self):
        contours, hierarchy = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        unscale = 1.0 / self.scale
        if contours != None:
            for contour in contours:
                if cv2.contourArea(contour) <= 2000:
                    continue
                rect = cv2.minAreaRect(contour)
                rect = (
                    (int(rect[0][0] * unscale), int(rect[0][1] * unscale)),
                    (int(rect[1][0] * unscale), int(rect[1][1] * unscale)), 
                    rect[2]
                )
                box = np.int0(cv2.cv2.BoxPoints(rect))
                cv2.drawContours(self.image_out, [box], 0, (0, 255, 0), thickness=2)
        return self.image_out

    def detect(self):
        self.scale_image()
        self.morphological_transformation()
        self.set_contours()
        plt.imshow(self.image_out)
        cv2.imwrite(r'./out.png', self.image_out)
        return self.image_out