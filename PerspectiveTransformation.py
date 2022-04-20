import cv2
import numpy as np

class PerspectiveTransformation:
    def __init__(self):
        self.src = np.float32([(500, 460),     # top-left
                            (150, 720),     # bottom-left
                            (1200, 720),    # bottom-right
                            (770, 400)])    # top-right
        self.dst = np.float32([(100, 0),
                            (100, 520),
                            (1200, 720),
                            (1100, 0)])

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)



    def forward(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        
        return cv2.warpPerspective(img, self.M, img_size, flags=flags)