import cv2
import numpy as np
import matplotlib.pyplot as plt

def sdvig(image, tx, ty):
    rows, cols = image.shape[0:2]
    T = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted_image = cv2.warpAffine(image, T, (cols, rows))
    return shifted_image

def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)