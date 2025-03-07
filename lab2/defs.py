import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def size(image):
    rows, cols = image.shape[0:2]
    return rows, cols

def sdvig(image, tx, ty):
    rows, cols = size(image)
    T = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted_image = cv2.warpAffine(image, T, (cols, rows))
    return shifted_image

def reflectOX(image):
    rows, cols = size(image)
    T = np.float32([[1 , 0 , 0] ,[0, -1, rows - 1]])
    I_reflect = cv2.warpAffine(image, T, (cols, rows))
    return I_reflect

def reflectOY(image):
    rows, cols = size(image)
    T = np.float32([[-1, 0, cols - 1], [0, 1, 0]])
    I_reflect = cv2.warpAffine(image, T, (cols, rows))
    return I_reflect

def scaling(image, scale_x, scale_y):
    rows, cols = size(image)
    T = np.float32([[scale_x, 0, 0] ,[0, scale_y, 0]])
    I_scale = cv2.warpAffine(image, T, (int(cols * scale_x), int(rows*scale_y)))
    return I_scale
