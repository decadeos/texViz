import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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

def rotate(image, phi):
    rows, cols = size(image)
    phi = math.radians(phi)
    T = np.float32([[ math.cos(phi), -math.sin(phi), 0], [math.sin(phi), math.cos(phi), 0]])
    I_rotate = cv2.warpAffine(image, T, (cols, rows))
    return I_rotate

def affine_transform(image, pts_src, pts_dst):
    rows, cols = size(image)
    T = cv2.getAffineTransform(pts_src, pts_dst)
    affine_image = cv2.warpAffine(image, T, (cols, rows))
    return affine_image

def bevel(image, skos):
    rows, cols = size(image)
    T = np.float32([[1, skos, 0], [0 ,1, 0]])
    I_bevel = cv2.warpAffine(image, T, (cols, rows))
    return I_bevel

def piecewiselinear(image, stretch):
    rows, cols = size(image)
    T = np.float32([[stretch, 0, 0], [0, 1, 0]])
    I_piecewiselinear = image.copy()
    I_piecewiselinear[:, int(cols/2):, :] = cv2.warpAffine(I_piecewiselinear[:, int(cols/2):, :], T, (cols - int(cols/2), rows))
    return I_piecewiselinear
