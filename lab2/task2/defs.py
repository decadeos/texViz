import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

## функции для удобства
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

## дистории
def podushka(image):
   

def bochka(image):
    
