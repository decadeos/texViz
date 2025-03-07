import cv2
import matplotlib.pyplot as plt
from defs import *

## Image
I = cv2.imread("lab2/image.jpg")
I_rgb = convert_to_rgb(I)

## param sdvig
tx = 50
ty = 100
I_shift = sdvig(I, tx, ty)
I_shift_rgb = convert_to_rgb(I_shift)

## param reflect
I_reflect = reflect(I)
I_reflect_rgb = convert_to_rgb(I_reflect)




## show image
show(I_rgb, "original image")

## show sdvig image
show(I_shift_rgb, "shifted image")

## show reflect image
show(I_reflect_rgb, "reflected image")