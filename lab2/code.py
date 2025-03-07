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

## param reflect OX
I_reflect_OX = reflectOX(I)
I_reflect_OX_rgb = convert_to_rgb(I_reflect_OX)

## param reflect OY
I_reflect_OY = reflectOY(I)
I_reflect_OY_rgb = convert_to_rgb(I_reflect_OY)

## param scaling
scale_x = 1.7
scale_y = 0.5
I_scale = scaling(I, scale_x, scale_y)
I_scale_rgb = convert_to_rgb(I_scale)















## show image
show(I_rgb, "original image")

## show sdvig image
show(I_shift_rgb, "shifted image")

## show reflect by OX image
show(I_reflect_OX_rgb, "reflected by OX image")

## show reflect by OY image
show(I_reflect_OY_rgb, "reflected by OY image")

## show scaled image
show(I_scale_rgb, "scaled image")