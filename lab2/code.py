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






## show image
plt.imshow(I_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

## show sdvig image
plt.imshow(I_shift_rgb)
plt.title("Shifted Image")
plt.axis('off')
plt.show()