import cv2
import numpy as np
import matplotlib.pyplot as plt
from defs import *
import os

# Путь к файлу
image_path = os.path.join(os.path.dirname(__file__), "podushka.png")
I = cv2.imread(image_path)
I_rgb = convert_to_rgb(I)

I_barrel = correct_distortion(I_rgb, 0.1, 0.12, "barrel")
I_pincushion = correct_distortion(I_rgb, 0.1, 0.12, "pincushion")

show(I_rgb, "Original Image")
show(I_barrel, "Barrel Distortion Correction")
show(I_pincushion, "Pincushion Distortion Correction")