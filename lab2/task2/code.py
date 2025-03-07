import cv2
import matplotlib.pyplot as plt
from defs import *

img1 = cv2.imread("task2/podushka.png")
img2 = cv2.imread("task2/bochka.png")

img1_rgb = convert_to_rgb(img1)
img2_rgb = convert_to_rgb(img2)

operations = [
    ("Подушкообразная дистория", lambda img: podushka(img2)),
    ("Бочкообразная дистория", lambda img: bochka(img1)),
]

for title, operation in operations:
    result = operation(None)
    result_rgb = convert_to_rgb(result)
    show(result_rgb, title)