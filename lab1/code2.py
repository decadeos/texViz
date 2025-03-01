import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_brightness_profile(image, line_start, line_end):
    """
    Строит профиль яркости вдоль заданной линии на изображении.
    
    :param image: Изображение (в оттенках серого).
    :param line_start: Начальная точка линии (x1, y1).
    :param line_end: Конечная точка линии (x2, y2).
    """
    # Получаем размеры изображения
    height, width = image.shape[:2]
    
    # Вычисляем профиль яркости вдоль линии
    num_points = 1000  # Количество точек для профиля
    x = np.linspace(line_start[0], line_end[0], num_points)
    y = np.linspace(line_start[1], line_end[1], num_points)
    
    # Собираем значения яркости вдоль линии
    profile = [image[int(y[i]), int(x[i])] for i in range(num_points)]
    
    # Строим график профиля яркости
    plt.figure(figsize=(8, 4))
    plt.plot(profile, color='black')
    plt.title("Профиль яркости")
    plt.xlabel("Позиция вдоль линии")
    plt.ylabel("Яркость")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Загрузка изображения
image = cv2.imread('lab1/im2.jpg', cv2.IMREAD_GRAYSCALE)  # Загружаем как灰度 изображение
if image is None:
    print("Ошибка: Не удалось загрузить изображение. Проверьте путь к файлу.")
else:
    # Задаем начальную и конечную точки линии
    line_start = (50, 100)  # (x1, y1)
    line_end = (400, 300)   # (x2, y2)
    
    # Строим профиль яркости
    plot_brightness_profile(image, line_start, line_end)