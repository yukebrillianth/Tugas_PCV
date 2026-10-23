import math
import random

import cv2
import numpy as np


def grayscale(img):
    tinggi, lebar, _ = img.shape
    gray = np.zeros((tinggi, lebar), dtype=np.uint8)

    for i in range(tinggi):
        for j in range(lebar):
            b, g, r = img[i, j]
            gray[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)

    return gray


def salt_pepper(gray, persen):
    # kasih noise titik hitam putih random buat ngetes filter smoothing
    tinggi, lebar = gray.shape
    noise = gray.copy()
    jumlah = int(tinggi * lebar * persen)

    for _ in range(jumlah):
        i = random.randint(0, tinggi - 1)
        j = random.randint(0, lebar - 1)
        noise[i, j] = random.choice([0, 255])

    return noise


def konvolusi(gray, kernel):
    tinggi, lebar = gray.shape
    k = len(kernel) // 2
    hasil = np.zeros((tinggi, lebar), dtype=np.float64)

    for i in range(tinggi):
        for j in range(lebar):
            total = 0
            for u in range(-k, k + 1):
                for v in range(-k, k + 1):
                    # pinggiran gambar pakai piksel terdekat (replicate)
                    y = min(max(i + u, 0), tinggi - 1)
                    x = min(max(j + v, 0), lebar - 1)
                    # casting ke int dulu, uint8 dikali angka negatif bakal error
                    total += int(gray[y, x]) * kernel[u + k][v + k]
            hasil[i, j] = total

    return hasil


def clip(hasil):
    # hasil konvolusi bisa < 0 atau > 255, dibatasi dulu biar jadi uint8
    tinggi, lebar = hasil.shape
    out = np.zeros((tinggi, lebar), dtype=np.uint8)

    for i in range(tinggi):
        for j in range(lebar):
            s = hasil[i, j]
            if s < 0:
                s = 0
            elif s > 255:
                s = 255
            out[i, j] = int(s)

    return out


def mean_filter(gray):
    kernel = [[1 / 9, 1 / 9, 1 / 9],
              [1 / 9, 1 / 9, 1 / 9],
              [1 / 9, 1 / 9, 1 / 9]]

    return clip(konvolusi(gray, kernel))


def median_filter(gray, ukuran):
    tinggi, lebar = gray.shape
    k = ukuran // 2
    hasil = np.zeros((tinggi, lebar), dtype=np.uint8)

    for i in range(tinggi):
        for j in range(lebar):
            tetangga = []
            for u in range(-k, k + 1):
                for v in range(-k, k + 1):
                    y = min(max(i + u, 0), tinggi - 1)
                    x = min(max(j + v, 0), lebar - 1)
                    tetangga.append(gray[y, x])
            tetangga.sort()
            hasil[i, j] = tetangga[len(tetangga) // 2]

    return hasil


def sharpening(gray):
    # kernel laplacian yang udah ditambah gambar asli (tengahnya 4 + 1)
    kernel = [[0, -1, 0],
              [-1, 5, -1],
              [0, -1, 0]]

    return clip(konvolusi(gray, kernel))


def sobel(gray):
    kernel_x = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
    kernel_y = [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]]

    gx = konvolusi(gray, kernel_x)
    gy = konvolusi(gray, kernel_y)

    tinggi, lebar = gray.shape
    magnitude = np.zeros((tinggi, lebar), dtype=np.float64)
    for i in range(tinggi):
        for j in range(lebar):
            magnitude[i, j] = math.sqrt(gx[i, j] ** 2 + gy[i, j] ** 2)

    return clip(magnitude)


def main():
    img = cv2.imread("./assets/image.png")

    if img is None:
        print("Image not loaded!")
        return

    # resize ke lebih kecil biar loop nya ga lama
    img = cv2.resize(img, (400, 266))

    gray = grayscale(img)
    noise = salt_pepper(gray, 0.05) # 5% piksel jadi noise
    mean = mean_filter(noise)
    median = median_filter(noise, 3)
    sharp = sharpening(gray)
    edge = sobel(gray)

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (15, 30)
    color = 255

    cv2.putText(gray, "Grayscale", position, font, 0.8, color, 2)
    cv2.putText(noise, "Salt & Pepper", position, font, 0.8, color, 2)
    cv2.putText(mean, "Mean 3x3", position, font, 0.8, color, 2)
    cv2.putText(median, "Median 3x3", position, font, 0.8, color, 2)
    cv2.putText(sharp, "Sharpening", position, font, 0.8, color, 2)
    cv2.putText(edge, "Sobel", position, font, 0.8, color, 2)

    top_row = np.hstack((gray, noise, mean))
    bottom_row = np.hstack((median, sharp, edge))

    grid = np.vstack((top_row, bottom_row))
    while True:
        cv2.imshow("Filter Spasial", grid)

        key = cv2.waitKey(0) & 0xFF

        if key == 32:  # Spasi
            cv2.imwrite("hasil_filter.png", grid)
            print("ok")

        elif key == 27:  # Esc
            break
    cv2.destroyAllWindows()

main()
