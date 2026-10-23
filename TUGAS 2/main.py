import math

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


def terapkan_lut(gray, lut):
    # lut = tabel hasil transformasi buat tiap nilai intensitas 0-255
    tinggi, lebar = gray.shape
    hasil = np.zeros((tinggi, lebar), dtype=np.uint8)

    for i in range(tinggi):
        for j in range(lebar):
            hasil[i, j] = lut[gray[i, j]]

    return hasil


def negatif(gray):
    # s = (L - 1) - r
    lut = [0] * 256
    for r in range(256):
        lut[r] = 255 - r

    return terapkan_lut(gray, lut)


def log_transform(gray):
    # s = c * log(1 + r), c dipilih biar hasil maks tetap 255
    c = 255 / math.log(1 + 255)
    lut = [0] * 256
    for r in range(256):
        lut[r] = int(c * math.log(1 + r))

    return terapkan_lut(gray, lut)


def gamma_transform(gray, gamma):
    # s = c * r^gamma, r dinormalisasi dulu ke 0-1
    lut = [0] * 256
    for r in range(256):
        lut[r] = int(255 * (r / 255) ** gamma)

    return terapkan_lut(gray, lut)


def histogram(gray):
    tinggi, lebar = gray.shape
    hist = [0] * 256

    for i in range(tinggi):
        for j in range(lebar):
            hist[gray[i, j]] += 1

    return hist


def ekualisasi(gray):
    # s_k = (L - 1) * jumlah p(r_j) dari j=0 sampai k
    tinggi, lebar = gray.shape
    total = tinggi * lebar
    hist = histogram(gray)

    lut = [0] * 256
    cdf = 0
    for k in range(256):
        cdf += hist[k]
        lut[k] = round(255 * cdf / total)

    return terapkan_lut(gray, lut)


def gambar_histogram(hist, lebar, tinggi):
    canvas = np.zeros((tinggi, lebar), dtype=np.uint8)
    maks = max(hist)

    for k in range(256):
        x = int(k * lebar / 256)
        h = int(hist[k] / maks * (tinggi - 50)) # sisain atas buat teks
        cv2.line(canvas, (x, tinggi - 1), (x, tinggi - 1 - h), 255, 1)

    return canvas


def main():
    img = cv2.imread("./assets/image.png")

    if img is None:
        print("Image not loaded!")
        return

    # resize ke lebih kecil biar loop nya ga lama
    img = cv2.resize(img, (400, 266))

    gray = grayscale(img)
    neg = negatif(gray)
    log = log_transform(gray)
    gamma = gamma_transform(gray, 0.5) # gamma < 1 bikin lebih terang
    eq = ekualisasi(gray)

    hist_gray = gambar_histogram(histogram(gray), 400, 266)
    hist_eq = gambar_histogram(histogram(eq), 400, 266)

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (15, 30)
    color = 255

    # copy dulu biar teks nya ga ikut keitung di histogram/gambar lain
    gray_teks = gray.copy()
    cv2.putText(gray_teks, "Grayscale", position, font, 0.8, color, 2)
    cv2.putText(neg, "Negatif", position, font, 0.8, color, 2)
    cv2.putText(log, "Log", position, font, 0.8, color, 2)
    cv2.putText(gamma, "Gamma 0.5", position, font, 0.8, color, 2)
    cv2.putText(eq, "Ekualisasi", position, font, 0.8, color, 2)
    cv2.putText(hist_gray, "Histogram Grayscale", position, font, 0.8, color, 2)
    cv2.putText(hist_eq, "Histogram Ekualisasi", position, font, 0.8, color, 2)

    grid_transformasi = np.vstack((np.hstack((gray_teks, neg)), np.hstack((log, gamma))))
    grid_ekualisasi = np.vstack((np.hstack((gray_teks, eq)), np.hstack((hist_gray, hist_eq))))

    while True:
        cv2.imshow("Transformasi Intensitas", grid_transformasi)
        cv2.imshow("Ekualisasi Histogram", grid_ekualisasi)

        key = cv2.waitKey(0) & 0xFF

        if key == 32:  # Spasi
            cv2.imwrite("hasil_transformasi.png", grid_transformasi)
            cv2.imwrite("hasil_ekualisasi.png", grid_ekualisasi)
            print("ok")

        elif key == 27:  # Esc
            break
    cv2.destroyAllWindows()

main()
