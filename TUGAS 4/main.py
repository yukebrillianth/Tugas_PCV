import math

import cv2
import numpy as np


def rgb_ke_cmyk(img):
    tinggi, lebar, _ = img.shape
    c = np.zeros((tinggi, lebar), dtype=np.uint8)
    m = np.zeros((tinggi, lebar), dtype=np.uint8)
    y = np.zeros((tinggi, lebar), dtype=np.uint8)
    k = np.zeros((tinggi, lebar), dtype=np.uint8)

    for i in range(tinggi):
        for j in range(lebar):
            b, g, r = img[i, j]
            # normalisasi ke 0-1
            r, g, b = r / 255, g / 255, b / 255

            kk = 1 - max(r, g, b)
            if kk == 1:  # piksel hitam, C M Y nya 0 semua
                cc = mm = yy = 0
            else:
                cc = (1 - r - kk) / (1 - kk)
                mm = (1 - g - kk) / (1 - kk)
                yy = (1 - b - kk) / (1 - kk)

            c[i, j] = int(cc * 255)
            m[i, j] = int(mm * 255)
            y[i, j] = int(yy * 255)
            k[i, j] = int(kk * 255)

    return c, m, y, k


def rgb_ke_hsi(img):
    tinggi, lebar, _ = img.shape
    h = np.zeros((tinggi, lebar), dtype=np.uint8)
    s = np.zeros((tinggi, lebar), dtype=np.uint8)
    inten = np.zeros((tinggi, lebar), dtype=np.uint8)

    for i in range(tinggi):
        for j in range(lebar):
            b, g, r = img[i, j]
            r, g, b = r / 255, g / 255, b / 255

            ii = (r + g + b) / 3

            if r + g + b == 0:
                ss = 0
            else:
                ss = 1 - 3 * min(r, g, b) / (r + g + b)

            atas = 0.5 * ((r - g) + (r - b))
            bawah = math.sqrt((r - g) ** 2 + (r - b) * (g - b))
            if bawah == 0:  # abu-abu, hue nya ga ada
                hh = 0
            else:
                # dibatasi -1 sampai 1 soalnya float kadang lewat dikit
                theta = math.degrees(math.acos(max(-1, min(1, atas / bawah))))
                hh = theta if b <= g else 360 - theta

            # hue 0-360 derajat diskalain ke 0-255 biar bisa ditampilin
            h[i, j] = int(hh / 360 * 255)
            s[i, j] = int(ss * 255)
            inten[i, j] = int(ii * 255)

    return h, s, inten


def rgb_ke_hsv(img):
    tinggi, lebar, _ = img.shape
    h = np.zeros((tinggi, lebar), dtype=np.uint8)
    s = np.zeros((tinggi, lebar), dtype=np.uint8)
    v = np.zeros((tinggi, lebar), dtype=np.uint8)

    for i in range(tinggi):
        for j in range(lebar):
            b, g, r = img[i, j]
            r, g, b = r / 255, g / 255, b / 255

            maks = max(r, g, b)
            minim = min(r, g, b)
            delta = maks - minim

            if delta == 0:
                hh = 0
            elif maks == r:
                hh = 60 * (((g - b) / delta) % 6)
            elif maks == g:
                hh = 60 * ((b - r) / delta + 2)
            else:
                hh = 60 * ((r - g) / delta + 4)

            ss = 0 if maks == 0 else delta / maks

            h[i, j] = int(hh / 360 * 255)
            s[i, j] = int(ss * 255)
            v[i, j] = int(maks * 255)

    return h, s, v


def ke_bgr(gray):
    # channel tunggal dijadikan 3 channel biar bisa distack sama gambar asli
    tinggi, lebar = gray.shape
    hasil = np.zeros((tinggi, lebar, 3), dtype=np.uint8)
    hasil[:, :, 0] = gray
    hasil[:, :, 1] = gray
    hasil[:, :, 2] = gray
    return hasil


def main():
    img = cv2.imread("./assets/image.png")

    if img is None:
        print("Image not loaded!")
        return

    # resize ke lebih kecil biar loop nya ga lama
    img = cv2.resize(img, (400, 266))

    red = img[:, :, 2].copy()
    green = img[:, :, 1].copy()
    blue = img[:, :, 0].copy()
    c, m, y, k = rgb_ke_cmyk(img)
    h_hsi, s_hsi, i_hsi = rgb_ke_hsi(img)
    h_hsv, s_hsv, v_hsv = rgb_ke_hsv(img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (15, 30)
    color = (255, 255, 255)

    original = img.copy()
    cv2.putText(original, "Original Lur", position, font, 0.8, color, 2)

    gambar = [
        (red, "R"), (green, "G"), (blue, "B"),
        (c, "C"), (m, "M"), (y, "Y"), (k, "K"),
        (h_hsi, "H (HSI)"), (s_hsi, "S (HSI)"), (i_hsi, "I (HSI)"),
        (h_hsv, "H (HSV)"), (s_hsv, "S (HSV)"), (v_hsv, "V (HSV)"),
    ]
    for g, teks in gambar:
        cv2.putText(g, teks, position, font, 0.8, 255, 2)

    grid_rgb = np.vstack((np.hstack((original, ke_bgr(red))), np.hstack((ke_bgr(green), ke_bgr(blue)))))
    grid_cmyk = np.vstack((np.hstack((ke_bgr(c), ke_bgr(m))), np.hstack((ke_bgr(y), ke_bgr(k)))))
    grid_hsi = np.vstack((np.hstack((original, ke_bgr(h_hsi))), np.hstack((ke_bgr(s_hsi), ke_bgr(i_hsi)))))
    grid_hsv = np.vstack((np.hstack((original, ke_bgr(h_hsv))), np.hstack((ke_bgr(s_hsv), ke_bgr(v_hsv)))))

    while True:
        cv2.imshow("RGB", grid_rgb)
        cv2.imshow("CMYK", grid_cmyk)
        cv2.imshow("HSI", grid_hsi)
        cv2.imshow("HSV", grid_hsv)

        key = cv2.waitKey(0) & 0xFF

        if key == 32:  # Spasi
            cv2.imwrite("hasil_rgb.png", grid_rgb)
            cv2.imwrite("hasil_cmyk.png", grid_cmyk)
            cv2.imwrite("hasil_hsi.png", grid_hsi)
            cv2.imwrite("hasil_hsv.png", grid_hsv)
            print("ok")

        elif key == 27:  # Esc
            break
    cv2.destroyAllWindows()

main()
