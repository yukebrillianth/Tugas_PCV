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


def threshold(gray, batas):
    tinggi, lebar = gray.shape
    biner = np.zeros((tinggi, lebar), dtype=np.uint8)

    for i in range(tinggi):
        for j in range(lebar):
            if gray[i, j] > batas:
                biner[i, j] = 255

    return biner


def erosi(biner, se):
    # piksel tetap putih kalau SEMUA tetangga di bawah SE putih
    tinggi, lebar = biner.shape
    k = len(se) // 2
    hasil = np.zeros((tinggi, lebar), dtype=np.uint8)

    for i in range(tinggi):
        for j in range(lebar):
            semua_putih = True
            for u in range(-k, k + 1):
                for v in range(-k, k + 1):
                    if se[u + k][v + k] == 0:
                        continue
                    # pinggiran gambar pakai piksel terdekat (replicate)
                    y = min(max(i + u, 0), tinggi - 1)
                    x = min(max(j + v, 0), lebar - 1)
                    if biner[y, x] == 0:
                        semua_putih = False
            if semua_putih:
                hasil[i, j] = 255

    return hasil


def dilasi(biner, se):
    # piksel jadi putih kalau ADA tetangga di bawah SE yang putih
    tinggi, lebar = biner.shape
    k = len(se) // 2
    hasil = np.zeros((tinggi, lebar), dtype=np.uint8)

    for i in range(tinggi):
        for j in range(lebar):
            ada_putih = False
            for u in range(-k, k + 1):
                for v in range(-k, k + 1):
                    if se[u + k][v + k] == 0:
                        continue
                    y = min(max(i + u, 0), tinggi - 1)
                    x = min(max(j + v, 0), lebar - 1)
                    if biner[y, x] == 255:
                        ada_putih = True
            if ada_putih:
                hasil[i, j] = 255

    return hasil


def opening(biner, se):
    # erosi dulu baru dilasi, buat ngilangin bintik putih kecil
    return dilasi(erosi(biner, se), se)


def closing(biner, se):
    # dilasi dulu baru erosi, buat nutup lubang hitam kecil
    return erosi(dilasi(biner, se), se)


def main():
    img = cv2.imread("./assets/image.png")

    if img is None:
        print("Image not loaded!")
        return

    # resize ke lebih kecil biar loop nya ga lama
    img = cv2.resize(img, (400, 266))

    se = [[1, 1, 1],
          [1, 1, 1],
          [1, 1, 1]] # structuring element kotak 3x3

    gray = grayscale(img)
    biner = threshold(gray, 127)
    ero = erosi(biner, se)
    dil = dilasi(biner, se)
    ope = opening(biner, se)
    clo = closing(biner, se)

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (15, 30)
    color = 127 # abu2 biar keliatan di area putih maupun hitam

    cv2.putText(gray, "Grayscale", position, font, 0.8, color, 2)
    cv2.putText(biner, "Biner", position, font, 0.8, color, 2)
    cv2.putText(ero, "Erosi", position, font, 0.8, color, 2)
    cv2.putText(dil, "Dilasi", position, font, 0.8, color, 2)
    cv2.putText(ope, "Opening", position, font, 0.8, color, 2)
    cv2.putText(clo, "Closing", position, font, 0.8, color, 2)

    top_row = np.hstack((gray, biner, ero))
    bottom_row = np.hstack((dil, ope, clo))

    grid = np.vstack((top_row, bottom_row))
    while True:
        cv2.imshow("Morphology", grid)

        key = cv2.waitKey(0) & 0xFF

        if key == 32:  # Spasi
            cv2.imwrite("hasil_morphology.png", grid)
            print("ok")

        elif key == 27:  # Esc
            break
    cv2.destroyAllWindows()

main()
