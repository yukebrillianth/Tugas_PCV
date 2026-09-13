import cv2
import numpy as np


def main():
    img = cv2.imread("./assets/image.png")

    if img is None:
        print("Image not loaded!")
        return

    # resize ke lebih kecil
    img = cv2.resize(img,(400, 266))

    red = img.copy()
    red[:,:,0] = 0 # Blue
    red[:,:,1] = 0 # Green

    blue = img.copy()
    blue[:,:,1] = 0 # Green
    blue[:,:,2] = 0 # Red

    green = img.copy()
    green[:,:,0] = 0 # Blue
    green[:,:,2] = 0 # Red

    # cv2.imshow("Original Image", img)
    # cv2.imshow("Red Filtered Image", red)
    # cv2.imshow("Green Filtered Image", green)
    # cv2.imshow("Blue Filtered Image", blue)

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (15,30)
    color = (255,255,255)

    cv2.putText(img, "Original Lur", position, font, 0.8, color, 2)
    cv2.putText(red, "Red Channel", position, font, 0.8, color, 2)
    cv2.putText(green, "Green Channel", position, font, 0.8, color, 2)
    cv2.putText(blue, "Blue Channel", position, font, 0.8, color, 2)

    top_row = np.hstack((img, red))
    bottom_row = np.hstack((green,blue))

    grid = np.vstack((top_row, bottom_row))
    while True:
        cv2.imshow("Hasil", grid)

        key = cv2.waitKey(0) & 0xFF

        if key == 32:  # Spasi
            cv2.imwrite("hasil_grid.png", grid)
            print("ok")

        elif key == 27:  # Esc
            break
    cv2.destroyAllWindows()

main()