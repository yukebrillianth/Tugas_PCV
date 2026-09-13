import time

import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(2) # ini idx camera di macbook

    if not cap.isOpened():
        print("Kamera gagal dibuka!")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255,255,255)
    position = (15,30)

    frame_width = 800
    frame_height = 450
    fps = 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('hasil_video.mp4', fourcc, fps, (frame_width, frame_height))

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Gagal!!!!!")
            return

        frame = cv2.resize(frame, (400, 225))

        red = frame.copy()
        red[:,:,0] = 0 # Blue
        red[:,:,1] = 0 # Green
    
        blue = frame.copy()
        blue[:,:,1] = 0 # Green
        blue[:,:,2] = 0 # Red
    
        green = frame.copy()
        green[:,:,0] = 0 # Blue
        green[:,:,2] = 0 # Red

        cv2.putText(frame, "Original Lur", position, font, 0.8, color, 2)
        cv2.putText(red, "Red Channel", position, font, 0.8, color, 2)
        cv2.putText(green, "Green Channel", position, font, 0.8, color, 2)
        cv2.putText(blue, "Blue Channel", position, font, 0.8, color, 2)
    
        top_row = np.hstack((frame, red))
        bottom_row = np.hstack((green,blue))
    
        grid = np.vstack((top_row, bottom_row))
        out.write(grid)
        cv2.imshow('Hasil', grid)

        elapsed_time = time.time() - start_time
        if elapsed_time >= 5:
            print(f"Selesai! Video berhasil disimpan.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

main()
