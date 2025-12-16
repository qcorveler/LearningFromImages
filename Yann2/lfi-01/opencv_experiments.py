import numpy as np
import cv2

def process_img(img, mode, want_show:bool=False):
    
# Question a

    if mode == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if mode == 'LAB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if mode == 'YUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
# Question b
    if mode in ['gaussian', 'otsu', 'Canny']:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive Gaussian Thresholding
    if mode == 'gaussian':
        img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 5, 2)
    if mode == 'otsu':
        # Otsu Thresholding
        _, img = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Question c
    if mode == 'Canny':
        img = cv2.Canny(gray, 100, 200)
    

    # Affichage optionnel
    if want_show:
        cv2.imshow(mode, img)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
    
    return img


def main():
    cap = cv2.VideoCapture(0)
    # Capture frame-by-frame
    mode = 'normal'  # Mode par défaut

    print("""Touches :\n1 - Flou gaussien\n2 - Niveaux de gris\n3 - Spectre FFT\nh - HSV\nl - LAB\ny - YUV\ng - Seuillage adaptatif (gaussian)\no - Seuillage d'Otsu\nc - Contours Canny\nq - Quitter\n    """)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ch = cv2.waitKey(1) & 0xFF

        # Gestion des touches
        if ch == ord('1'):
            mode = 'blur'
        elif ch == ord('2'):
            mode = 'gray'
        elif ch == ord('3'):
            mode = 'fft'
        elif ch == ord('h'):
            mode = 'HSV'
        elif ch == ord('l'):
            mode = 'LAB'
        elif ch == ord('y'):
            mode = 'YUV'
        elif ch == ord('g'):
            mode = 'gaussian'
        elif ch == ord('o'):
            mode = 'otsu'
        elif ch == ord('c'):
            mode = 'Canny'
        elif ch == ord('q'):
            break

        # Application du mode sélectionné
        if mode == 'blur':
            frame_processed = cv2.GaussianBlur(frame, (5, 5), 0)
        elif mode == 'gray':
            frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif mode == 'fft':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
            frame_processed = cv2.convertScaleAbs(magnitude_spectrum)
        elif mode in ['HSV', 'LAB', 'YUV', 'gaussian', 'otsu', 'Canny']:
            frame_processed = process_img(frame, mode)
        else:
            frame_processed = frame

        cv2.imshow('Webcam Processing', frame_processed)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
#     # Wait for key press and switch mode
#     ch = cv2.waitKey(1) & 0xFF
#     if ch == ord('1'):
#         mode = 1  # Gaussian Blur
#     elif ch == ord('2'):
#         mode = 2  # Grayscale
#     elif ch == ord('3'):
#         mode = 3  # FFT Magnitude Spectrum
#     elif ch == ord('h'):
#         mode = 4
#     elif ch == ord('l') :
#         mode = 5

#     elif ch == ord('q'):
#         break

#     # Apply selected mode
#     if mode == 1:
#         frame_processed = cv2.GaussianBlur(frame, (5, 5), 0)
#     elif mode == 2:
#         frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     elif mode == 3:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         f = np.fft.fft2(gray)
#         fshift = np.fft.fftshift(f)
#         magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
#         frame_processed = cv2.convertScaleAbs(magnitude_spectrum)
#     else:
#         frame_processed = frame

#     # Display the resulting frame
#     cv2.imshow('frame', frame_processed)

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
