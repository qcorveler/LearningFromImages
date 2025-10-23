import numpy as np
import cv2 # type: ignore

cap = cv2.VideoCapture(0)
mode = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Wait for key press and switch mode
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('1'):
        mode = 1  # Gaussian Blur
    elif ch == ord('2'):
        mode = 2  # Grayscale
    elif ch == ord('3'):
        mode = 3  # FFT Magnitude Spectrum
    elif ch == ord('q'):
        break

    # Apply selected mode
    if mode == 1:
        frame_processed = cv2.GaussianBlur(frame, (5, 5), 0)
    elif mode == 2:
        frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif mode == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        frame_processed = cv2.convertScaleAbs(magnitude_spectrum)
    else:
        frame_processed = frame

    # Display the resulting frame
    cv2.imshow('frame', frame_processed)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
