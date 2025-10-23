import numpy as np
import cv2 # type: ignore


def load_and_display_img(path: str, 
                         gray: bool = False, 
                         sbs: bool = False, 
                         wait_time: int = 5000,
                         store_it:bool = False) -> None:
    graffiti_img = cv2.imread("Yann\lfi-01\graffiti.png")
    print("Shape of the normal image :",graffiti_img.shape)
    if gray:
        graffiti_img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print("Shape of the gray image :",graffiti_img_gray.shape)
        if store_it:
            cv2.imwrite(f"{path}_gray", graffiti_img_gray)
    
        # Two methods to convert a gray image to rvb : 
        graffiti_img_gray_rvb = np.concatenate((graffiti_img_gray[:,:,None], graffiti_img_gray[:,:,None], graffiti_img_gray[:,:,None]), axis = 2)
        graffiti_img_gray_rvb = cv2.cvtColor(graffiti_img_gray, cv2.COLOR_GRAY2BGR)
        if sbs: 
            graffiti_sbs = np.concatenate((graffiti_img, graffiti_img_gray_rvb), axis = 1)
            print("Shape of the gray image and the normal image side by side :",graffiti_sbs.shape)
            if store_it:
                cv2.imwrite('.\Yann\lfi-01\graffiti_concat.png', graffiti_sbs)

    cv2.imshow("Graffiti : First image loaded", graffiti_img)
    cv2.imshow("Graffiti", graffiti_img_gray_rvb)
    cv2.imshow("Graffiti sbs", graffiti_sbs)
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_and_display_img("Yann\lfi-01\graffiti.png", gray= True, sbs = True, wait_time=2000, store_it = True)