from skimage import color, io
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
import time
import cv2
import os 
import numpy as np

def histogram_stretch(img_org):
    """
    Stretches the histogram of an image 
    :param img_in: Input image
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
    """
    #Convert to float
    img_float = img_as_float(img_org)
    #Stretch the histogram
    img_float_strecthed = 1/(np.max(img_float)-np.min(img_float)) * (img_float-np.min(img_float))
    #Convert to byte and return
    return img_as_ubyte(img_float_strecthed)

def gamma_map(img,gamma):
    img_float = img_as_float(img)
    img_float_gamma = img_float**gamma
    img_ubyte_gamma = img_as_ubyte(img_float_gamma)
    return img_ubyte_gamma

def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)


def process_gray_image(img):
    """
    Do a simple processing of an input gray scale image and return the processed image.
    # https://scikit-image.org/docs/stable/user_guide/data_types.html#image-processing-pipeline
    """
    img_proc = histogram_stretch(img)
    img_proc = gamma_map(img_proc, 2)
    return img_as_ubyte(img_proc)


def detect_plants(img):
    """
    Simple processing of a color (RGB) image
    """
    # Copy the image information so we do not change the original image
    proc_img = img.copy()
    proc_img = color.rgb2hsv(proc_img)
    hcomp = proc_img[:, :, 0]
    scomp = proc_img[:, :, 1]
    vcomp = proc_img[:, :, 2]
    plant_mask = (hcomp > 0.2) & (hcomp < 0.3) & (scomp > 0.6)
    return plant_mask


def capture_from_camera_and_show_images():
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = True # Use phone cam (true, or false for computer cam)
    if use_droid_cam:
        url = "http://10.126.115.201:4747/video" # Change this to macth phone
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # To keep track of frames per second using a high-performance counter
    old_time = time.perf_counter()
    fps = 0
    stop = False
    process_rgb = True
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Change from OpenCV BGR to scikit image RGB
        new_image = new_frame[:, :, ::-1]
        
        new_image_gray = color.rgb2gray(new_image)
        if process_rgb:
            proc_img = detect_plants(new_image)
            # convert back to OpenCV BGR to show it
            proc_img = proc_img[:, :, ::-1]
        else:
            proc_img = process_gray_image(new_image_gray)

        # update FPS - but do it slowly to avoid fast changing number
        new_time = time.perf_counter()
        time_dif = new_time - old_time
        old_time = new_time
        fps = fps * 0.95 + 0.05 * 1 / time_dif

        # Put the FPS on the new_frame
        str_out = f"fps: {int(fps)}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10)
        #show_in_moved_window('Input gray', new_image_gray, 600, 10)
        show_in_moved_window('Processed image', proc_img, 600, 10)

        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()