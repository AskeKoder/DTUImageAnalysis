import time
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.util import img_as_ubyte

#Add alarm function if something moves


def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name,img)

# Acquire a new image and convert to grescale and the floating point
def capture_from_camera_and_show_images():
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = True
    if use_droid_cam:
        url = "http://192.168.1.11:4747/video"
    cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # Get first image
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        exit()

    # Transform image to gray scale and then to float, so we can do some processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = img_as_float(frame_gray)

    # To keep track of frames per second
    start_time = time.time()
    n_frames = 0
    stop = False
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Transform image to gray scale and then to float, so we can do some processing
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_frame_gray = img_as_float(new_frame_gray)

        # Compute difference image
        dif_img = np.abs(new_frame_gray - frame_gray)

        #Create a binary image by applying thresholding
        T = 0.2 # our boundary 
        binary_img = dif_img > T # True/False array of pixels exceeding the threshold
        F = sum(binary_img.reshape(-1,1).squeeze()) / np.prod(binary_img.shape) # Number of pixels exceeding the threshold
        A =  0.05 # 5% of the pixels change raise an alarm
        if F > A:
            alarm_str = f"Achtung! {F*100:.1f}% of the pixels changed"
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(new_frame, alarm_str, (200, 200), font, 1, 1, 1)
        binary_img = img_as_ubyte(binary_img) # Convert to 8-bit image

        # Keep track of frames-per-second (FPS)
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Put the FPS on the new_frame
        str_out = f"fps: {fps}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10)
        show_in_moved_window('Input gray', new_frame_gray, 600, 10)
        show_in_moved_window('Difference image', dif_img, 1200, 10)
        show_in_moved_window('Binary image', binary_img, 1200, 10)

        # Old frame is updated
        alpha = 0.95
        frame_gray = alpha * frame_gray + (1-alpha)*new_frame_gray


        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()