#%% 
import numpy as np
from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import math


#%% Computing camera parameters ===============================================
in_dir = "figures/"
im = io.imread(in_dir + "ArcTangens.png")
io.imshow(im)

#%% Calculate the angle
#b = tan(theta)*a =>b/a = tan(theta) => theta = atan(b/a)
a=10
b=3

#Different ways to calculate the angle
theta_1 = math.atan(b/a)
theta_2 = math.atan2(b,a)
print([theta_1,theta_2])
#Same result
# %%Ex2 Create a Python function called `camera_b_distance`.
# The function should accept two arguments, a focal length f and an
# object distance g. It should return the distance from the lens to
# where the rays are focused (b) (where the CCD should be placed)
# The function should start like this:
def camera_b_distance(f, g):
    """
    camera_b_distance returns the distance (b) where the CCD should be placed
    when the object distance (g) and the focal length (f) are given
    :param f: Focal length [mm]
    :param g: Object distance [m]
    :return: b, the distance where the CCD should be placed [mm]
    """
    b = 1/(1/f-1/(g*1000))
    return b

#Testing
print(camera_b_distance(15,np.array([0.1,1,5,15])))

#Plotting
f = 15
gseq = np.linspace(0.1, 100, 100)
plt.plot(gseq, camera_b_distance(f, gseq))
#The further the object is away from the lens, the parallel the rays are
#and the closer the CCD should be placed to the focal point

# %% Ex 3

# In the following exercise, you should remember to explain when
# something is in mm and when it is in meters. To convert between
# radians and degrees you can use:

def angle_degrees (angle_radians):
    return 180.0 / np.pi * angle_radians

# Thomas is 1.8 meters tall and standing 5 meters from a camera. The
# cameras focal length is 5 mm. The CCD in the camera can be seen in
# the figure below. It is a 1/2" (inches) CCD chip and the
# image formed by the CCD is 640x480 pixels in a (x,y) coordinate system.

im = io.imread(in_dir + "CCDchip.png")
io.imshow(im)

g = 5 #m
f = 5 #mm
# Where should the CCD be placed to capture Thomas sharply?
# Use the function we just created
b = camera_b_distance(f, g)
print(f"The CCD should be placed {b} mm from the lens")

# How tall is Thomas on the CCD?
G = 1.8/2 #m
#b/B = g/G => B = b*G/g #Units cancel out
B = b*G/g
h_ccd = 2*B
print(f"Thomas is {height_ccd} mm tall on the CCD")

#What is the size of a single pixel on the CCD chip? (in mm)?
#The CCD is 640x480 pixels
#The CCD chip is 6.4mm x 4.8mm
A_CCD = 6.4*4.8 #mm^2
A_pixel = A_CCD/(640*480)
print(f"The size of a single pixel on the CCD chip is {A_pixel} mm^2")

#How tall (in pixels) will Thomas be on the CCD-chip?
#Pixels are square, thereforfore the height of pixel is
h_pixel = np.sqrt(A_pixel)

#The height of thomas in pixels is
h_ccd_pixels = h_ccd/h_pixel
print(f"Thomas is {h_ccd_pixels} pixels tall on the CCD")

#What is the horizontal field-of-view (in degrees)?
#The largest horizontal B
Bmax_horizontal = 640/2*h_pixel
#konwing b, we can calculate the angle
theta_horizontal = 2*np.arctan(Bmax_horizontal/b)

#What is the vertical field-of-view (in degrees)?
Bmax_vertical = 480/2*h_pixel
theta_vertical = 2*np.arctan(Bmax_vertical/b)

print(f"The horizontal field-of-view is {angle_degrees(theta_horizontal)} degrees")
print(f"The vertical field-of-view is {angle_degrees(theta_vertical)} degrees")

# %%
