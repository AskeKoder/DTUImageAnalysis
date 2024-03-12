#%% Import packages
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
import os


# %% Explorative analysis================================================================
# Load the image
os.chdir(r"C:\Users\askeb\OneDrive - Danmarks Tekniske Universitet\DTU\9. Semester\Image analysis\Scripts og data\DTUImageAnalysis\exercises\ex3-PixelwiseOperations")
img_org = io.imread('data/vertebra.png')
# Display the image

#Histogram
plt.figure()
plt.hist(img_org.ravel(), bins=256)
#Looks difficult to separate bones from background
# %% Ex 2 comput minium and maximum pixel values
min_val = np.min(img_org)
max_val = np.max(img_org)
print(f"Min value: {min_val}, Max value: {max_val}")
#The full grey-scale space is not used

# %%
from skimage.util import img_as_float
from skimage.util import img_as_ubyte

# *Use `img_as_float` to compute a new float version of your 
# input image. Compute the minimum and maximum values of this 
# float image. Can you verify that the float image is equal to 
# the original image, where each pixel value is divided by 255?*

img_float = img_as_float(img_org)
min_val_float = np.min(img_float)
max_val_float = np.max(img_float)

print(f"Min value: {min_val_float}, Max value: {max_val_float}")
np.sum(img_float - img_org/255)
# They're not exactly equal, but the difference is very small and 
#can be due to rounding of the numbers
# %%
#As stated above, an (unsigned) 
# float image can have pixel values in [0, 1]. 
# When using the Python skimage function `img_as_ubyte` 
# on an (unsigned) float image, it will multiply all values 
# with 255 before converting into a byte. Remember that all 
# decimal number will be converted into integers by this, and 
# some information might be lost.

# *Use `img_as_ubyte` on the float image
# you computed in the previous exercise. 
# Compute the Compute the minimum and maximum 
# values of this image. Are they as expected?*

img_ubyte = img_as_ubyte(img_float)
min_val_ubyte = np.min(img_ubyte)
max_val_ubyte = np.max(img_ubyte)

print(f"Min value: {min_val_ubyte}, Max value: {max_val_ubyte}")
#It is the same in this case, but might not be the case for all pixels
# %% Automatic function for stecthing of histogram
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

# %% Testing the function
fig, axs=plt.subplots(2,2)
io.imshow(img_org, ax=axs[0,0])
axs[0,0].set_title('Original image')
axs[1,0].hist(img_org.ravel(), bins=256)
axs[1,0].set_xlim([0,255])
io.imshow(histogram_stretch(img_org), ax=axs[0,1])
axs[0,1].set_title('Stretched image')
axs[1,1].hist(histogram_stretch(img_org).ravel(), bins=256)
axs[1,1].set_xlim([0,255])
plt.show()

# %% Nonlinear pixel value mapping
def gamma_map(img,gamma):
    img_float = img_as_float(img)
    img_float_gamma = img_float**gamma
    img_ubyte_gamma = img_as_ubyte(img_float_gamma)
    return img_ubyte_gamma

# %% Testing the function
gamma = 2
fig, axs=plt.subplots(1,2)
io.imshow(img_org, ax=axs[0])
axs[0].set_title('Original image')
io.imshow(gamma_map(img_org,gamma), ax=axs[1])
axs[1].set_title(f'Gamma = {gamma}')
plt.show()

#Dark pixels are darkened more than light pixels

# %% Thresholding function
def threshold_image(img,thres):
     """
    Apply a threshold in an image and return the resulting image
    :param img_in: Input image in
    :param thres: The treshold value in the range [0, 255]
    :return: Resulting image (unsigned byte) where background is 0 and foreground is 255
    """
     img = img_as_ubyte(img)
     return img_as_ubyte((img>thres) ) 

# %% Testing
thres = 195
fig, axs=plt.subplots(1,2)
io.imshow(img_org, ax=axs[0])
axs[0].set_title('Original image')
io.imshow(threshold_image(img_org,thres), ax=axs[1])
axs[1].set_title(f'Threshold = {thres}')
plt.show()


# %% Automatic thresholds using Otsu's metho
from skimage.filters import threshold_otsu

thres_otsu = threshold_otsu(img_org)

# Compare with the manual threshold
fig, axs=plt.subplots(1,2)
io.imshow(img_as_ubyte(img_org>thres_otsu), ax=axs[0])
axs[0].set_title(f'Otsu threshold = {thres_otsu}')
io.imshow(threshold_image(img_org,thres), ax=axs[1])
axs[1].set_title(f'Threshold = {thres}')
plt.show()

#It depends on what you desire
plt.hist(img_org.ravel(), bins=256)
plt.axvline(thres, color='r')
plt.axvline(thres_otsu, color='g')
plt.show()

# %% Try to separate the background
img = io.imread('data/dark_background.png')
# convert to greyscale
img_gray = color.rgb2gray(img)
img_gray = img_as_ubyte(img_gray)
fig, axs=plt.subplots(2,1)
io.imshow(img_gray,ax=axs[0])
axs[1].hist(img_gray.ravel(),bins=256)
plt.show()

#Apply threshold
thres = 5
img_thresh = threshold_image(img_gray,thres)
io.imshow(img_thresh)

# %% Segmentation on color images
img_org = io.imread('data/DTUSigns2.jpg')
io.imshow(img_org)

#At first we want to separate to blue signe from the background
def detect_dtu_signs(img_org):
    #Separate the color channels
    r_comp = img_org[:, :, 0]
    g_comp = img_org[:, :, 1]
    b_comp = img_org[:, :, 2]
    #Create a mask for the blue sign
    segm_blue = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & (b_comp > 180) & (b_comp < 200)
    return segm_blue

# %% Extend the function so it can also detect red signs

#At first we must find out how the red sign looks to the computer
from skimage.measure import profile_line
start = (1300,2315)
end = (1700, 2315)
pred = profile_line(img_org[:,:,0], start, end)
pgreen = profile_line(img_org[:,:,1], start, end)
pblue = profile_line(img_org[:,:,2], start, end)
plt.plot(pred, color='r')
plt.plot(pgreen, color='g')
plt.plot(pblue, color='b')
plt.yticks(np.array([0, 50, 100, 150,160,170,180,190, 200, 255]))
plt.grid()
#%%
def detect_dtu_signs(img_org):
    #Separate the color channels
    r_comp = img_org[:, :, 0]
    g_comp = img_org[:, :, 1]
    b_comp = img_org[:, :, 2]
    #Create a mask for the blue sign
    segm_blue = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & (b_comp > 180) & (b_comp < 200)
        #Create a mask for the red sign
    segm_red = (r_comp > 160) & (r_comp < 175) & (g_comp > 50) & (g_comp < 70) & (b_comp > 50) & (b_comp < 70)
    return segm_blue, segm_red

# %% Testing the function
segm_blue, segm_red = detect_dtu_signs(img_org)
fig, axs=plt.subplots(1,2)
axs[0].imshow(segm_blue)
axs[1].imshow(segm_red)


# %%Sometimes it gives better segmentation results when 
# the tresholding is done in HSI (also known as HSV - hue, 
# saturation, value) space. Start by reading the  
# **DTUSigns2.jpg** image, convert it to HSV and show the 
# hue and value

#We already have the image
io.imshow(img_org)

#Convert to HSV
hsv_img = color.rgb2hsv(img_org)
#Show the hue and value
hue_img = hsv_img[:, :, 0]
value_img = hsv_img[:, :, 2]
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 2))
ax0.imshow(img_org)
ax0.set_title("RGB image")
ax0.axis('off')
ax1.imshow(hue_img, cmap='hsv')
ax1.set_title("Hue channel")
ax1.axis('off')
ax2.imshow(value_img)
ax2.set_title("Value channel")
ax2.axis('off')

fig.tight_layout()
io.show()

# %%**Exercise 15:** *Now make a sign segmentation 
# function using tresholding in HSV space and locate both
# the blue and the red sign.*

#Explorative analysis
#%% Red sign
start_red = (1300,2315)
end_red = (1700, 2315)
ph = profile_line(hsv_img[:,:,0], start_red, end_red)
ps = profile_line(hsv_img[:,:,1], start_red, end_red)
pv = profile_line(hsv_img[:,:,2], start_red, end_red)
plt.plot(ph, 'b')
plt.plot(ps, color='r')
plt.plot(pv, color='g')
plt.show()

#%%Blue sign
start_blue = (1500,800)
end_blue = (2400, 800)
ph = profile_line(hsv_img[:,:,0], start_blue, end_blue)
ps = profile_line(hsv_img[:,:,1], start_blue, end_blue)
pv = profile_line(hsv_img[:,:,2], start_blue, end_blue)
plt.plot(ph, 'b')
plt.plot(ps, color='r')
plt.plot(pv, color='g')
plt.show()


#%%Function
def HSV_detect_dtu_signs(img_org):
    #Convert to HSV
    hsv_img = color.rgb2hsv(img_org)
    #Separate the color channels
    h_comp = hsv_img[:, :, 0]
    s_comp = hsv_img[:, :, 1]
    v_comp = hsv_img[:, :, 2]
    #Create a mask for the blue sign
    segm_blue = (h_comp > 0.55) & (h_comp < 0.65) & (v_comp > 0.7) & (v_comp < 0.8)
    #Create a mask for the red sign
    segm_red = (h_comp > 0.95)
    return segm_blue, segm_red

# %% Test of function
segm_blue, segm_red = HSV_detect_dtu_signs(img_org)
fig, axs=plt.subplots(1,2)
axs[0].imshow(segm_blue)
axs[1].imshow(segm_red)
#We also get the red signs in the background!


# %% Function that connects to a camera


# %% Explorative analysis to identify plants in an image
img_org = io.imread('Plants.jpg')
io.imshow(img_org)

io.imshow(img_org[700:1200,1400:1600,2])
img_hsv = color.rgb2hsv(img_org)
#%%
fig, axs=plt.subplots(1,3)
io.imshow(img_hsv[700:1200,1400:1600,0],ax=axs[0])
io.imshow(img_hsv[700:1200,1400:1600,1],ax=axs[1])
io.imshow(img_hsv[700:1200,1400:1600,2],ax=axs[2])
plt.show()

ph = profile_line(img_hsv[:,:,0], (1000,1400), (1000, 1600))
ps = profile_line(img_hsv[:,:,1], (1000,1400), (1000, 1600))
pv = profile_line(img_hsv[:,:,2], (1000,1400), (1000, 1600))
plt.plot(ph, 'b')
plt.plot(ps, color='r')
plt.plot(pv, color='g')

