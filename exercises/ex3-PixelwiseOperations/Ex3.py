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

# %%
