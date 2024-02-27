#%% Import packages
from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom

# %% Reading image
# Directory containing data and images
in_dir = "data/"

# X-ray image
im_name = "metacarpals.png"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)
#%% Check the image dimensions
print(im_org.shape)
# %% Check the pixel type
print(im_org.dtype)

#%% Display the image ===================================
#Display the image and try to use the simple viewer 
#tools like the **zoom** tool to inspect the finger 
# bones. You can see the pixel values at a given pixel position
# (in x, y coordinates) in the upper right corner.
# Where do you see the highest and lowest pixel values?
io.imshow(im_org)
plt.title('Metacarpal image')
io.show()
#Maximum pixel value
idMax = np.where(im_org==np.max(im_org))
print(idMax)
print(np.max(im_org))
#Minimum pixel value
idMin = print(np.where(im_org==np.min(im_org)))
print(idMin)
print(np.min(im_org))

# %% Display an image using colormap:
io.imshow(im_org, cmap="jet")
plt.title('Metacarpal image (with colormap)')
io.show()

# %% Experiment with different colormaps 
#For example cool, hot, pink, copper, coolwarm, cubehelix, and terrain.
print('cool')
io.imshow(im_org, cmap="cool", label='cool')
print('hot')
io.imshow(im_org, cmap="hot", label='hot')
print('pink')
io.imshow(im_org, cmap="pink", label='pink')
print('cubehelix')
io.imshow(im_org, cmap="cubehelix", label='cubehelix')

# %% Grey scale scaling ======================

# Sometimes, there is a lack of contrast in an image or the brightness
# levels are not optimals. It possible to scale the way the image is 
# visualized, by forcing a pixel value range to use the full gray scale
# range (from white to black)
io.imshow(im_org, vmin=20, vmax=170) # 20 wil be displayed as black and 170 as white 
plt.title('Metacarpal image (with gray level scaling)')
io.show()

# %% Automatic scaling of the visualization darkest is black, brightest is white
img = im_org
pxrange = [np.min(img),np.max(img)]
print("Range of pixel values: ", pxrange)
io.imshow(img, vmin=pxrange[0] , vmax=pxrange[1])

# %% Histogram functions ================================================
# Computing and visualizing the image histogram is a very important tool
# to get an idea of the quality of an image.
plt.hist(im_org.ravel(), bins=256)
plt.title('Image histogram')
io.show()

#%%
# Since the histogram functions takes 1D arrays as input,
# the function `ravel` is called to convert the image into a 1D array.
# The bin values of the histogram can also be stored by writing:
h = plt.hist(im_org.ravel(), bins=256)

#%%
# The value of a given bin can be found by:
bin_no = 100
count = h[0][bin_no]
print(f"There are {count} pixel values in bin {bin_no}")

#%%
#Here `h` is a list of tuples, where in each tuple the first 
#element is the bin count and the second is the bin edge. 
#So the bin edges can for example be found by:
bin_left = h[1][bin_no]
bin_right = h[1][bin_no + 1]
print(f"Bin edges: {bin_left} to {bin_right}")

# %%
#Here is an alternative way of calling the histogram function:
y, x, _ = plt.hist(im_org.ravel(), bins=256)

# %%
commonIntensityID = np.argmax(h[0])
bin_left = h[1][commonIntensityID]
bin_right = h[1][commonIntensityID + 1]
print(f"The most common pixel intensity is from: {bin_left} to {bin_right}")

# %% Pixel values and image coordinate systems ==========================
# We are using **scikit-image** and the image is represented using
# a **NumPy** array. Therefore, a two-dimensional image is indexed 
# by rows and columns (abbreviated to **(row, col)** or **(r, c)**) 
# with **(0, 0)** at the top-left corner.

r = 100
c = 50
im_val = im_org[r, c]
print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

# %% Pixel value at (110,90)
im_val = im_org[110,90]
print(f"The pixel value at (110,90) is {im_val}")
# %% Slicing exmaple
im_org[:30] = 0 #Sets the pixel values of the first 30 rows to 0
io.imshow(im_org)
io.show()

# %% Mask
# A **mask** is a binary image of the same size as 
# the original image, where the values are either 0 or 1¨
# (or True/False). Here a mask is created from the original image. 

#At first we reload the original image and then create a mask
im_org = io.imread(in_dir + im_name)
mask = im_org > 150
io.imshow(mask)
io.show()

#The mask creates a binary image where the pixel values are 1 if they
#were originally above 150 and 0 otherwise.
# %% What does this do?
im_org[mask] = 255
io.imshow(im_org)
io.show()
# All pixels above 150 are set to 255
# %% Color images ===============================================

# In a color image, each pixel is defined using three values:
# R (red), G (green), and B(blue).
# An example image **ardeche.jpg** is provided.
img = io.imread(in_dir + "ardeche.jpg")

#%% Check the image dimensions
print(img.shape)

# %% Check the pixel type
print(img.dtype)

# %% Display the image
io.imshow(img)
io.show()

# %%What are the (R, G, B) pixel values at (r, c) = (110, 90)?
print(img[110,90])

# %% We can change the pixel values at a given position
r = 110 
c = 90
img[r, c] = [255, 0, 0]
io.imshow(img)
io.show()

# %% We can also change the pixel values in a region
img[:300] = [0,255,0]
io.imshow(img)
io.show()

# %% Working with out own image! ===================================
im_org = io.imread(in_dir + "Leaves.jpg")
print(im_org.shape)
print(im_org.dtype)
io.imshow(im_org)
io.show()   
# %% Scaling the image
image_rescaled = rescale(im_org, 0.25, anti_aliasing=True,
                         channel_axis=2)

#Here we selected a scale factor of 0.25.
# We also specify, that we have more than one channel (since it is RGB)
# and that the channels are kept in the third dimension of the NumPy
# array. The rescale function has this side effect, that it changes the
# type of the pixel values. 
# %% Inspect pixels
print(image_rescaled.shape)
print(image_rescaled.dtype)
range = [np.min(image_rescaled),np.max(image_rescaled)]
print("Range of pixel values: ", range)
io.imshow(image_rescaled)

# %%
# The function `rescale` scales the height and the width of the image
# with the same factor. The `resize` functions can scale the height
#  and width of the image with different scales. For example:

image_resized = resize(im_org, (im_org.shape[0] // 4,
                       im_org.shape[1] // 6),
                       anti_aliasing=True)

io.imshow(image_resized)
# %% Automatic scale image to have 400 columns
print(im_org.shape)
scaleFactor =  400 / im_org.shape[1]
image_w400 = resize( im_org, (im_org.shape[0] * scaleFactor,
                       im_org.shape[1] * scaleFactor),
                       anti_aliasing=True)
print(image_w400.shape)
io.imshow(image_w400)


# %%
# To be able to work with the image,
#  it can be transformed into a
# gray-level image:
im_gray = color.rgb2gray(im_org)
im_byte = img_as_ubyte(im_gray)

# We are forcing the pixel type back into **unsigned bytes** 
# using the `img_as_ubyte` function, since the `rgb2gray` functions 
# returns the pixel values as floating point numbers.
# %% Compute and show histogram of your own image
plt.figure()
plt.hist(image_w400[:,:,0].ravel(),bins=256,color="red",alpha=0.3)
plt.hist(image_w400[:,:,1].ravel(),bins=256,color='Green',alpha=0.3)
plt.hist(image_w400[:,:,2].ravel(),bins=256,color='Blue',alpha=0.3)
plt.show()
# %% Color channels ===============================================
