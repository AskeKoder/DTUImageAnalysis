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

#Read image
im_org = io.imread(in_dir + "DTUSign1.jpg")
io.imshow(im_org)
io.show()

#Only show the red channel
r_comp = im_org[:, :, 0]
io.imshow(r_comp)
plt.title('DTU sign image (Red)')
io.show()

# %%Visualize the R, G, and B components individually. 
# Why does the DTU Compute sign look bright on the R channel
# image and dark on the G and B channels?  Why do the walls of
# the building look bright in all channels?
fig, axs= plt.subplots(1,3, figsize=(15,5))
axs[0].imshow(im_org[:, :, 0])
axs[0].set_title('Red channel')
axs[1].imshow(im_org[:, :, 1])
axs[1].set_title('Green channel')
axs[2].imshow(im_org[:, :, 2])
axs[2].set_title('Blue channel')
plt.show()

# %%  Simple Image Manipulations ===================================

# Show the image again and save it to disk as 
# **DTUSign1-marked.jpg** using the `io.imsave` 
# function. Try to save the image using different image
# formats like for example PNG.*

io.imshow(im_org)
im_org[500:1000, 800:1500, :] = 0 #Insert a black rectangle
io.imshow(im_org)
io.imsave(in_dir + "DTUSign1-marked.PNG", im_org)
# %%
# Try to create a blue rectangle
# around the DTU Compute sign and save the resulting image.

#Read original image
im_org = io.imread(in_dir + "DTUSign1.jpg")
io.imshow(im_org)
im_blue = im_org.copy()
im_blue[1500:1800,2300:2800,[0,1]] = 0
io.imshow(im_blue)

# %% Try to automatically create an image based on
#  **metacarpals.png** where the bones are colored blue. 
#read image
im_org = io.imread(in_dir + "metacarpals.png")

#Standardize the image
Y = im_org-np.mean(im_org,axis=1)
Y = Y/np.std(im_org,axis=1)
# Set values below 0 to 0
Y[Y<0.0] = 0
Y = Y/np.max(Y) * 255
Y = Y.astype(np.uint8)
Y = color.gray2rgb(Y)
Y[:,:,[0,1]] = 0
io.imshow(Y, vmin=np.min(Y), vmax=np.max(Y))

# %% ## Advanced Image Visualisation=================================

# Before implementing a fancy image analysis algorithm, it is very important
#  to get an intuitive understanding of how the image *looks as seen from the 
# computer*. The next set of tools can help to gain a better understanding.

# In this example, we will work with an x-ray image of the human hand. Bones 
# are hollow and we want to understand how a hollow structure looks on an image. 

# Start by reading the image **metarcarpals.png**. To investigate the 
# properties of the hollow bone, a grey-level profile can be sampled across 
# the bone. The tool `profile_line` can be used to sample a profile across the 
# bone:

p = profile_line(im_org, (342, 77), (320, 160))
plt.plot(p)
plt.ylabel('Intensity')
plt.xlabel('Distance along line')
plt.show()


# %%
# An image can also be viewed as a landscape, 
# where the height is equal to the grey level
in_dir = "data/"
im_name = "road.png"
im_org = io.imread(in_dir + im_name)
im_gray = color.rgb2gray(im_org)
ll = 200
im_crop = im_gray[40:40 + ll, 150:150 + ll]
xx, yy = np.mgrid[0:im_crop.shape[0], 0:im_crop.shape[1]]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xx, yy, im_crop, rstride=1, cstride=1, cmap=plt.cm.jet,
                       linewidth=0)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# %% ## DICOM images ===============================================

# Typical images from the hospital are stored in the DICOM format. 
#An example image from a computed tomography examination of abdominal area 
#is used in the following.

# Start by examining the header information using:
in_dir = "data/"
im_name = "1-442.dcm"
ds = dicom.dcmread(in_dir + im_name)
print(ds)

# %% access image
im = ds.pixel_array
im.shape
im.dtype
io.imshow(im, vmin=-1000, vmax=1000, cmap='gray')
io.show()

# %%
