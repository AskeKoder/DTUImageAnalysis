#%%
from scipy.ndimage import correlate
from skimage import color
import numpy as np

#create small image 
input_img = np.arange(25).reshape(5, 5)
print(input_img)

#Add simple filter 
weights = [[0, 1, 0],
		   [1, 2, 1],
		   [0, 1, 0]]

#Correlate image with the weights
res_img = correlate(input_img, weights)

#%% Ex1: Print the value in position (3, 3) in 
#`res_img`. Explain the value?
print(res_img[2,2])

#The weights sacle the values in the image.
#2*12 + 7 + 13 + 17 + 11 = 72
# %%Exercise 2 #How the borders area handled

# Compare the output images when using `reflect` and 
# `constant` for the border. Where and why do you see
# the differences.

res_img_constant = correlate(input_img, weights, mode="constant", cval=10)

res_img_reflect = correlate(input_img, weights, mode="reflect")

print(input_img)
print(res_img_constant) #a bit darker around the edges
print(res_img_reflect)
# %% Mean filtering =================================
#Read and show img
from skimage import io
import os
os.chdir(r"C:\Users\askeb\OneDrive - Danmarks Tekniske Universitet\DTU\9. Semester\Image analysis\Scripts og data\DTUImageAnalysis\exercises\ex4-ImageFiltering")
im_org = io.imread('data/Gaussian.png')
im_org = color.rgb2gray(im_org)
io.imshow(im_org)

#%%
#Create weight matrix for mean filtering
size = 10
# Two dimensional filter filled with 1
weights = np.ones([size, size])
# Normalize weights
weights = weights / np.sum(weights)
print(weights)
#%%
#Apply filter
mean_im = correlate(im_org, weights)
io.imshow(mean_im)
# Image becomes blurry and more so when filter size is increased

# %% Median filtering =================================
from skimage.filters import median

size = 10
footprint = np.ones([size, size])
med_im = median(im_org, footprint)
io.imshow(med_im)

#Noise is removed and edges are preserved 
#better than with mean filtering
# %% Test on salt and pepper img
from skimage import img_as_ubyte
im_org = io.imread('data/SaltPepper.png')
print(im_org.shape)
im_gray = img_as_ubyte(color.rgb2gray(im_org))
io.imshow(im_gray)


import matplotlib.pyplot as plt
fig, axs=plt.subplots(1,2)
size = 5
#Apply mean filter
weights = np.ones([size, size])
weights = weights / np.sum(weights)
mean_im = correlate(im_gray, weights)
io.imshow(mean_im, ax=axs[0])

#Apply median filter
footprint = np.ones([size, size])
med_im = median(im_gray, footprint)
io.imshow(med_im)
med_im = median(im_gray, footprint)
io.imshow(med_im, ax=axs[1])


# %% Gaussian filter =================================
im = io.imread('data/Gaussian.png')
print(im.shape)
im_gray = color.rgb2gray(im)
io.imshow(im_gray)
# %% Apply gaussian filter
from skimage.filters import gaussian
sigma = 1
gauss_img = gaussian(im_org, sigma)
io.imshow(gauss_img)

# %% Ex 7 Use the filters on a real image
im_org = io.imread('data/car.png')
print(im_org.shape)
im_gray = color.rgb2gray(im_org)
io.imshow(im_gray)

#Apply mean filter
size = 10
weights = np.ones([size, size])
weights = weights / np.sum(weights)
mean_im = correlate(im_gray, weights)
io.imshow(mean_im)
# %% Median filter
size =20
footprint = np.ones([size, size])
med_im = median(im_gray, footprint)
io.imshow(med_im)
# %% Gaussian filter
sigma = 20
gauss_img = gaussian(im_gray, sigma)
io.imshow(gauss_img)
#The differences are more distinct when using the median filter


#%% Edge filtering
from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
from skimage.filters import prewitt

img_org = io.imread('data/donald_1.png')
io.imshow(img_org)
#RBG image, we ee to convert
img_gray = color.rgb2gray(img_org)
io.imshow(img_gray)
#%%Horizontal edge filter
#[-1,0,1
# -1,0,1
# -1,0,1]
im_h = prewitt_h(img_gray)
io.imshow(im_h)

#Output image is has values from -1 to 1
#The filter returns positive values for dark to light
#transitions from left to right and vice versa



#%%Vertical edge filter
#[-1,-1,-1
# 0, 0, 0
# 1, 1, 1]
im_v = prewitt_v(img_gray)
io.imshow(im_v)
#%%Combining the two filters
im = prewitt(img_gray)
io.imshow(im)

#All edges appear

#%% Doing the same manually
im = np.sqrt(im_h**2 + im_v**2)
io.imshow(im)

# %% Edge detection on medical image

im_org = io.imread('data/ElbowCTSlice.png')
io.imshow(im_org)
im_org.shape
im_gray = color.rgb2gray(im_org)


im = im_gray.copy()

#Filter image with median filter to get clean edges ( set 1 for no filter)
size = 15
footprint=(np.ones([size, size]))
im = median(im, footprint)

#Add gaussian filter to smooth the image (set 0 for no filter)
sigma = 1
im = gaussian(im, sigma)

#Identify edges using a prwitt filter
im_edge = prewitt(im)

#Otsus method
im_edge = img_as_ubyte(im_edge)
from skimage.filters import threshold_otsu
T = threshold_otsu(im_edge)
im_edge = im_edge > T
io.imshow(im_edge)

#%% For understanding
min_val = im_edge.min()
max_val = im_edge.max()
io.imshow(im_edge, vmin=min_val, vmax=max_val, cmap="terrain")


# %%
