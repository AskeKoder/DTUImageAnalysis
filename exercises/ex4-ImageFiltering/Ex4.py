#%%
from scipy.ndimage import correlate
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
io.imshow(im_org)

#%%
#Create weight matrix for mean filtering
size = 5
# Two dimensional filter filled with 1
weights = np.ones([size, size])
# Normalize weights
weights = weights / np.sum(weights)
print(weights)
#%%
#Apply filter
im_filter = correlate(im_org, weights)


# %%
