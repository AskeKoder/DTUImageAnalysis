#%% Import packages
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

# %% Explorative analysis================================================================
# Load the image
img_org = io.imread('data/vertebra.png')
# Display the image
io.imshow(img_org)

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
# %%
