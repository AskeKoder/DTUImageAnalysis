#%% Import packages
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk 
from skimage import io
import matplotlib.pyplot as plt
import os
import numpy as np
os.chdir(r'C:\Users\askeb\OneDrive - Danmarks Tekniske Universitet\DTU\9. Semester\Image analysis\Scripts og data\DTUImageAnalysis\exercises\ex4b-ImageMorphology')

def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    io.show()

#%% Load image
in_dir = 'data/'
filename = 'lego_5.png'
im_org = io.imread(in_dir + filename)

# Convert to greyscale
from skimage.color import rgb2gray
im_gray = rgb2gray(im_org)

#Apply otsu thresholding
from skimage.filters import threshold_otsu
thresh = threshold_otsu(im_gray)
bin_img = im_gray < thresh

plot_comparison(im_gray, bin_img, 'Binary image') #Not perfect, som holes in the brick

# %% we will try to improve 
footprint = disk(3)
# Check the size and shape of the structuring element
print(footprint)

# Apply erosion
eroded = erosion(bin_img, footprint)
plot_comparison(bin_img, eroded, 'erosion')
#Holes get bigger
# %% Apply dilation
footprint = disk(3)
dilated = dilation(bin_img, footprint)
plot_comparison(bin_img, dilated, 'dilation') #At footprint = 7, the holes are filled
# there are also errors outside th brick
# %%
footprint = disk(2)
opened = opening(bin_img, footprint)
plot_comparison(bin_img, opened, 'opening')

# %%
footprint = disk(15)
closed = closing(bin_img, footprint)
plot_comparison(bin_img, closed, 'closing')
# %% Object Outline
def compute_outline(bin_img):
    """
    Computes the outline of a binary image
    """
    footprint = disk(1)
    dilated = dilation(bin_img, footprint)
    outline = np.logical_xor(dilated, bin_img)
    return outline
# %% EX 6
outline_img = compute_outline(bin_img)
plot_comparison(bin_img, outline_img, 'Outline')

# %% Ex 7 
footprint = disk(1)
opened = opening(bin_img, footprint)
plot_comparison(bin_img, compute_outline(opened), 'Outline - opening (1)')

footprint = disk(15)
closed = closing(bin_img, footprint)
plot_comparison(bin_img, compute_outline(closed), 'Outline - closing(15)')

#Closing captures the outline very well
# %% Let's try on multiple legos
filename = 'lego_7.png'
im_org = io.imread(in_dir + filename)

# Convert to greyscale
from skimage.color import rgb2gray
im_gray = rgb2gray(im_org)

#Apply otsu thresholding
from skimage.filters import threshold_otsu
thresh = threshold_otsu(im_gray)
bin_img = im_gray < thresh

plot_comparison(im_gray, bin_img, 'Binary image') #Not perfect, som holes in the brick
outline_img = compute_outline(bin_img)
plot_comparison(im_gray, outline_img, 'Outline image')
#different "levels" of outlines are identified

#%%Ex 8 Find the outlines using closing
footprint = disk(8)
closed = closing(bin_img, footprint)
plot_comparison(bin_img, compute_outline(closed), 'Outline - closing(15)')
 # disk(8) is good


# %% Try the same on another lego image
filename = 'lego_3.png'
im_org = io.imread(in_dir + filename)
# Convert to greyscale
from skimage.color import rgb2gray
im_gray = rgb2gray(im_org)
#Apply otsu thresholding
from skimage.filters import threshold_otsu
thresh = threshold_otsu(im_gray)
bin_img = im_gray < thresh
#Closing and find outline
footprint = disk(15)
closed = closing(bin_img, footprint)
outline_img = compute_outline(closed)
plot_comparison(im_gray, outline_img, 'Outline image')
#Not a very good result we have to increase the footprint
# we cannot use closing to find  perfect outlines in this case

# %% Ex 11
filename = 'lego_9.png'
im_org = io.imread(in_dir + filename)
# Convert to greyscale
from skimage.color import rgb2gray
im_gray = rgb2gray(im_org)
#Apply otsu thresholding
from skimage.filters import threshold_otsu
thresh = threshold_otsu(im_gray)
bin_img = im_gray < thresh
plot_comparison(im_gray, bin_img, 'Binary image') 
# bricks are touching ( not a good sign for us :( )

# %% Separate the objects using erosion
footprint = disk(27)
eroded = erosion(bin_img, footprint)
plot_comparison(bin_img, eroded, 'erosion')

# %%
footprint = disk(10)
dilated = dilation(eroded, footprint)
plot_comparison(eroded, dilated, 'erosion+dilation')
# Around 10 it gets problematic

# %%
filename = 'puzzle_pieces.png'
im_org = io.imread(in_dir + filename)
# Convert to greyscale
from skimage.color import rgb2gray
im_gray = rgb2gray(im_org)
#Apply otsu thresholding
from skimage.filters import threshold_otsu
thresh = threshold_otsu(im_gray)
bin_img = im_gray < thresh
plot_comparison(im_gray, bin_img, 'Binary image')
# Since they have different intensities, otsus
#method has a hard time identifying all the objects

# %% Ex 16 Opening to clean up the image
footprint = disk(3)
opened = opening(bin_img, footprint)
outline_img = compute_outline(opened)
io.imshow(outline_img)
plot_comparison(im_gray, outline_img, 'Outline image')

# no good outlines - bad image for morphology

# %%
