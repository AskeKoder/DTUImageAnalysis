#%% Packages
import numpy as np
from skimage import io, color, morphology
from skimage.util import img_as_float, img_as_ubyte
import math
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb
import os
import matplotlib.pyplot as plt
os.chdir(r'C:\Users\askeb\OneDrive - Danmarks Tekniske Universitet\DTU\9. Semester\Image analysis\Scripts og data\DTUImageAnalysis\exercises\ex5-BLOBAnalysis_mangler')

#%% Function to show images side by side 
def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()

#%% Lego classification =======================================================

# Load image
im_org = io.imread('data/lego_4_small.png')

# Convert to grayscale
im_gray = color.rgb2gray(im_org)

#Apply Otsu thresholding
thresh = threshold_otsu(im_gray)
im_thresh = im_gray < thresh #Light background, we want what's darker
show_comparison(im_org, im_thresh, 'Thresholded')

# %% Remove border BLOBs
im_no_border = segmentation.clear_border(im_thresh)
io.imshow(im_no_border)

# %% Clean up th image using morphological operations
from skimage.morphology import disk 
footprint = disk(5)
print(footprint)
im_closed = morphology.closing(im_no_border, footprint)
show_comparison(im_no_border, im_closed, 'Closed')

im_clean = morphology.opening(im_closed, footprint)
show_comparison(im_closed, im_clean, 'Cleaned')

# %% Find labels
label_img = measure.label(im_clean) #Label connected regions
n_labels = label_img.max()
print(f"Number of labels: {n_labels}")

# %% # Display the labels
im_labels = label2rgb(label_img) # Colorize the labels
show_comparison(im_org, im_labels, 'Labels')

# %% Compute blob features 
region_props = measure.regionprops(label_img)
areas = np.array([prop.area for prop in region_props])
plt.hist(areas, bins=50)
plt.show()

# %% Cell counting ============================================================
in_dir = "data/"
img_org = io.imread(in_dir + 'Sample E2 - U2OS DAPI channel.tiff')
# slice to extract smaller image
img_small = img_org[700:1200, 900:1400]
img_gray = img_as_ubyte(img_small) 
io.imshow(img_gray, vmin=0, vmax=80)
plt.title('DAPI Stained U2OS cell nuclei')
io.show()

# %% Inspect histogram
# avoid bin with value 0 due to the very large number 
#of background pixels
plt.hist(img_gray.ravel(), bins=256, range=(1, 100))
io.show()

# %% Select threshold
thresh = threshold_otsu(img_gray)
img_thresh = img_gray > thresh
show_comparison(img_gray, img_thresh, 'Thresholded')


# %%Remove border blobs
img_no_border = segmentation.clear_border(img_thresh)

label_img = measure.label(img_no_border)
image_label_overlay = label2rgb(label_img)
show_comparison(img_gray, image_label_overlay, 'Found BLOBS')

# %% Compute object features
region_props = measure.regionprops(label_img)
areas = np.array([prop.area for prop in region_props])
plt.hist(areas, bins=100)

# %% BLOB classification by area
min_area = 50
max_area = 125

# Create a copy of the label_img
label_img_filter = label_img
for region in region_props:
	# Find the areas that do not fit our criteria
	if region.area > max_area or region.area < min_area:
		# set the pixels in the invalid areas to background
		for cords in region.coords:
			label_img_filter[cords[0], cords[1]] = 0
# Create binary image from the filtered label image
i_area = label_img_filter > 0
show_comparison(img_small, i_area, 'Found nuclei based on area')



# %% Extract all perimeters
perimeters = np.array([prop.perimeter for prop in region_props])

plt.scatter(areas, perimeters)
plt.xlabel('Area')
plt.ylabel('Perimeter')

# %% Copute circularity
circularity = 4 * np.pi * areas / perimeters ** 2
plt.hist(circularity, bins=100)

# %% Use circularity to filter
min_circularity = 0.85
max_circularity = 1.15

#Copy of the label image
label_img_filter = label_img
for region in region_props:
    # Find the areas that do not fit our criteria
    if circularity[region.label - 1] > max_circularity or circularity[region.label - 1] < min_circularity:
        # set the pixels in the invalid areas to background
        for cords in region.coords:
            label_img_filter[cords[0], cords[1]] = 0
# Create binary image from the filtered label image
i_circularity = label_img_filter > 0
show_comparison(img_small, i_circularity, 'Found nuclei based on circularity')

# %% Plot areas versus circularity
plt.scatter(areas, circularity)
plt.xlabel('Area')
plt.ylabel('Circularity')

# %% Filtering based on circulatiry and area
min_area = 50
max_area = 125
min_circularity = 0.85
max_circularity = 1.15

# Create a copy of the label_img
label_img_filter = label_img
for region in region_props:
    # Find the areas that do not fit our criteria
    if region.area > max_area or region.area < min_area or circularity[region.label - 1] > max_circularity or circularity[region.label - 1] < min_circularity:
        # set the pixels in the invalid areas to background
        for cords in region.coords:
            label_img_filter[cords[0], cords[1]] = 0
# Create binary image from the filtered label image
i_area_circularity = img_as_ubyte(label_img_filter > 0)
show_comparison(img_small, i_area_circularity, 'Found nuclei based on area and circularity')

new_labels = measure.label(i_area_circularity)
print(f"Number of cells: {np.max(new_labels)}")

# %% Test method on larger set
in_dir = "data/"
file = "Sample G1 - COS7 cells DAPI channel.tiff"
im_org = io.imread(in_dir + file)
im_org.shape

# slice to extract smaller image and convert to greyscale
im_small = im_org[0:500, 1400:1900]
im_gray = img_as_ubyte(im_small) 
show_comparison(im_org, im_small, 'DAPI Stained U2OS cell nuclei')
io.imshow(im_gray, vmin=0, vmax=80)

#Apply Otsu thresholding
thresh = threshold_otsu(im_gray)
im_thresh = im_gray > thresh
show_comparison(im_gray, im_thresh, 'Thresholded')

#Handle overlaps with opening
footprint = disk(2)
im_opened = morphology.opening(im_thresh, footprint)
show_comparison(im_thresh, im_opened, 'Opened')

#Remove border blobs
im_no_border = segmentation.clear_border(im_thresh)
label_im = measure.label(im_no_border)
image_label_overlay = label2rgb(label_im)
show_comparison(im_opened, image_label_overlay, 'Found BLOBS')

#Compute object features
region_props = measure.regionprops(label_im)
areas = np.array([prop.area for prop in region_props])
perimeters = np.array([prop.perimeter for prop in region_props])
circularity = 4 * np.pi * areas / perimeters ** 2

# Filtering based on circularity and area
min_area = 50
max_area = 125
min_circularity = 0.85
max_circularity = 1.15

#apply filter
print(f"Number of cells before filtering: {np.max(label_im)}")
label_img_filter = label_im
for region in region_props:
    # Find the areas that do not fit our criteria
    if region.area > max_area or region.area < min_area or circularity[region.label - 1] > max_circularity or circularity[region.label - 1] < min_circularity:
        # set the pixels in the invalid areas to background
        for cords in region.coords:
            label_img_filter[cords[0], cords[1]] = 0
# Create binary image from the filtered label image
i_area_circularity = img_as_ubyte(label_img_filter > 0)
show_comparison(im_small, i_area_circularity, 'Found nuclei based on area and circularity')

new_labels = measure.label(i_area_circularity)
print(f"Number of cells: {np.max(new_labels)}")
# %%
