#%% Imports
from skimage import io, color
from skimage.morphology import binary_closing, binary_opening
from skimage.morphology import disk
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.color import label2rgb
import pydicom as dicom
from scipy.stats import norm
from scipy.spatial import distance

#%% Given functions
def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap="gray", vmin=-200, vmax=500)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()

#%% Load the dicom image
in_dir = "data/"
ct = dicom.read_file(in_dir + 'Training.dcm')
img = ct.pixel_array
print(img.shape)
print(img.dtype)
io.imshow(img,vmin=np.min(img),vmax=np.max(img))
# %% Ex 1 try to make good visualisation of the spleen
#HU typically ranges from 0 to 150.
io.imshow(img, vmin=-50, vmax=150, cmap='gray')
io.show()
# %% Inspecting the spleen
spleen_roi = io.imread(in_dir + 'SpleenROI.png')
# convert to boolean image
spleen_mask = spleen_roi > 0
spleen_values = img[spleen_mask]

#Plot histogram
plt.hist(spleen_values, bins=100)
plt.show()
mu_spleen = np.mean(spleen_values)
std_spleen = np.std(spleen_values)

print(f"Average HU: {AvgHU}")
print(f"Standard deviation HU: {sdHU}")

# Mean a bit lower than expected
# Standard deviation is quite high as well
#Histogram looks gaussian

# %% EX 4 Fit gaussian to histogram and plot

n, bins, patches = plt.hist(spleen_values, 60, density=1)
pdf_spleen = norm.pdf(bins, mu_spleen, std_spleen)
plt.plot(bins, pdf_spleen)
plt.xlabel('Hounsfield unit')
plt.ylabel('Frequency')
plt.title('Spleen values in CT scan')
plt.show()


# %% Ex 5: PLot histogram and gaussian for all organs
organs = ['Background','Bone','Liver','Spleen',
          'Trabec','Fat','Kidney']
colors = ['black','red','blue','green','yellow','purple','orange']
plt.figure()
for organ in organs:
    mask = io.imread(in_dir + organ + 'ROI.png') > 0
    values = img[mask]
    mu = np.mean(values)
    std = np.std(values)
    n, bins, patches = plt.hist(values, 60, density=1,
                                 alpha=0.3, color=colors[organs.index(organ)])
    pdf = norm.pdf(bins, mu, std)
    plt.plot(bins, pdf, color=colors[organs.index(organ)], label=organ)

plt.legend()
plt.xlim(-200, 1000) # Exclude background pixels
plt.show()

#Bone does not look gaussian, the rest do
#Bone does not look difficult to separate from the others

# %% Ex 6 Define classes 
# Define classes. Kidney, spleen and liver are combined du to overlap
classes = ['Background','Bone','Trabec','Fat',
          'SoftTissue']

#%% Minimum distance classification =================================================

#Compute class ranges
class_means = []
soft_vals = np.zeros(3)
for organ in organs:
    mask = io.imread(in_dir + organ + 'ROI.png') > 0
    values = img[mask]
    mu = np.mean(values)
    std = np.std(values)
    i = 0
    if organ == 'Kidney' or organ == 'Spleen' or organ == 'Liver':
        soft_vals.append(values,)
        i += 1 
        if i == 3:
            mu = np.mean(soft_vals)
            class_means.append(mu)
    else:
        class_means.append(mu)

#%% Compute class ranges by shifting all cells and dividing by two

t_background = -200
t_soft = 

# %%
