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
class_stds = []
soft_vals = np.zeros(3)
i=0
for organ in organs:
    mask = io.imread(in_dir + organ + 'ROI.png') > 0
    values = img[mask]
    mu = np.mean(values)
    std = np.std(values)
    if organ == 'Kidney' or organ == 'Spleen' or organ == 'Liver':
        soft_vals = np.concatenate((soft_vals, values))
        i += 1 
        if i == 3:
            mu = np.mean(soft_vals)
            std = np.std(soft_vals)
            class_means.append(mu)
            class_stds.append(std)
    else:
        class_means.append(mu)
        class_stds.append(std)

#%% Compute class ranges by shifting all cells and dividing by two

t_background = -200
t_fat = -30
t_soft = 91.0
t_trabec = 453
class_ranges = [t_background, t_fat, t_soft, t_trabec]

# Show separation in histogram
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
for class_range in class_ranges:
    plt.axvline(class_range, color='black')
plt.show()


# Create class images
background_img = img < t_background
fat_img = (img >= t_background) & (img < t_fat)
soft_img = (img >= t_fat) & (img < t_soft)
trabec_img = (img >= t_soft) & (img < t_trabec)
bone_img = img >= t_trabec

#Print result
label_img = fat_img + 2 * soft_img + 3 * bone_img + 4 * trabec_img
image_label_overlay_minDist = label2rgb(label_img)
show_comparison(img, image_label_overlay_minDist, 'Classification result')


# %% Parametric classification =================================================
#New threshold values base on the intersections of the gaussians
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

#Change xlim to see the intersection
plt.xlim(50, 400) # Exclude background pixels
plt.legend()

t_background = -200
t_fat = -40
t_soft = 85
t_trabec = 300
class_ranges = [t_background, t_fat, t_soft, t_trabec]
background_img = img < t_background
fat_img = (img >= t_background) & (img < t_fat)
soft_img = (img >= t_fat) & (img < t_soft)
trabec_img = (img >= t_soft) & (img < t_trabec)
bone_img = img >= t_trabec

label_img = fat_img + 2 * soft_img + 3 * bone_img + 4 * trabec_img
image_label_overlay_ParCla = label2rgb(label_img)
show_comparison(image_label_overlay_minDist,image_label_overlay_ParCla, 'Classification result')



# %% Find optimal class ranges between fat soft tissue and bone
#Compute class ranges
class_means = []
class_stds = []
soft_vals = np.zeros(3)
bone_vals = np.zeros(2)
i=0
j=0
for organ in organs:
    mask = io.imread(in_dir + organ + 'ROI.png') > 0
    values = img[mask]
    mu = np.mean(values)
    std = np.std(values)
    if organ == 'Kidney' or organ == 'Spleen' or organ == 'Liver':
        soft_vals = np.concatenate((soft_vals, values))
        i += 1 
        if i == 3:
            mu = np.mean(soft_vals)
            std = np.std(soft_vals)
            class_means.append(mu)
            class_stds.append(std)
    elif organ == 'Trabec' or organ == 'Bone':
        bone_vals = np.concatenate((bone_vals, values))
        j += 1
        if j==2:
            mu = np.mean(bone_vals)
            std = np.std(bone_vals)
            class_means.append(mu)
            class_stds.append(std)
    else:
        class_means.append(mu)
        class_stds.append(std)

#Get values for the classes
mu_background, mu_bone,mu_fat, mu_soft = class_means
std_background, std_bone,std_fat, std_soft = class_stds
for test_value in range(-100,164):
    if (norm.pdf(test_value, mu_soft, std_soft) > norm.pdf(test_value, mu_bone, std_bone)) & (norm.pdf(test_value, mu_soft, std_soft) > norm.pdf(test_value, mu_fat, std_fat)):
        print(f"For value {test_value} the class is soft tissue")
    elif norm.pdf(test_value, mu_bone, std_bone) > norm.pdf(test_value, mu_fat, std_fat):
        print(f"For value {test_value} the class is bone")
    else:
        print(f"For value {test_value} the class is fat")

t_background = -200
t_fat = -71
t_bone1 = 4
t_soft = 80

background_img = img < t_background
fat_img = (img >= t_background) & (img < t_fat)
bone_img1 = (img >= t_fat) & (img < t_bone1)
soft_img = (img >= t_bone1) & (img < t_soft)
bone_img2 = img >= t_soft

label_img = fat_img + 2 * soft_img + 3 * (bone_img1+bone_img2) 
image_label_overlay_ParCla = label2rgb(label_img)
show_comparison(image_label_overlay_minDist,image_label_overlay_ParCla, 'Classification result')


#%% Object segmentation

