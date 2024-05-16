#%%Abdominal CT analysis
import pydicom as dicom
from skimage import io
import numpy as np
import os
os.chdir(r'C:\Users\askeb\OneDrive - Danmarks Tekniske Universitet\DTU\9. Semester\Image analysis\Scripts og data\DTUImageAnalysis\exercises\Test exams\May2023')

#read image and ROI
dir = "data/Abdominal/"
ct = dicom.read_file(dir + '1-166.dcm')
img = ct.pixel_array

LKidROI = io.imread(dir + 'KidneyRoi_l.png')
RKidROI = io.imread(dir + 'KidneyRoi_r.png')
AoROI = io.imread(dir + 'AortaROI.png')
LivROI = io.imread(dir + 'LiverROI.png')

#display images
io.imshow(img, cmap='gray')
io.imshow(LKidROI+RKidROI+AoROI+LivROI)

#Extract pixel values from ROIs
LKid = img[LKidROI==1]
RKid = img[RKidROI==1]
Ao = img[AoROI==1]
Liv = img[LivROI==1]

#Hounsfield unit in right and left kidney
print(f'Left kidney: {np.mean(LKid)}')
print(f'Right kidney: {np.mean(RKid)}')

#Threshfold for the liver
t_1 = np.mean(Liv)-np.std(Liv)
t_2 = np.mean(Liv)+np.std(Liv)

#%%Binary image of liver
Bin_img = np.zeros(img.shape)
Bin_img[(img>t_1) & (img<t_2)] = 1

#Dilate 
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk 

footprint = disk(3)
dilated = dilation(Bin_img, footprint)
footprint = disk(10)
eroded = erosion(dilated, footprint)
dilated = dilation(eroded, footprint)
io.imshow(dilated)
# %% Extract BLOBs
from skimage.measure import label, regionprops
label_img = label(dilated)
props = regionprops(label_img)

io.imshow(label_img)

#Get areas
areas = np.array([prop.area for prop in props])
perims = np.array([prop.perimeter for prop in props])

#%%Filter blobs
id = np.where((areas>1500)&(areas<7000)&(perims>300))

filtered_img = label_img == id[0]+1 #Backgroud not included
io.imshow(filtered_img)

from scipy.spatial import distance
ground_truth_img = LivROI
dice_score = 1 - distance.dice(filtered_img.ravel(), ground_truth_img.ravel())
print(f"DICE score {dice_score}")



# %% Forensic Glass ====================
import pandas as pd
dir = "data/GlassPCA/"
X_org = pd.read_csv(dir + 'glass_data.txt', sep=' ')
#Columns are shifted
names =X_org.columns [1:len(X_org.columns)]
X = X_org.values[:,0:len(names)]

#Subtract mean from data
X = X - np.mean(X, axis=0)

#Comput min and max
min_val = np.min(X, axis=0)
max_val = np.max(X, axis=0)
dif  = max_val - min_val

#Divide by difference
X = X/dif

#%%Do PCA
#Covariance matrix
C = 1/(X.shape[0]-1)*np.dot(X.T,X)
C[0,0]
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=9)
pca.fit(X)


plt.plot(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_),'o')
plt.yticks([0.97])
plt.grid()


projection = pca.transform(X)
np.max(np.abs(projection))


# %% Advanced 3D image registration======================
import SimpleITK as sitk

def rotation_matrix(pitch,roll,yaw, degrees = True):
    if degrees:
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
        yaw = np.deg2rad(yaw)

    A_pitch = np.array([[1,0,0,0],
                        [0,np.cos(pitch),-np.sin(pitch),0],
                        [0,np.sin(pitch),np.cos(pitch),0],
                        [0,0,0,1]])
    A_roll = np.array([[np.cos(roll),0,np.sin(roll),0],
                       [0,1,0,0],
                       [-np.sin(roll),0,np.cos(roll),0],
                       [0,0,0,1]])
    A_yaw = np.array([[np.cos(yaw),-np.sin(yaw),0,0],
                      [np.sin(yaw),np.cos(yaw),0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
    A_rot = np.dot(A_pitch,np.dot(A_roll,A_yaw))
    return A_rot

T = np.array([[1,0,0,10],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

A = rotation_matrix(0,0,10) @T @rotation_matrix(0,30,0)



# %% Product names on boxes =========================== 

#read image
img = io.imread('nike.png')
io.imshow(img)
from skimage.color import rgb2hsv
hsv = rgb2hsv(img)
h_comp =hsv[:,:,0]

#Create binary img 0.3 < H < 0.7
bin_img = np.zeros(h_comp.shape)
bin_img[(h_comp>0.3) & (h_comp<0.7)] = 1
#dilation
footprint = disk(8)
dilated = dilation(bin_img, footprint)

#Number of foreground pixels
n_pix = np.sum(dilated)
print(f'Number of foreground pixels: {n_pix}')

# %% Shoe comparison

#read
dir = 'data/LMRegistration/'
src_img = io.imread(dir+'shoe_1.png')
dst_img = io.imread(dir + 'shoe_2.png')

#Given landmarks
src = np.array([[40, 320], [425, 120], [740, 330]])
dst = np.array([[80, 320], [380, 155], [670, 300]])

# %% Compute objective function ( How well they are aligned)
e_x = src[:, 0] - dst[:, 0]
error_x = np.dot(e_x, e_x)
e_y = src[:, 1] - dst[:, 1]
error_y = np.dot(e_y, e_y)
f1 = error_x + error_y
print(f"Landmark alignment error before: {f1}")

# %% Transform the image with euclidean transformation
from skimage.transform import SimilarityTransform
from skimage.transform import warp
tform = SimilarityTransform()
tform.estimate(src, dst)
src_transform = src @ tform.params[:2, :2].T + tform.params[0:2, 2]



e_x = src_transform[:, 0] - dst[:, 0]
error_x = np.dot(e_x, e_x)
e_y = src_transform[:, 1] - dst[:, 1]
error_y = np.dot(e_y, e_y)
f2 = error_x + error_y
print(f"Landmark alignment error after: {f2}")

#Scale of the transformation
scale = tform.scale
#Apply transformation to image
warped = warp(src_img, tform.inverse)


#%%Transform to ubyte
from skimage.util import img_as_ubyte
srcTu = img_as_ubyte(warped)
dstTu = img_as_ubyte(dst_img)
io.imshow(srcTu)
io.imshow(dstTu)

#Extract blue component
src_blue = srcTu[:,:,2]
dst_blue = dstTu[:,:,2]
print(np.abs(src_blue[200,200]-dst_blue[200,200]))
# %% Change detection algorithm ====================================

#read images
bk = io.imread('data/ChangeDetection/background.png')
new = io.imread('data/ChangeDetection/new_frame.png')

#Convert to grayscale
from skimage.color import rgb2gray
bk_gray = rgb2gray(bk)
new_gray = rgb2gray(new)

#Update background image
alpha = 0.90
bk_new = alpha * bk_gray + (1-alpha) * new_gray

#Absolute difference
diff = np.abs(bk_new - new_gray)

#Threshold
chpx = diff >0.1
print(f'Number of changed pixels: {np.sum(chpx)}')

#Average region value
print(f'Average region value: {np.mean(bk_gray[150:200, 150:200])}')
# %% Character recognition =========================================================
dir = 'data/Letters/'

#Read image
img = io.imread(dir + 'Letters.png')

#Extract RGB components
red = img[:,:,0]
green = img[:,:,1]
blue = img[:,:,2]

#Bin image >100
bin_img = np.zeros(img.shape[0:1])
bin_img = (red>100)&(green<100)&(blue<100)
io.imshow(bin_img)

#%%Erosion
footprint = disk(3)
eroded = erosion(bin_img, footprint)
io.imshow(eroded)
print(f'Number of foreground pixels: {np.sum(eroded)}')

#%% Preprocessing
img_gray = rgb2gray(img)

from skimage.filters import median
size = 8
footprint = np.ones([size, size])
med_im = median(img_gray, footprint)
io.imshow(med_im)

print(f'Pixel value at (100,100): {med_im[100,100]}')
# %% BLOB on eroded image
io.imshow(eroded)

#Find BLOBs
label_img = label(eroded)
props = regionprops(label_img)

#compute areas and perimeters
areas = np.array([prop.area for prop in props])
perims = np.array([prop.perimeter for prop in props])

#Remove all BLOBs with an area<1000 or an area>4000 or a perimeter<300

idx = np.where((areas>1000)&(areas<4000)&(perims>300))
filtered_img = (label_img == (idx[0]+1)[0])|(label_img == (idx[0]+1)[1])|(label_img == (idx[0]+1)[2])|(label_img == (idx[0]+1)[3])



# %% Surverillance system =======================================

# You are developing a new surveillance system that consists of a camera connected to a
# computer using a USB-2 connection. The images are RGB (8-bits per channel) images
# with a size of 1600 x 800 pixels. When you tested your system, you did some timings of
# your system. You found out that your image analysis algorithm takes 230 milliseconds
# to process one frame. You also found out that the camera can successfully send 6.25
# images per second to the computer (this is an average value measured over a minute).

size = 1600*800*3 #bytes
camrate = 6.25 #fps
algo_time = 0.230 #s
algo_rate = 1/algo_time #fps

print(f'framerate of system {np.min([camrate,algo_rate])} fps')

print(f'The camera transfers {size*camrate/(10**6)} megabytes per second')
# %% Pizza AI
dir = 'data/PizzaPCA/'
#Read data
import glob
all_images = glob.glob(dir +'training/'+ "*.png")

#Read images
images = []
for img in all_images:
    images.append(io.imread(img).reshape(-1,1))

org_shape = io.imread(img).shape 
#Convert to numpy array
images = np.array(images).squeeze()

avg = np.mean(images, axis=0)

def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out

avg_img = create_u_byte_image_from_vector(avg, org_shape[0],
                                               org_shape[1], org_shape[2])
io.imshow(avg_img.reshape(org_shape))


#run pca
pca = PCA(n_components=5)
pca.fit(images)
project = pca.transform(images)


#Compare find pizza that is furthest from the average
idx = np.argmax(np.sum((images - avg)**2,axis=1))
all_images[idx]

#First pricipal component
print(pca.explained_variance_ratio_)

#Find the most distant pizzas on first principal axis
#negative direction
idN = np.argmin(project[:,0])
#positive direction
idP = np.argmax(project[:,0])
#%%
print(all_images[idN])
print(all_images[idP])


#%% Find the super pizza
super_pizza = io.imread(dir + 'super_pizza.png').flatten().reshape(1,-1)
super_pizza_pca = pca.transform(super_pizza)

#Find the pizza that is closest to the super pizza
dist = np.sum((project - super_pizza_pca)**2, axis=1)
idSup = np.argmin(dist)
all_images[idSup]
# %% Concert lights
dir = 'data/GeomTrans/'
#Read images
img = io.imread(dir + 'lights.png')

#Rotate image
from skimage.transform import rotate
rot_img = rotate(img, 11, resize=False,center=(40,40))
io.imshow(rot_img)

# Convert to grayscale
from skimage.color import rgb2gray
img_gray = rgb2gray(rot_img)

#Threshold
from skimage.filters import threshold_otsu
thresh = threshold_otsu(img_gray)
bin_img = img_gray > thresh
print(f'Number of foreground pixels: {np.sum(bin_img)}')

#percentage of foreground pixels
n_pix = np.sum(bin_img)
n_pix_total = bin_img.size
print(f'Percentage of foreground pixels: {n_pix/n_pix_total*100}%')

# %%
