#%% Import
import matplotlib.pyplot as plt
import math
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import swirl
from skimage import io
import numpy as np

def show_comparison(original, transformed, transformed_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(transformed)
    ax2.set_title(transformed_name)
    ax2.axis('off')
    io.show()

import os
os.chdir(r'C:\Users\askeb\OneDrive - Danmarks Tekniske Universitet\DTU\9. Semester\Image analysis\Scripts og data\DTUImageAnalysis\exercises\Ex7-GeometricTransformationsAndRegistration_mangler')

#%% Image rotation
# Load an image
in_dir = 'data/'
im_org = io.imread(in_dir + 'NusaPenida.png')

# angle in degrees - counter clockwise around the center of the image
rotation_angle = 10
rot_center = [0,0]
rotated_img = rotate(im_org, rotation_angle,center = rot_center)
show_comparison(im_org, rotated_img, "Rotated image")
# %% Try som more
# angle in degrees - counter clockwise around the center of the image
rotation_angle = 10
rot_center = [200,0]
rotated_img = rotate(im_org, rotation_angle,center = rot_center, mode="reflect")
show_comparison(im_org, rotated_img, "Rotated image")
#Reflect reflects the outer pixels

rotation_angle = 10
rot_center = [200,0]
rotated_img = rotate(im_org, rotation_angle,center = rot_center, mode="wrap")
show_comparison(im_org, rotated_img, "Rotated image")
#Wrap places it in a grid of the same image

# %% Backgroud can also be replace with constant pixel values
rotated_img = rotate(im_org, rotation_angle,resize=True, mode="constant", cval=0.3)
show_comparison(im_org, rotated_img, "Rotated image")
#cval = takes a float value [0,1] to fill the empty pixels with

# Notice the automatic resizing function, that fits the rotated image to the frame
# %% Rigid body transformation - euclidiant transformation
# angle in radians - counter clockwise
rotation_angle = 10.0 * math.pi / 180.
trans = [10, 20]
tform = EuclideanTransform(rotation=rotation_angle, translation=trans)
print(tform.params)

#3x3 Used for homogenous coordinates 

#apply transformation to the image
transformed_img = warp(im_org, tform)
show_comparison(im_org, transformed_img, "Transformed image")

# Actually an inverse transformation is applied to the image when using warp

#Here we invert the transformation
transformed_img = warp(im_org, tform.inverse)
show_comparison(im_org, transformed_img, "Transformed image")

# %% Test transformation with only rotation
rotation_angle = 10.0 * math.pi / 180.
tform = EuclideanTransform(rotation=rotation_angle)
transformed_img = warp(im_org, tform)
show_comparison(im_org, transformed_img, "Transformed image")

# inverse transformation
transformed_img = warp(im_org, tform.inverse)
show_comparison(im_org, transformed_img, "Transformed image")

# When using warp the rotation is centered around the upper left corner 
# %% Similarity transformation ( translation, rotation, scaling)
rotation_angle = 15.0 * math.pi / 180
scaling = 1/0.6
translation = [40,30]
tform = SimilarityTransform(rotation=rotation_angle, scale=scaling, translation=translation)
print(tform.params)
transformed_img = warp(im_org, tform)
show_comparison(im_org, transformed_img, "Transformed image")

#scale < 1 will make the image bigger, 1> will make it smaller

# %% Swirl transformation
str = 100
rad = 200
c = [300, 400]
swirl_img = swirl(im_org, strength=str, radius=rad, center=c)
show_comparison(im_org, swirl_img, "Swirl image")

# %% Landmark based registration ============================================

#Load hand images
src_img = io.imread(in_dir + 'Hand1.jpg') #source ( to fit on top of hand 2)
dst_img = io.imread(in_dir + 'Hand2.jpg') #destination 

#Blend images
from skimage.util import img_as_float, img_as_ubyte
blend = 0.5 * img_as_float(src_img) + 0.5 * img_as_float(dst_img)
io.imshow(blend)
io.show()
# Not the best fit yet
# %% Manually place landmarks on source image
src = np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]])
plt.imshow(src_img)
plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
plt.show()

# %% Exercise 13 :Now place landmarks on destination image
dst = np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]])
plt.imshow(dst_img)
plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
plt.show()

# %%
