#%% Importing libraries
from skimage import io
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.decomposition import PCA
from skimage.transform import SimilarityTransform
from skimage.transform import warp
import os
import pathlib

#%% Exercise 1: Prepocess all images in the folder =================================
def read_landmark_file(file_name):
    f = open(file_name, 'r')
    lm_s = f.readline().strip().split(' ')
    n_lms = int(lm_s[0])
    if n_lms < 3:
        print(f"Not enough landmarks found")
        return None

    new_lms = 3
    # 3 landmarks each with (x,y)
    lm = np.zeros((new_lms, 2))
    for i in range(new_lms):
        lm[i, 0] = lm_s[1 + i * 2]
        lm[i, 1] = lm_s[2 + i * 2]
    return lm

def align_and_crop_one_cat_to_destination_cat(img_src, lm_src, img_dst, lm_dst):
    """
    Landmark based alignment of one cat image to a destination
    :param img_src: Image of source cat
    :param lm_src: Landmarks for source cat
    :param lm_dst: Landmarks for destination cat
    :return: Warped and cropped source image. None if something did not work
    """
    tform = SimilarityTransform()
    tform.estimate(lm_src, lm_dst)
    warped = warp(img_src, tform.inverse, output_shape=img_dst.shape)

    # Center of crop region
    cy = 185
    cx = 210
    # half the size of the crop box
    sz = 180
    warp_crop = warped[cy - sz:cy + sz, cx - sz:cx + sz]
    shape = warp_crop.shape
    if shape[0] == sz * 2 and shape[1] == sz * 2:
        return img_as_ubyte(warp_crop)
    else:
        print(f"Could not crop image. It has shape {shape}. Probably to close to border of image")
        return None
    
def preprocess_all_cats(in_dir, out_dir):
    """
    Create aligned and cropped version of image
    :param in_dir: Where are the original photos and landmark files
    :param out_dir: Where should the preprocessed files be placed
    """
    dst = "data/ModelCat"
    dst_lm = read_landmark_file(f"{dst}.jpg.cat")
    dst_img = io.imread(f"{dst}.jpg")

    all_images = glob.glob(in_dir + "*.jpg")
    for img_idx in all_images:
        name_no_ext = os.path.splitext(img_idx)[0]
        base_name = os.path.basename(name_no_ext)
        out_name = f"{out_dir}/{base_name}_preprocessed.jpg"

        src_lm = read_landmark_file(f"{name_no_ext}.jpg.cat")
        src_img = io.imread(f"{name_no_ext}.jpg")

        proc_img = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
        if proc_img is not None:
            io.imsave(out_name, proc_img)

# Preprocess all images
in_dir = "data/TrainingData100Cats/"
out_dir = "data/100CatsPreprocessed"
# Create output directory if it does not exist
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

preprocess_all_cats(in_dir, out_dir)





#%%% Exercise 2: Construct data matrix =======================================

#read first image to get the needed size of the data matrix
im_org = io.imread(glob.glob("data/100CatsPreprocessed/" + "*.jpg")[0])
n_samples = 100 # number of images
n_features = np.prod(im_org.shape) # number of pixels in each image

#Create empty matrix
X = np.zeros((n_samples,n_features), dtype = np.uint8)

#Convert all images to vectors and store in matrix
all_images = glob.glob("data/100CatsPreprocessed/" + "*.jpg")
for i in range(len(all_images)):
    flat_img = io.imread(all_images[i]).flatten()
    X[i,:] = flat_img

# %% Exercise 3 and 4: Compute the average cat =================================================

#Compute the average cat
avg_cat = np.mean(X, axis=0)
avg_cat = np.std(X, axis=0) #Standard deviation cat

def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out

#Create image from average cat
avg_cat_img = create_u_byte_image_from_vector(avg_cat, im_org.shape[0],
                                               im_org.shape[1], im_org.shape[2])
#Show the average cat
io.imshow(avg_cat_img)
# %% Exercise 5: SSD copmarison of missing cats
