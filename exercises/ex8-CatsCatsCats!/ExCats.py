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
os.chdir(r'C:\Users\askeb\OneDrive - Danmarks Tekniske Universitet\DTU\9. Semester\Image analysis\Scripts og data\DTUImageAnalysis\exercises\ex8-CatsCatsCats_mangler')

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


def preprocess_one_cat(name,out_name):
    dst = "data/ModelCat"
    dst_lm = read_landmark_file(f"{dst}.jpg.cat")
    dst_img = io.imread(f"{dst}.jpg")

    src_lm = read_landmark_file(f"{name}.jpg.cat")
    src_img = io.imread(f"{name}.jpg")

    proc_img = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
    if proc_img is not None:
        io.imsave(out_name, proc_img)




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
    flat_img = flat_img.reshape(1, -1)
    X[i,:] = flat_img

# %% Exercise 3 and 4: Compute the average cat =================================================

#Compute the average cat
avg_cat = np.mean(X, axis=0)

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
# %% Exercise 5: SSD copmarison of missing cats'
#Preprocess missing cat
preprocess_one_cat("data/MissingCat","data/MissingCat" + "_preprocessed.jpg")

#%%Flatten the missing cat to a vector of pixel values
mis_img = io.imread("data/MissingCat_preprocessed.jpg")
mis_img_flat = mis_img.flatten()
mis_img_flat
#%% Subtract missing cat from all rows in the data matrix and compute Sum of squared
#differences for each cat
sub_data = X - mis_img_flat
SSD = np.sum(sub_data**2,axis=1)

# %% Find least different cat
idx_similarCat = np.argmin(SSD)
similarCat_vec = X[idx_similarCat,:] 
sim_cat_img = create_u_byte_image_from_vector(similarCat_vec, im_org.shape[0],
                                               im_org.shape[1], im_org.shape[2])
io.imshow(sim_cat_img)
io.imshow(mis_img)
#Probably not a very convicing substitute

# %% least alike cat
idx_difCat = np.argmax(SSD)
difCat_vec = X[idx_difCat,:] 
dif_cat_img = create_u_byte_image_from_vector(difCat_vec, im_org.shape[0],
                                               im_org.shape[1], im_org.shape[2])
io.imshow(dif_cat_img)

#Also very different


#%% PCA 50 components
print("Computing PCA")
cats_pca = PCA(n_components=50)
cats_pca.fit(X)
#%%
plt.plot(cats_pca.explained_variance_ratio_)
plt.plot(np.cumsum(cats_pca.explained_variance_ratio_),'o')
plt.grid()

#First three PCA cats explain
np.round(cats_pca.explained_variance_ratio_[0:3]*100,2)
#%

# %%Project the cats onto PCA space
components = cats_pca.transform(X)

# %% Plot the cats on the first two dimensions
pc_1 = components[:, 0]
pc_2 = components[:,1]

plt.scatter(pc_1,pc_2)
plt.xlabel('PC1')
plt.ylabel("PC2")
plt.show()

# %% Find cats in space

#Most extreme cats
def PrintPCAcat(idx):
    img = create_u_byte_image_from_vector(X[idx],im_org.shape[0],
                                            im_org.shape[1],im_org.shape[2])
    plt.subplot(1,2,1)
    io.imshow(img)
    plt.subplot(1,2,2)
    plt.scatter(pc_1,pc_2,color="blue")
    plt.scatter(pc_1[idx],pc_2[idx],color="red")
    plt.xlabel('PC1')
    plt.ylabel("PC2")
    plt.show()

PrintPCAcat(np.argmin(pc_1))
PrintPCAcat(np.argmin(pc_2))
PrintPCAcat(np.argmax(pc_1))
PrintPCAcat(np.argmax(pc_2))
PrintPCAcat(np.argmax(pc_1+pc_2))

# %%Synthesize cats from the principal components
def synthesizeCat(weights,PCs):
    synth_cat = avg_cat + weights @ cats_pca.components_[PCs, :]
    synth_cat_im = create_u_byte_image_from_vector(synth_cat, im_org.shape[0],im_org.shape[1],im_org.shape[2])
    io.imshow(synth_cat_im)

synthesizeCat([-20000,-20000,-30000],[0,1,3])
# %% PLot the modes of variation
def modePlot(m):
    synth_cat_plus = avg_cat + 3 * np.sqrt(cats_pca.explained_variance_[m]) * cats_pca.components_[m, :]
    synth_cat_minus = avg_cat - 3 * np.sqrt(cats_pca.explained_variance_[m]) * cats_pca.components_[m, :]
    img_plus = create_u_byte_image_from_vector(synth_cat_plus, im_org.shape[0],im_org.shape[1],im_org.shape[2])
    img_minus = create_u_byte_image_from_vector(synth_cat_minus, im_org.shape[0],im_org.shape[1],im_org.shape[2])
    img_avg = create_u_byte_image_from_vector(avg_cat, im_org.shape[0],im_org.shape[1],im_org.shape[2])
    plt.subplot(1,3,1)
    io.imshow(img_minus)
    plt.subplot(1,3,2)
    io.imshow(img_avg)
    plt.title(f"PC{m+1}")
    plt.subplot(1,3,3)
    io.imshow(img_plus)
#%%
modePlot(0)
#%%
modePlot(1)
#%%
modePlot(2)
# %% Proper random cat synthesizer
n_components_to_use = 10
synth_cat = avg_cat
for idx in range(n_components_to_use):
	w = np.random.uniform(-1, 1) * 3 * np.sqrt(cats_pca.explained_variance_[idx])
	synth_cat = synth_cat + w * cats_pca.components_[idx, :]
synth_img = create_u_byte_image_from_vector(synth_cat, im_org.shape[0],im_org.shape[1],im_org.shape[2])
io.imshow(synth_img)
# %% Cat identification

#%% Find PCA coordinates of my missing cat
im_miss = io.imread("data/MissingCat_preprocessed.jpg")
im_miss_flat = im_miss.flatten()
im_miss_flat = im_miss_flat.reshape(1, -1)
pca_coords = cats_pca.transform(im_miss_flat)
pca_coords = pca_coords.flatten()

plt.scatter(pc_1,pc_2)
plt.scatter(pca_coords[0],pca_coords[1])

# %% Synthesize its closest neighbor in space
n_components_to_use = 50
synth_cat = avg_cat
for idx in range(n_components_to_use):
	synth_cat = synth_cat + pca_coords[idx] * cats_pca.components_[idx, :]
synth_img = create_u_byte_image_from_vector(synth_cat, im_org.shape[0],im_org.shape[1],im_org.shape[2])
io.imshow(synth_img)
#Not good
# %% Ex 27 Find the cats with the smallest and largest distance in space
comp_sub = components - pca_coords
pca_distances = np.linalg.norm(comp_sub, axis=1)
idx_l = np.argmax(pca_distances)
idx_s = np.argmin(pca_distances)

#largest distance in pca space
l_img = create_u_byte_image_from_vector(X[idx_l,:], im_org.shape[0],im_org.shape[1],im_org.shape[2])
#Smallest distance in space
s_img = create_u_byte_image_from_vector(X[idx_s,:], im_org.shape[0],im_org.shape[1],im_org.shape[2])

io.imshow(l_img)
# %%
io.imshow(s_img)
#Smallest distance is pretty good! 

#%% Find the ids of the n smallest disntances
n=3
for i in range(0,n):
    idx = np.argpartition(pca_distances,n+1)[0:n][i]
    img = create_u_byte_image_from_vector(X[idx,:],im_org.shape[0],im_org.shape[1],im_org.shape[2])
    plt.subplot(1,n,i+1)
    io.imshow(img)

# %%
