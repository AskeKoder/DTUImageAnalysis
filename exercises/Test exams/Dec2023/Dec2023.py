#%%% Problem 1
npixels = 2400*1200*3 #bytes/frame
tprocess = 0.130 #s/frame
Speed = 35*10**6 #byte/s

transferfps = Speed/npixels #frames per second
processfps = 1/tprocess #s/image



print(f'transferfps: {transferfps} frames per second')
print(f'processfps: {processfps} frames per second') 

#The slowest process dictates the speed therefore 4 fps
#%% Pixelwise operations

#read file
import skimage.io as io
import numpy as np
dir = 'data/Pixelwise/'
filename = 'ardeche_river.jpg'
img = io.imread(dir+filename)
#convert to grayscale
from skimage import color
img_gray = color.rgb2gray(img)

#Hist stretch
min_d = 0.2
max_d = 0.8
img_stretch = (max_d-min_d)/(np.max(img_gray)-np.min(img_gray))*(img_gray-np.min(img_gray))+min_d

#Compute average
avg = np.mean(img_stretch)

#apply prewitt filter
from skimage.filters import prewitt_h
img_filter = prewitt_h(img_stretch)

#Max absolute value of prewitt filter
prew_max = np.max(np.abs(img_filter))

#Binary image based on average threshold
thresh = avg
img_bin = img_stretch>thresh

#Number of foreground pixels 
n_for = np.sum(img_bin)

print(f'Number of foreground pixels {n_for}')
print(f'Maximum absolute value of prewitt filter {prew_max}')
print(f'Average of the stretched image {avg}')



# %% 3D medical image registration =========================
import matplotlib.pyplot as plt
import SimpleITK as sitk
from IPython.display import clear_output
from skimage.util import img_as_ubyte

#%% Useful functions
def imshow_orthogonal_view(sitkImage, origin = None, title=None):
    """
    Display the orthogonal views of a 3D volume from the middle of the volume.

    Parameters
    ----------
    sitkImage : SimpleITK image
        Image to display.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    data = sitk.GetArrayFromImage(sitkImage)

    if origin is None:
        origin = np.array(data.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    data = img_as_ubyte(data/np.max(data))
    axes[0].imshow(data[origin[0], ::-1, ::-1], cmap='gray')
    axes[0].set_title('Axial')

    axes[1].imshow(data[::-1, origin[1], ::-1], cmap='gray')
    axes[1].set_title('Coronal')

    axes[2].imshow(data[::-1, ::-1, origin[2]], cmap='gray')
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)

def overlay_slices(sitkImage0, sitkImage1, origin = None, title=None):
    """
    Overlay the orthogonal views of a two 3D volume from the middle of the volume.
    The two volumes must have the same shape. The first volume is displayed in red,
    the second in green.

    Parameters
    ----------
    sitkImage0 : SimpleITK image
        Image to display in red.
    sitkImage1 : SimpleITK image
        Image to display in green.
    origin : array_like, optional
        Origin of the orthogonal views, represented by a point [x,y,z].
        If None, the middle of the volume is used.
    title : str, optional
        Super title of the figure.

    Note:
    On the axial and coronal views, patient's left is on the right
    On the sagittal view, patient's anterior is on the left
    """
    vol0 = sitk.GetArrayFromImage(sitkImage0)
    vol1 = sitk.GetArrayFromImage(sitkImage1)

    if vol0.shape != vol1.shape:
        raise ValueError('The two volumes must have the same shape.')
    if np.min(vol0) < 0 or np.min(vol1) < 0: # Remove negative values - Relevant for the noisy images
        vol0[vol0 < 0] = 0
        vol1[vol1 < 0] = 0
    if origin is None:
        origin = np.array(vol0.shape) // 2

    sh = vol0.shape
    R = img_as_ubyte(vol0/np.max(vol0))
    G = img_as_ubyte(vol1/np.max(vol1))

    vol_rgb = np.zeros(shape=(sh[0], sh[1], sh[2], 3), dtype=np.uint8)
    vol_rgb[:, :, :, 0] = R
    vol_rgb[:, :, :, 1] = G

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(vol_rgb[origin[0], ::-1, ::-1, :])
    axes[0].set_title('Axial')

    axes[1].imshow(vol_rgb[::-1, origin[1], ::-1, :])
    axes[1].set_title('Coronal')

    axes[2].imshow(vol_rgb[::-1, ::-1, origin[2], :])
    axes[2].set_title('Sagittal')

    [ax.set_axis_off() for ax in axes]

    if title is not None:
        fig.suptitle(title, fontsize=16)

def composite2affine(composite_transform, result_center=None):
    """
    Combine all of the composite transformation's contents to form an equivalent affine transformation.
    Args:
        composite_transform (SimpleITK.CompositeTransform): Input composite transform which contains only
                                                            global transformations, possibly nested.
        result_center (tuple,list): The desired center parameter for the resulting affine transformation.
                                    If None, then set to [0,...]. This can be any arbitrary value, as it is
                                    possible to change the transform center without changing the transformation
                                    effect.
    Returns:
        SimpleITK.AffineTransform: Affine transformation that has the same effect as the input composite_transform.
    
    Source:
        https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/blob/master/Python/22_Transforms.ipynb
    """
    # Flatten the copy of the composite transform, so no nested composites.
    flattened_composite_transform = sitk.CompositeTransform(composite_transform)
    flattened_composite_transform.FlattenTransform()
    tx_dim = flattened_composite_transform.GetDimension()
    A = np.eye(tx_dim)
    c = np.zeros(tx_dim) if result_center is None else result_center
    t = np.zeros(tx_dim)
    for i in range(flattened_composite_transform.GetNumberOfTransforms() - 1, -1, -1):
        curr_tx = flattened_composite_transform.GetNthTransform(i).Downcast()
        # The TranslationTransform interface is different from other
        # global transformations.
        if curr_tx.GetTransformEnum() == sitk.sitkTranslation:
            A_curr = np.eye(tx_dim)
            t_curr = np.asarray(curr_tx.GetOffset())
            c_curr = np.zeros(tx_dim)
        else:
            A_curr = np.asarray(curr_tx.GetMatrix()).reshape(tx_dim, tx_dim)
            c_curr = np.asarray(curr_tx.GetCenter())
            # Some global transformations do not have a translation
            # (e.g. ScaleTransform, VersorTransform)
            get_translation = getattr(curr_tx, "GetTranslation", None)
            if get_translation is not None:
                t_curr = np.asarray(get_translation())
            else:
                t_curr = np.zeros(tx_dim)
        A = np.dot(A_curr, A)
        t = np.dot(A_curr, t + c - c_curr) + t_curr + c_curr - c

    return sitk.AffineTransform(A.flatten(), t, c)

# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations
    
    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations
    
    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations
    
    metric_values.append(registration_method.GetMetricValue())                                       
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()
    
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
# metric_values list. 
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))

def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )




#%% solution
dir_in = 'data/ImageRegistration/'
ImgV1 = sitk.ReadImage(dir_in + 'ImgT1_v1.nii.gz')

ImgV2 = sitk.ReadImage(dir_in + 'ImgT1_v2.nii.gz')

# Given affine matrix 
A = np.array([[0.98, -0.16, 0.17,0],
               [0.26, 0.97, 0,-15],
                 [-0.17, 0.04, 0.98,0],
                 [0,0,0,1]])
transform = sitk.AffineTransform(3)
transform.SetTranslation([0,-15,0])
centre_image = np.array(ImgV1.GetSize()) / 2 - 0.5 # Image Coordinate System
centre_world = ImgV1.TransformContinuousIndexToPhysicalPoint(centre_image) # World Coordinate System
rot_matrix = A[:3, :3] # SimpleITK inputs the rotation and the translation separately

transform.SetCenter(centre_world) # Set the rotation centre
transform.SetMatrix(rot_matrix.flatten())

# Apply the transformation to the image
ImgV1_A = sitk.Resample(ImgV1, transform)
imshow_orthogonal_view(ImgV1_A, title='ImgT1_A', origin=None)


#%% Rigid euler transform
fixed_image = ImgV1
moving_image = ImgV2

R = sitk.ImageRegistrationMethod()

# Set a one-level the pyramid scheule. [Pyramid step]
R.SetShrinkFactorsPerLevel(shrinkFactors = [2])
R.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

# Set the interpolator [Interpolation step]
R.SetInterpolator(sitk.sitkLinear)

# Set the similarity metric [Metric step]
R.SetMetricAsMeanSquares()

# Set the sampling strategy [Sampling step]
R.SetMetricSamplingStrategy(R.RANDOM)
R.SetMetricSamplingPercentage(0.50)

# Set the optimizer [Optimization step]
R.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)

# Initialize the transformation type to rigid 
initTransform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
R.SetInitialTransform(initTransform, inPlace=False)

# Some extra functions to keep track to the optimization process 
# R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R)) # Print the iteration number and metric value
R.AddCommand(sitk.sitkStartEvent, start_plot) # Plot the similarity metric values across iterations
R.AddCommand(sitk.sitkEndEvent, end_plot)
R.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
R.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(R))

# Estimate the registration transformation [metric, optimizer, transform]
tform_reg = R.Execute(fixed_image, moving_image)

# Apply the estimated transformation to the moving image
ImgT1_B = sitk.Resample(moving_image, tform_reg)
params = tform_reg.GetParameters() # Parameters (Rx, Ry, Rz, Tx, Ty, Tz)
print(params)

# %% Manual registration
thresh = 50
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

A_rot = rotation_matrix(0,-20,0)

transform = sitk.AffineTransform(3)
rot_matrix = A_rot[:3, :3] # SimpleITK inputs the rotation and the translation separately

#Default center
#transform.SetCenter(centre_world) # Set the rotation centre
transform.SetMatrix(rot_matrix.T.flatten()) #.T because inverse rotation

# Apply the transformation to the image
moving_image_reg= sitk.Resample(moving_image, transform)
imshow_orthogonal_view(moving_image_reg, title='ImgT1_A', origin=None)

mask = sitk.GetArrayFromImage(fixed_image) > 50
fixedImageNumpy = sitk.GetArrayFromImage(fixed_image)
movingImageNumpy = sitk.GetArrayFromImage(moving_image_reg)

fixedImageVoxels = fixedImageNumpy[mask]
movingImageVoxels = movingImageNumpy[mask]
mse = np.mean((fixedImageVoxels - movingImageVoxels)**2)
print('Anwer: MSE = {:.2f}'.format(mse))

# %% Change detection 
dir = 'data/ChangeDetection/'
Img1 = io.imread(dir+'frame_1.jpg')
Img2 = io.imread(dir+'frame_2.jpg')

#Convert to HSV
Img1_hsv = color.rgb2hsv(Img1)
Img2_hsv = color.rgb2hsv(Img2)

#Extract S channel
S1 = Img1_hsv[:,:,1]*255
S2 = Img2_hsv[:,:,1]*255

# COmpute absolute difference image
absdif = np.abs(S1-S2)
io.imshow(absdif)

# Avg and std
avg_dif = np.mean(absdif)
sd_dif = np.std(absdif)

#Tresh
thresh = avg_dif + 2*sd_dif

# Change image
Ch_img = absdif > thresh
io.imshow(Ch_img)

#Number of changed pixel
N_ch = np.sum(Ch_img)

#Blob analysis
from skimage import measure
label_img = measure.label(Ch_img) #Label connected regions
n_labels = label_img.max()
print(f"Number of labels: {n_labels}")
# Display the labels
im_labels = color.label2rgb(label_img) # Colorize the labels
io.imshow(im_labels)
#Compute blob areas
region_props = measure.regionprops(label_img)
areas = np.array([prop.area for prop in region_props])

# Blob with largest area
idmax = np.argmax(areas)
maxarea = areas[idmax]

print(f'area of the largest blob: {maxarea}')
print(f'Threshold: {thresh}')
print(f'Number of changed pixels: {N_ch}')
# %% Heart analysis ===== 
import pydicom as dicom

#Read dicom image
dir = 'data/HeartCT/'
ct = dicom.read_file(dir + '1-001.dcm')
#Get pixel values
img = ct.pixel_array

my_ROI = io.imread(dir + 'MyocardiumROI.png')
blood_ROI =io.imread(dir+'BloodROI.png')

#Get values in region of interest
my_values = img[my_ROI]
blood_values = img[blood_ROI]

#Get averages and sds for the classes
my_avg = np.mean(my_values)
my_sd = np.std(my_values)
blood_avg = np.mean(blood_values)
blood_sd = np.std(blood_values)

#Separate blood class
bin_img = (img > blood_avg-3*blood_sd) & (img < blood_avg+3*blood_sd)
io.imshow(bin_img)

#Clean up th image using morphological operations
from skimage import morphology
from skimage.morphology import disk 
footprint = disk(3)
im_closed = morphology.closing(bin_img, footprint)
footprint = disk(5)
im_clean = morphology.opening(im_closed, footprint)
io.imshow(im_clean)

#Blob analysis
label_img = measure.label(im_clean) #Label connected regions
n_labels = label_img.max()
print(f"Number of labels: {n_labels}")

# %% # Display the labels
im_labels = color.label2rgb(label_img) # Colorize the labels
io.imshow(im_labels)
# %% Compute blob features 
region_props = measure.regionprops(label_img)
areas = np.array([prop.area for prop in region_props])
blobs = np.where((areas < 5000)&(areas > 2000))

found = label_img == blobs[0][0]+1 # add one because background is not a blob
io.imshow(found)

#Compare
Gt = io.imread(dir+'BloodGT.png')
io.imshow(Gt)
gt_bin = Gt > 0
from scipy.spatial import distance
dice_score = 1 - distance.dice(found.ravel(), gt_bin.ravel())
print(f"DICE score {dice_score}")


#%%Minimum distance classfication
#Get middle value
thresh = (my_avg+blood_avg)/2

# %% PCA on pistachios
import pandas as pd
X = pd.read_csv('data/pistachio/pistachio_data.txt',sep=' ')
names = X.iloc[:,1:13].columns
X = X.values[:,0:12] # A row has shifted itself
#Standardize
X = (X - np.mean(X, axis=0))
std = np.std(X, axis=0)
X = X/std

names[np.where(std==np.min(std))]

#Do PCA
from sklearn.decomposition import PCA
print("Computing PCA")
pca = PCA(n_components=12)
pca.fit(X)
plt.plot(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_),'o')
plt.yticks([0.97])
plt.grid()

#Project all pistachios to PCA space
components = pca.transform(X)
#sum of squared distances for the first pistachio
print(np.sum(components[0,:]**2))

#Variable with smallest std
names[np.where(std==np.min(std))]

#Covariance matrix
np.max(np.abs((X.T@X)/len(X)))


# %% Set up landmarks
dst = np.array([[1,0], [2,4], [3,6], [4,4], [5,0]])
src = np.array([[3,1], [3.5,3], [4.5,6], [5.5,5], [7,1]])
plt.plot(dst[:, 0], dst[:, 1], '.r', markersize=12,color="Green")
plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
plt.show()

#Compute sum of squares before
SS_before = np.sum((src[:,0]-dst[:,0])**2 + (src[:,1]-dst[:,1])**2)
print(SS_before)

cm_1 = np.mean(src, axis=0)
cm_2 = np.mean(dst, axis=0)
translations = cm_2 - cm_1
print(f"Answer: translation {translations}")

#similarity transform
tform = SimilarityTransform()
tform.estimate(src, dst)
tform.rotation*180/np.pi
# %% Fish
dir = "data/Fish/"
filenames = ["discus.jpg", "guppy.jpg", "kribensis.jpg", "neon.jpg", "oscar.jpg", "platy.jpg","rummy.jpg", "scalare.jpg", "tiger.jpg", "zebra.jpg"]

#Empty set
im_org = io.imread(dir+filenames[0])
X = np.zeros((len(filenames),im.flatten().reshape(1, -1).shape[1]))
i=0
for filename in filenames:
    temp = io.imread(dir+filename).flatten()
    X[i,:] = temp.reshape(1, -1)
    i+=1

#Image of average fish
X_avg = np.mean(X, axis=0)
def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out

avg_fish_img = create_u_byte_image_from_vector(X_avg, im_org.shape[0],
                                               im_org.shape[1], im_org.shape[2])
io.imshow(avg_fish_img)


#Compute PCA
from sklearn.decomposition import PCA
print("Computing PCA")
pca = PCA(n_components=6)
pca.fit(X)
plt.plot(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_),'o')
plt.grid()


#
im_gup = io.imread(dir+filenames[1])
im_neon = io.imread(dir+filenames[3])

#Sum of squared differences
np.sum((im_gup-im_neon)**2)


#Project fish to pca space
components = pca.transform(X)

#neon fish
pca_neon = components[3,:]

id = np.argmax(np.sum((pca_neon - components)**2,axis=1))
print(filenames[id])
# %%
