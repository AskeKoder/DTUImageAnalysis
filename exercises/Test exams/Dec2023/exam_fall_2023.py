import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from skimage.morphology import erosion, dilation, binary_closing, binary_opening
from skimage.morphology import disk
from skimage.morphology import square
from skimage.filters import median
from scipy.stats import norm
from skimage import color, io, measure, img_as_ubyte, img_as_float
from skimage.filters import threshold_otsu
from scipy.spatial import distance
from skimage.transform import rotate
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import matrix_transform
import glob
from sklearn.decomposition import PCA
import random
from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
import SimpleITK as sitk

def pca_on_pistachio_e_2023():
    in_dir = "data/pistachio/"
    txt_name = "pistachio_data.txt"
    pistachio_data = np.loadtxt(in_dir + txt_name, comments="%")
    x = pistachio_data
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")
    mn = np.mean(x, axis=0)
    data = x - mn
    std = np.std(data, axis=0)
    min_std = np.min(std)
    print(f"Answer: Minimum standard deviation {min_std:.2f} of ECCENTRICITY")
    data = data / std
    c_x = np.cov(data.T)
    max_cov = abs(np.max(c_x))
    min_cov = abs(np.min(c_x))
    max_ans = max(max_cov, min_cov)


    print(f"Answer: Max covariance matrix value: {max_ans:.3f}")

    # print(f"Answer: Covariance matrix at (0, 0): {c_x[0][0]:.3f}")

    values, vectors = np.linalg.eig(c_x)
    v_norm = values / values.sum() * 100
    # plt.plot(v_norm)
    # plt.xlabel('Principal component')
    # plt.ylabel('Percent explained variance')
    # plt.ylim([0, 100])
    # plt.show()

    answer = v_norm[0] + v_norm[1] + v_norm[2] + v_norm[3]
    print(f"Answer: Variance explained by the first four PC: {answer:.2f}")

    answer = v_norm[0] + v_norm[1] + v_norm[2] + v_norm[3] + v_norm[4]
    print(f"Answer: Variance explained by the first five PC: {answer:.2f}")

    # Project data
    pc_proj = vectors.T.dot(data.T)

    first_proj = pc_proj[:, 0]
    sum_sq = np.sum(first_proj ** 2)
    print(f"Answer: Sum of squares of first projected data {sum_sq:.2f}")


def change_detection_e_2023():
    name_1 = 'data/ChangeDetection/frame_1.jpg'
    name_2 = 'data/ChangeDetection/frame_2.jpg'

    im_1 = io.imread(name_1)
    im_2 = io.imread(name_2)

    hsv_img_1 = color.rgb2hsv(im_1)
    hsv_img_2 = color.rgb2hsv(im_2)

    im_1_g = hsv_img_1[:, :, 1] * 255
    im_2_g = hsv_img_2[:, :, 1] * 255

    dif_img = np.abs(im_1_g - im_2_g)
    average_change_val = np.average(dif_img)
    std_dev_change_val = np.std(dif_img)
    print(f"Answer: Average change value {average_change_val:.2f} and standard deviation {std_dev_change_val:.2f}")

    dif_thres = average_change_val + 2 * std_dev_change_val
    print(f"Difference trehsold {dif_thres:.2f}")
    dif_bin = (dif_img > dif_thres)
    io.imshow(dif_bin)
    io.show()
    changed_pixels = np.sum(dif_bin)
    print(f"Answer: Changed pixels {changed_pixels:.0f}")

    label_img = measure.label(dif_bin)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")
    region_props = measure.regionprops(label_img)
    max_area = -1
    for region in region_props:
        if region.area > max_area:
            max_area = region.area
    print(f"Answer: Area of the largest region {max_area:.0f}")


def system_frame_rate_e_2023():
    # bytes per second
    transfer_speed = 35000000
    image_mb = 2400 * 1200 * 3
    images_per_second = transfer_speed / image_mb
    print(f"Images transfered per second {images_per_second:.3f}")

    proc_time = 0.130
    proc_per_second = 1/proc_time
    print(f"Images processed per second {proc_per_second:.1f}")

    max_fps = min(proc_per_second, images_per_second)
    print(f"System framerate {max_fps:.1f}")


def landmark_based_registration_e_2023():
    # Stick man
    src = np.array([[3, 1], [3.5, 3], [4.5, 6], [5.5, 5], [7, 1]])
    dst = np.array([[1, 0], [2, 4], [3, 6], [4, 4], [5, 0]])

    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Answer: Landmark alignment error F before: {f}")

    # plt.imshow(src_img)
    # plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
    # plt.title("Source image")
    # plt.show()

    fig, ax = plt.subplots()
    ax.plot(src[:, 0], src[:, 1], '*r', markersize=12, label="Source")
    ax.plot(dst[:, 0], dst[:, 1], '*g', markersize=12, label="Destination")
    # ax.invert_yaxis()
    ax.legend()
    ax.set_title("Landmarks before alignment")
    plt.show()

    cm_1 = np.mean(src, axis=0)
    cm_2 = np.mean(dst, axis=0)
    translations = cm_2 - cm_1
    print(f"Answer: translation {translations}")
    # src_transform = src + translations

    tform = SimilarityTransform()
    tform.estimate(src, dst)
    print(f"Answer: rotation {(tform.rotation * 180 / np.pi):.2f} degrees")

    src_transform = matrix_transform(src, tform.params)

    e_x = src_transform[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src_transform[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f_after = error_x + error_y
    print(f"Aligned landmark alignment error F: {f_after}")


def pixel_value_mappings_and_filtering_e_2023():
    in_dir = "data/Pixelwise/"
    im_name = "ardeche_river.jpg"
    im_org = io.imread(in_dir + im_name)
    gray_img = color.rgb2gray(im_org)
    # gray_img = img_as_ubyte(gray_img)

    io.imshow(gray_img)
    plt.title('Gray image')
    io.show()

    # img_float = img_as_float(img_in)
    min_val = gray_img.min()
    max_val = gray_img.max()
    min_desired = 0.2
    max_desired = 0.8

    print(f"Float image minimum pixel value: {min_val} and max value: {max_val}")

    img_out = (max_desired - min_desired) / (max_val - min_val) * (gray_img - min_val) + min_desired

    min_val = img_out.min()
    max_val = img_out.max()
    print(f"Out float image minimum pixel value: {min_val} and max value: {max_val}")

    average_value = np.average(img_out)
    print(f"Answer: image average value {average_value:.2f}")

    edge_img_h = prewitt_h(img_out)
    # io.imshow(img_as_ubyte(edge_img_h))
    io.imshow(edge_img_h)
    plt.title('Prewitt H filtered image')
    io.show()

    max_edge = np.max(np.abs(edge_img_h))
    print(f"Answer: max edge value {max_edge:.2f}")

    threshold = average_value
    img_bin = img_out > threshold

    io.imshow(img_bin)
    plt.title('Bin image')
    io.show()

    print(f"Answer: number of pixels in binary image {img_bin.sum()}")


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
    On the sagittal view, patient's anterior is on the left    """
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
    # min_val = np.min([np.min(vol0), np.min(vol1)])
    # max_val = np.max([np.max(vol0), np.max(vol1)])
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


def rotation_matrix(pitch, roll, yaw, deg=False):
    """
    Return the rotation matrix associated with the Euler angles roll, pitch, yaw.

    Parameters
    ----------
    pitch : float
        The rotation angle around the x-axis.
    roll : float
        The rotation angle around the y-axis.
    yaw : float
        The rotation angle around the z-axis.
    deg : bool, optional
        If True, the angles are given in degrees. If False, the angles are given
        in radians. Default: False.
    """
    if deg:
        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

    R_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(pitch), -np.sin(pitch), 0],
                    [0, np.sin(pitch), np.cos(pitch), 0],
                    [0, 0, 0, 1]])

    R_y = np.array([[np.cos(roll), 0, np.sin(roll), 0],
                    [0, 1, 0, 0],
                    [-np.sin(roll), 0, np.cos(roll), 0],
                    [0, 0, 0, 1]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                    [np.sin(yaw), np.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    R = np.dot(np.dot(R_x, R_y), R_z)

    return R


def image_registration_e_2023():
    folder_in = 'data/ImageRegistration/'
    fixedImage = sitk.ReadImage(folder_in + 'ImgT1_v1.nii.gz')
    movingImage = sitk.ReadImage(folder_in + 'ImgT1_v2.nii.gz')

    imshow_orthogonal_view(fixedImage, title='Fixed image')
    plt.show()
    imshow_orthogonal_view(movingImage, title='Moving image')
    plt.show()

    # Define the roll rotation in radians
    angle = -20  # degrees
    roll_radians = np.deg2rad(angle)

    # Create the Affine transform and set the rotation
    transform = sitk.AffineTransform(3)
    rot_matrix = rotation_matrix(0, roll_radians, 0)[:3, :3] # SimpleITK inputs the rotation and the translation separately
    transform.SetMatrix(rot_matrix.T.flatten())

    # centre_image = np.array(movingImage.GetSize()) / 2 - 0.5  # Image Coordinate System
    # centre_world = movingImage.TransformContinuousIndexToPhysicalPoint(centre_image) # World Coordinate System

    # Apply the transformation to the image
    movingImage_reg = sitk.Resample(movingImage, transform)
    imshow_orthogonal_view(movingImage_reg, title='Moving image')

    mask = sitk.GetArrayFromImage(fixedImage) > 50
    fixedImageNumpy = sitk.GetArrayFromImage(fixedImage)
    movingImageNumpy = sitk.GetArrayFromImage(movingImage_reg)

    fixedImageVoxels = fixedImageNumpy[mask]
    movingImageVoxels = movingImageNumpy[mask]
    mse = np.mean((fixedImageVoxels - movingImageVoxels)**2)
    print('Anwer: MSE = {:.2f}'.format(mse))


def automatic_image_registration_e_2023():
    folder_in = 'data/ImageRegistration/'
    fixedImage = sitk.ReadImage(folder_in + 'ImgT1_v1.nii.gz')
    movingImage = sitk.ReadImage(folder_in + 'ImgT1_v2.nii.gz')

    # Set the registration - Fig. 1 from the Theory Note
    R = sitk.ImageRegistrationMethod()

    # Set a one-level the pyramid scheule. [Pyramid step]
    R.SetShrinkFactorsPerLevel(shrinkFactors=[2])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set the interpolator [Interpolation step]
    R.SetInterpolator(sitk.sitkLinear)

    # Set the similarity metric [Metric step]
    R.SetMetricAsMeanSquares()

    # Set the sampling strategy [Sampling step]
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10)

    # Set the optimizer [Optimization step]
    R.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=25)

    # Initialize the transformation type to rigid
    initTransform = sitk.CenteredTransformInitializer(fixedImage, movingImage, sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(initTransform, inPlace=False)

    # Some extra functions to keep track to the optimization process
    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R)) # Print the iteration number and metric value

    # Estimate the registration transformation [metric, optimizer, transform]
    tform_reg = R.Execute(fixedImage, movingImage)

    # Apply the estimated transformation to the moving image
    ImgT1_B = sitk.Resample(movingImage, tform_reg)
    imshow_orthogonal_view(ImgT1_B, title='Moving image')
    overlay_slices(fixedImage, ImgT1_B, title='Overlay')

    params = tform_reg.GetParameters()
    angles = params[:3]
    trans = params[3:6]
    print('Estimated translation: ')
    print(np.round(trans, 3))
    print('Estimated rotation (deg): ')
    print(np.round(np.rad2deg(angles), 3))


def image_registration_matrix_e_2023():
    folder_in = 'data/ImageRegistration/'
    # fixedImage = sitk.ReadImage(folder_in + 'ImgT1_v1.nii.gz')
    movingImage = sitk.ReadImage(folder_in + 'ImgT1_v2.nii.gz')

    tform = sitk.AffineTransform(3)
    centre_image = np.array(movingImage.GetSize()) / 2 - 0.5  # Image Coordinate System
    centre_world = movingImage.TransformContinuousIndexToPhysicalPoint(centre_image) # World Coordinate System
    tform.SetCenter(centre_world)

    A = np.array([[0.98, -0.16, 0.17], [0.26, 0.97, 0], [-0.17, 0.04, 0.98]])

    tform.SetMatrix(A.flatten())

    t = [0, -15, 0]
    tform.SetTranslation(t)

    newImage = sitk.Resample(movingImage, tform)
    imshow_orthogonal_view(newImage, title='Moving image')
    plt.show()


def aquarium_pca_e_2023():
    in_dir = "data/Fish/"
    all_images = ["discus.jpg", "guppy.jpg", "kribensis.jpg", "neon.jpg", "oscar.jpg",
                  "platy.jpg", "rummy.jpg", "scalare.jpg", "tiger.jpg", "zebra.jpg"]
    n_samples = len(all_images)

    # Read first image to get image dimensions
    im_org = io.imread(in_dir + all_images[0])
    im_shape = im_org.shape
    height = im_shape[0]
    width = im_shape[1]
    channels = im_shape[2]
    n_features = height * width * channels

    print(f"Found {n_samples} image files. Height {height} Width {width} Channels {channels} n_features {n_features}")

    data_matrix = np.zeros((n_samples, n_features))

    idx = 0
    for image_file in all_images:
        img = io.imread(in_dir + image_file)
        flat_img = img.flatten()
        data_matrix[idx, :] = flat_img
        idx += 1

    average_fish = np.mean(data_matrix, 0)
    io.imshow(create_u_byte_image_from_vector(average_fish, height, width, channels))
    plt.title('The Average Fish')
    io.show()

    im_favourite = io.imread("data/Fish/neon.jpg")
    im_favourite_flat = im_favourite.flatten()
    im_favourite_2 = io.imread("data/Fish/guppy.jpg")
    im_favourite_2_flat = im_favourite_2.flatten()

    sub_favourite = im_favourite_flat - im_favourite_2_flat
    ssd_dist = np.sum(sub_favourite**2)

    print(f"Answer: SSD distance from Neon to Guppy fish {ssd_dist}")

    print("Computing PCA")
    fishs_pca = PCA(n_components=6)
    fishs_pca.fit(data_matrix)

    plt.plot(fishs_pca.explained_variance_ratio_ * 100)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.show()

    print(f"Answer: Total variation explained by first two component {fishs_pca.explained_variance_ratio_[0] * 100 + fishs_pca.explained_variance_ratio_[1] * 100}")

    components = fishs_pca.transform(data_matrix)

    pc_1 = components[:, 0]
    pc_2 = components[:, 1]

    plt.plot(pc_1, pc_2, '.')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    extreme_pc_1_fish_m = np.argmin(pc_1)
    extreme_pc_1_fish_p = np.argmax(pc_1)
    extreme_pc_2_fish_m = np.argmin(pc_2)
    extreme_pc_2_fish_p = np.argmax(pc_2)

    print(f'PC 1 extreme minus fish: {all_images[extreme_pc_1_fish_m]}')
    print(f'PC 1 extreme minus fish: {all_images[extreme_pc_1_fish_p]}')
    print(f'PC 2 extreme minus fish: {all_images[extreme_pc_2_fish_m]}')
    print(f'PC 2 extreme minus fish: {all_images[extreme_pc_2_fish_p]}')

    fig, ax = plt.subplots(ncols=4, figsize=(16, 6))
    ax[0].imshow(create_u_byte_image_from_vector(data_matrix[extreme_pc_1_fish_m, :], height, width, channels))
    ax[0].set_title(f'PC 1 extreme minus fish')
    ax[1].imshow(create_u_byte_image_from_vector(data_matrix[extreme_pc_1_fish_p, :], height, width, channels))
    ax[1].set_title(f'PC 1 extreme plus fish')
    ax[2].imshow(create_u_byte_image_from_vector(data_matrix[extreme_pc_2_fish_m, :], height, width, channels))
    ax[2].set_title(f'PC 2 extreme minus fish')
    ax[3].imshow(create_u_byte_image_from_vector(data_matrix[extreme_pc_2_fish_p, :], height, width, channels))
    ax[3].set_title(f'PC 2 extreme plus fish')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()

    plt.plot(pc_1, pc_2, '.', label="All fishs")
    plt.plot(pc_1[extreme_pc_1_fish_m], pc_2[extreme_pc_1_fish_m], "*", color="green", label="Extreme fish 1 minus")
    plt.plot(pc_1[extreme_pc_1_fish_p], pc_2[extreme_pc_1_fish_p], "*", color="green", label="Extreme fish 1 plus")
    plt.plot(pc_1[extreme_pc_2_fish_m], pc_2[extreme_pc_2_fish_m], "*", color="green", label="Extreme fish 2 minus")
    plt.plot(pc_1[extreme_pc_2_fish_p], pc_2[extreme_pc_2_fish_p], "*", color="green", label="Extreme fish 2 plus")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("fishs in PCA space")
    plt.legend()
    plt.show()

    im_miss = io.imread("data/Fish/neon.jpg")
    im_miss_flat = im_miss.flatten()
    im_miss_flat = im_miss_flat.reshape(1, -1)
    pca_coords = fishs_pca.transform(im_miss_flat)
    pca_coords = pca_coords.flatten()

    plt.plot(pc_1, pc_2, '.', label="All fishs")
    plt.plot(pca_coords[0], pca_coords[1], "*", color="red", label="Missing fish")
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Fish in PCA space")
    plt.legend()
    plt.show()

    comp_sub = components - pca_coords
    pca_distances = np.linalg.norm(comp_sub, axis=1)

    best_match = np.argmin(pca_distances)
    best_twin_fish = data_matrix[best_match, :]

    worst_match = np.argmax(pca_distances)
    print(f"Answer: Worst matching PCA fish {all_images[worst_match]}")
    worst_twin_fish = data_matrix[worst_match, :]
    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
    ax[0].imshow(im_miss)
    ax[0].set_title('The Real Missing fish')
    ax[1].imshow(create_u_byte_image_from_vector(best_twin_fish, height, width, channels))
    ax[1].set_title('The Best Matching Twin fish')
    ax[2].imshow(create_u_byte_image_from_vector(worst_twin_fish, height, width, channels))
    ax[2].set_title('Answer: The Worst Matching Twin fish')
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()


def heart_pixel_analysis_e_2023():
    in_dir = "data/HeartCT/"
    im_name = "1-001.dcm"

    ct = dicom.read_file(in_dir + im_name)
    img = ct.pixel_array

    myo_roi = io.imread(in_dir + 'MyocardiumROI.png')
    myo_mask = myo_roi > 0
    myo_values = img[myo_mask]
    (mu_myo, std_myo) = norm.fit(myo_values)
    print(f"Myo: Average {mu_myo:.0f} standard deviation {std_myo:.0f}")

    blood_roi = io.imread(in_dir + 'bloodRoi.png')
    blood_mask = blood_roi > 0
    blood_values = img[blood_mask]
    (mu_blood, std_blood) = norm.fit(blood_values)
    print(f"Blood: Average {mu_blood:.0f} standard deviation {std_blood:.0f}")

    blood_gt = io.imread(in_dir + 'bloodGT.png')

    threshold = (mu_myo + mu_blood) / 2
    print(f"Answer: Threshold {threshold:.0f}")

    min_hu = mu_blood - 3 * std_blood
    max_hu = mu_blood + 3 * std_blood
    print(f"Answer: HU limits : {min_hu:0.2f} {max_hu:0.2f}")

    bin_img = (img > min_hu) & (img < max_hu)
    blood_label_colour = color.label2rgb(bin_img)
    io.imshow(blood_label_colour)
    plt.title("First blood estimate")
    io.show()

    footprint = disk(3)
    closing = binary_closing(bin_img, footprint)
    io.imshow(closing)
    plt.title("Second blood estimate")
    io.show()

    footprint = disk(5)
    opening = binary_opening(closing, footprint)
    io.imshow(opening)
    plt.title("Third blood estimate")
    io.show()

    label_img = measure.label(opening)
    n_labels = label_img.max()
    print(f"Answer: Number of labels: {n_labels}")

    region_props = measure.regionprops(label_img)

    min_area = 2000
    max_area = 5000

    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        a = region.area
        # p = region.perimeter

        if a < min_area or a > max_area:
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0

    # Create binary image from the filtered label image
    i_blood = label_img_filter > 0
    io.imshow(i_blood)
    io.show()

    gt_bin = blood_gt > 0
    dice_score = 1 - distance.dice(i_blood.ravel(), gt_bin.ravel())
    print(f"Answer: DICE score {dice_score:.3f}")


def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out


if __name__ == '__main__':
    pca_on_pistachio_e_2023()
    change_detection_e_2023()
    system_frame_rate_e_2023()
    landmark_based_registration_e_2023()
    pixel_value_mappings_and_filtering_e_2023()

    image_registration_e_2023()
    automatic_image_registration_e_2023()
    image_registration_matrix_e_2023()
    aquarium_pca_e_2023()
    heart_pixel_analysis_e_2023()


