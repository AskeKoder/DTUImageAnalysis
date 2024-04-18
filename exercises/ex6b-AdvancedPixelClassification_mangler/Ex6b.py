#%% Load packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
os.chdir(r'C:\Users\askeb\OneDrive - Danmarks Tekniske Universitet\DTU\9. Semester\Image analysis\Scripts og data\DTUImageAnalysis\exercises\ex6b-AdvancedPixelClassification_mangler')

#%% Get data
in_dir = 'data/'
in_file = 'ex6_ImagData2Load.mat'
data = sio.loadmat(in_dir + in_file)
ImgT1 = data['ImgT1']
ImgT2 = data['ImgT2']
ROI_GM = data['ROI_GM'].astype(bool)
ROI_WM = data['ROI_WM'].astype(bool)


# %% Display T1 and T2 images
plt.figure()
plt.subplot(3, 2, 1)
plt.imshow(ImgT1,cmap='grey')
plt.title('T1 image')
plt.axis('off')
plt.subplot(3, 2, 2)
plt.imshow(ImgT2,cmap='grey')
plt.title('T2 image')
plt.axis('off')
plt.subplot(3, 2, 3)
plt.hist(ImgT1[ImgT2>20].ravel(), bins=255)
plt.xlabel('Intensity')
plt.subplot(3, 2, 4)
plt.hist(ImgT2[ImgT2>20].ravel(), bins=255)
plt.xlabel('Intensity')
plt.show()

#2d histogram
plt.figure()
plt.hist2d(ImgT1[ImgT2>20].ravel(), ImgT2[ImgT2>20].ravel(), bins=255)
plt.show()

#Scatter plot
plt.figure()
plt.plot(ImgT1[ImgT2>20], ImgT2[ImgT2>20],'.', color="blue", label='GM')

# %% Show the exper drawings
C1 = ROI_WM
C2 = ROI_GM

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(C1, cmap='gray')
plt.title('WM')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(C2, cmap='gray')
plt.title('GM')
plt.axis('off')
plt.show()


# %%
