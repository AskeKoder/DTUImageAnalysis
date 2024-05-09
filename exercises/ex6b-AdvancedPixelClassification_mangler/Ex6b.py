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
org_shape = ImgT1.shape
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
plt.plot(ImgT1[ImgT2>20], ImgT2[ImgT2>20],'.', color="blue", label='GM',ms=1)

#**Q1**: What is the intensity threshold that can 
#separate the GM and WM classes (roughly) from the 1D histograms? 

#It seems:
# In the T1 WM an GM can be separated around the intensity 500
# In the T2 WM and GM can be separated around the intensity 180


#**Q2**: Can the GM and WM intensity classes be observed in the
# 2D histogram and scatter plot?
#In the histogram it looks like there is are two structures that can be
#separated, also in the scatter plot it looks like there are two clusters

# %% Show the expert drawings
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

#**Q3**: Does the ROI drawings look like what you expect from an expert?
# Yes they are binary images, and only contain a subset of what I suspect
#is GM and WM


# %% Get indices
qC1 = np.argwhere(C1)
qC2 = np.argwhere(C2)

#Get training examples
WMT1 = ImgT1[qC1[:,0],qC1[:,1]]
GMT1 = ImgT1[qC2[:,0],qC2[:,1]]
WMT2 = ImgT2[qC1[:,0],qC1[:,1]]
GMT2 = ImgT1[qC2[:,0],qC2[:,1]]

#Plot histogram
fig ,axs = plt.subplots(2,2)
axs[0,0].hist(WMT1.ravel(),bins=255)
axs[0,0].set_title('T1')
axs[0,0].set_ylabel('WM')
axs[0,0].set_xlim(200,900)

axs[1,0].hist(GMT1.ravel(),bins=255)
axs[1,0].set_ylabel('GM')
axs[1,0].set_xlim(200,900)

axs[0,1].hist(WMT2.ravel(),bins=255)
axs[0,1].set_title('T2')
axs[0,1].set_xlim(100,600)
axs[1,1].hist(GMT2.ravel(),bins=255)
axs[1,1].set_xlim(100,600)
plt.show
# They actually confirm the approximate boundaries we suggested before
# with some adjustment it should be good.
# It also looks like T2 is most useful for separating the two
# %% Set up training and target vector

#Training vector
#First include the full two images as columns
X_temp = np.hstack([ImgT1.reshape(-1,1),ImgT2.reshape(-1,1)])

#Make list of indices, first C1 the C2
idxC1 = np.argwhere(C1.reshape(-1,1).squeeze()).squeeze()
idxC2 = np.argwhere(C2.reshape(-1,1).squeeze()).squeeze()
idx = np.hstack([idxC1,idxC2])

#Get the right points to finish X
X = X_temp[idx,:]

#Target vector
T = np.hstack([np.zeros(len(idxC1)),np.ones(len(idxC2))])

# %% Exercise 5 Scatter plot
plt.scatter(X[T==0,0],X[T==0,1], label = "Class 1", color="green", s=1)
plt.scatter(X[T==1,0],X[T==1,1], label = "Class 2", color="black", s=1)
plt.legend()
plt.xlabel('T1')
plt.ylabel('T2')

# It looks very goood for the training samples!!
# %% Ex 6 Train LDA using the function given in the data folder
from LDA import LDA
W = LDA(X,T)

# %% Ex 7 apply the linear discriminatn function
Xall= np.c_[ImgT1.ravel(), ImgT2.ravel()]
Y = np.c_[np.ones((len(Xall), 1)), Xall] @ W.T

#So Y is the log(P(Ci|x))
# %% Ex 8
#Calculate posterior probability
PosteriorProb = np.clip(np.exp(Y) / np.sum(np.exp(Y),1)[:, np.newaxis], 0, 1)

#clip funciton limits the function within [0,1] exceedence is set to limit values
# %% Ex 9 Apply segmentation
SegT1C1 =  ImgT1[np.where(PosteriorProb[:,0].reshape(org_shape)>0.5)]
SegT2C1 =  ImgT2[np.where(PosteriorProb[:,0].reshape(org_shape)>0.5)]
SegT1C2 =  ImgT1[np.where(PosteriorProb[:,1].reshape(org_shape)>0.5)]
SegT2C2 =  ImgT2[np.where(PosteriorProb[:,1].reshape(org_shape)>0.5)]

#%%Scatter plot showing segmentation results
plt.scatter(SegT1C1,SegT2C1, label = "Class 1", color="green", s=1)
plt.scatter(SegT1C2,SegT2C2, label = "Class 2", color="black", s=1)
plt.legend()
plt.xlabel('T1')
plt.ylabel('T2')

#The hyperplane is clearly identifiable
#I would assume a nonlinear hyperplane woul perform better to also separate
#the background

#The segmentation is better when we use two images because otherwise we
#would not be able to get the sloped descision boundary

#Yes, but since we have only two options, the background and brain water
# are also classified as one of the classe, maybe one should work more on filtering
#before LDA

# I guess yes you need to be an expert but 
# %%
Overlay_img_WM = ImgT1.copy()
Overlay_img_GM = ImgT1.copy()

Overlay_img_WM[np.where(PosteriorProb[:,0].reshape(org_shape)>0.5)] = 1000
Overlay_img_WM[np.where(PosteriorProb[:,0].reshape(org_shape)<0.5)] = 0
Overlay_img_GM[np.where(PosteriorProb[:,1].reshape(org_shape)>0.5)] = 500
Overlay_img_GM[np.where(PosteriorProb[:,1].reshape(org_shape)<0.5)] = 0
plt.imshow(ImgT1)
plt.imshow(Overlay_img_WM + Overlay_img_GM+ImgT1, cmap="Reds")
plt.show()

#We should control more for the background as it is mostly segmented as class2
#right now
# %%
