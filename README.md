# HERA-MI, Machine Learning Ingeneer
## Pre-Interview entrance test
## Guillaume Martinod, 03/11/2019

### 1. Dataset preparation

From the repository downloaded on GitHub, I searched for every Dicom Data pathes and splited it into a list of images and a list of the corresponding masks (data.find_dicom())

Then, for each of the 4 cases (image and mask), I loaded the Dicoms as numpy arraies and pre-process it (data.read_adjust_dicom()). In 3 cases, the image and the mask have the same size, but the last one has a smaller mask. Without more info about the relative localization of both, I cropped each side of the image to fit the mask dimensions.
I added a step of normalization between percentil 2 and percentil 99 to bring the range of pixel values between 0 and 1 (with precision float32). To handle the data easier I resized every image and mask to size (2200, 1200) corresponding to 40% of the mean size (5500, 3000).
I combined each image with its mask in a 3D array, stored in a list. 

From each case resized and normalized, I extracted 1000 (128, 128) patches with extract_patches_2d() from sklearn library (data.create_patches()). To avoid learning on patches full of zeros, I removed the patches for which the sum of the pixel values is lower than a threshold.
Each non empty patch is considered as "Mass" (classe 1) if its corresponding mask has 5% of positive pixels. Otherwise the patch is considered "Non mass" (classe 0). This reference annotation is stored in a binary array.
The patches and annotations are stored in 2 lists, and the percentage of masses in the dataset is calculated (average of 7% of mass patches)

The next step is to differenciate the mass and non mass patches in two different lists (data.two_classes_split()). We get a list of the 4 cases, each composed of a list of the mass patches and a list of the non mass patches. 


