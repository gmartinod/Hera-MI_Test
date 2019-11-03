import os
import pydicom
import numpy as np
from sklearn.feature_extraction import image
import cv2
from PIL import Image


def find_dicom (root):

    # Define empty list
    lstFilesDCM = []

    # get pathes to every dicom files in sub-directories
    for dirName, subdirList, fileList in os.walk(root):
        for filename in fileList:
            if ".dcm" in filename.lower():
                lstFilesDCM.append(os.path.join(dirName, filename))

    # Separate images to segmentation masks (one out of two)
    lstImDCM = lstFilesDCM[::2]
    lstMaskDCM = lstFilesDCM[1::2]

    return lstImDCM, lstMaskDCM

def read_adjust_dicom(ImPath, Mask_Path, Resize=(1100,600)):

    # Define list of Image+Mask dataset
    Dataset = list()

    # loop through all the DICOM images (0 to 3)
    for id in range(len(ImPath)):

        # temporary array of the Image+Mask
        tmp = np.zeros((Resize[0], Resize[1], 2), dtype=np.float32)

        # read the file
        Im = pydicom.read_file(ImPath[id])
        Mask = pydicom.read_file(Mask_Path[id])

        # Cropping the image to get same image size than mask size (difference on last image)
        (Rm, Cm) = Mask.pixel_array.shape
        (Ri, Ci) = Im.pixel_array.shape
        Half_diff_R = int((Ri - Rm) / 2)
        Half_diff_C = int((Ci - Cm) / 2)

        I = Im.pixel_array[Half_diff_R:Ri - Half_diff_R, Half_diff_C:Ci - Half_diff_C]
        M = Mask.pixel_array

        # Normalize Image between q2 and q99
        flat = np.ndarray.flatten(I)
        q2 = np.percentile(flat, 2)
        q99 = np.percentile(flat, 99)
        if np.min(I) < 0.0:
            I = I - q2 * 1.0
        I = I / q99 * 1.0

        I[I > 1] = 1
        I[I < 0] = 0

        # Resize Image and Mask
        I_resized = cv2.resize(I, (Resize[1], Resize[0]), cv2.INTER_AREA)
        M_resized = cv2.resize(M, (Resize[1], Resize[0]), cv2.INTER_AREA)

        # Combine Image+Mask in same array
        tmp[:, :, 0] = I_resized
        tmp[:, :, 1] = M_resized

        # store it in Dataset
        Dataset.append(tmp)

    return Dataset

def create_patches (Dataset, Size=(64,64), Max_patches=1000, Threshold=200):

    # Intern variables for statistics through all images
    nb_patches_tot = 0
    nb_masses_tot = 0
    patches_tot = []
    annotation_tot = []

    for K, dcm in enumerate(Dataset):

        # Build Patches
        array = np.array(dcm)
        patches = image.extract_patches_2d(array, Size, max_patches=Max_patches)
        print(f"Shapes of patches for image {K} : {patches.shape}")

        # Selection of non empty patches
        patches_NZ = patches[np.sum(patches, axis=(1, 2, 3)) > Threshold, :, :, :]
        print(f"Shapes of non-zero patches for image {K}: {patches_NZ.shape}")
        nb_patches_tot = nb_patches_tot + len(patches_NZ)

        # Build array of annotations
        isPatchaMass = np.zeros(len(patches_NZ))

        for i, patch in enumerate(patches_NZ):

            # Threshold to consider the mask as mass (5% of total pixels)
            if sum(sum(patch[:, :, 1])) > 0.05 * (Size[0]*Size[1]):
                isPatchaMass[i] = 1

        print(f"Total number of masses on image {K} : {sum(isPatchaMass)} \n")
        nb_masses_tot = nb_masses_tot + sum(isPatchaMass)

        patches_tot.append(patches_NZ)
        annotation_tot.append(isPatchaMass)

        # Delete the variable "patches" to save memory
        del patches

    proportion_of_masses = nb_masses_tot/nb_patches_tot
    print(f"Proportion of masses through the patches : {proportion_of_masses}\n")

    return patches_tot, annotation_tot, proportion_of_masses

def two_classes_split(dataset, annotation):

    # Gobal list of the 4 cases
    differenciated_dataset = []

    for i, image in enumerate(dataset):

        # Temporary list of each case
        current_image = []
        mass_patches = []
        non_mass_patches = []

        for p, patch in enumerate(image):

            # Diffenciation of mass patches and non mass patches from annotation array
            if annotation[i][p] == 1.:
                mass_patches.append(np.array(patch))
            else:
                non_mass_patches.append(np.array(patch))

        # Storage of the lists
        current_image.append(np.array(non_mass_patches))
        current_image.append(np.array(mass_patches))

        differenciated_dataset.append(current_image)

    return differenciated_dataset
