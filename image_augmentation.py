import os
import pathlib
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import albumentations as A
from albumentations.core.composition import Compose
import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter
from PIL import Image

from medpy.filter.smoothing import anisotropic_diffusion

zse_sbir_dir = "./ZSE-SBIR"
datasets = {"QuickDraw": {}, "Sketchy": {}, "TUBerlin": {}}

zs_dataset_dir = os.path.join(zse_sbir_dir, "datasets")
da_dir = "/content/gdrive/MyDrive/cs413/dataset/"


# Create folders for the augmented data
def makeAugPath(augment_type):
    if not os.path.exists(da_dir):
        for ds in datasets.keys():
            augment_dir = pathlib.Path(str(os.path.join(da_dir, ds)))
            augment_dir = pathlib.Path(str(os.path.join(augment_dir, str(augment_type))))
            if not os.path.exists(augment_dir):
                os.makedirs(augment_dir)


# Fill datasets set & its dictionary
def getTestData():
    for ds in datasets.keys():
        ds_path = pathlib.Path(os.path.join(zs_dataset_dir, ds))
        for folder in ds_path.iterdir():
            for file in folder.iterdir():
                # Read from the list of test sketches paths
                if file.name.endswith("_zero.txt") and ("png_ready" in file.name or "sketch" in file.name):
                    with open(file, 'r') as f:
                        for line in f.readlines():
                            # Obtain the path of test sketches and add to datasets dictionary
                            img_path_in_class = re.search(r"(.*) ", line).group(1)
                            class_name = img_path_in_class.split("/")[-2]
                            img_full_path = os.path.join(da_dir, ds, img_path_in_class)
                            # Dictionary {dataset: {class: image}...}
                            if class_name not in datasets[ds]:
                                datasets[ds][class_name] = {img_full_path}
                            datasets[ds][class_name].add(img_full_path)


def augmentData(augment_type):
    makeAugPath(augment_type)
    getTestData()
    for ds_name, ds_set in datasets.items():
        for class_name in ds_set.keys():
            for img_full_path in ds_set[class_name]:
                # Show original image
                # img = mpimg.imread(img_full_path)
                # imgplot = plt.imshow(img)
                # plt.show()

                augmented_path = pathlib.Path(os.path.join(da_dir, ds_name))
                augmented_path = pathlib.Path(os.path.join(augmented_path, str(augment_type)))

                # Apply augmentation
                if augment_type == "gaussian-noise":
                    original = Image.open(img_full_path)
                    arr = np.array(original)
                    augmented = Image.fromarray(np.uint8(gaussian_filter(arr, sigma=2)))
                elif augment_type == "rotation":
                    original = cv2.imread(img_full_path)
                    transform = Compose([A.Rotate(limit=90, p=1)])
                    augmented = transform(image=original)['image']
                elif augment_type == "translation":
                    original = cv2.imread(img_full_path)
                    transform = Compose([A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0, rotate_limit=0, p=1)])
                    augmented = transform(image=original)['image']
                elif augment_type == "anisotropic-diffusion":
                    original = cv2.imread(img_full_path)
                    augmented = anisotropic_diffusion(original)
                elif augment_type == "sharpen":
                    original = cv2.imread(img_full_path)
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    augmented = cv2.filter2D(original, -1, 250 * kernel)

                    # Insert to data augmentation folder
                augmented_path = os.path.join(augmented_path, class_name)
                if not os.path.exists(augmented_path):
                    os.makedirs(augmented_path)
                augmented_path = os.path.join(augmented_path, img_full_path.split("/")[-1])
                if augment_type == "gaussian-noise":
                    augmented.save(augmented_path)
                else:
                    cv2.imwrite(augmented_path, augmented)

                # Display augmented image
                # img = mpimg.imread(augmented_path)
                # imgplot = plt.imshow(img)
                # plt.show()

if __name__ == '__main__':
    # augmentData(augment_type = "anisotropic-diffusion")
    augmentData(augment_type = "sharpen")
