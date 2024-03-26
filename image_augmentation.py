import os
import pathlib
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import albumentations as A
from albumentations.core.composition import Compose


zse_sbir_dir = "./ZSE-SBIR"
datasets = {"QuickDraw": {}, "Sketchy": {}, "TUBerlin": {}}

zs_dataset_dir = os.path.join(zse_sbir_dir, "datasets")
da_dir = "./data-augmentation"


# Create folders for the augmented data
def makeAugPath(augment_type, augment_dir):
    if not os.path.exists(da_dir):
        os.makedirs(da_dir)
        augment_dir = pathlib.Path(str(os.path.join(augment_dir, augment_type)))
        if not os.path.exists(augment_dir):
            os.makedirs(augment_dir)
            for ds in datasets.keys():
                os.makedirs(pathlib.Path(os.path.join(augment_dir, ds)))
    augment_dir = pathlib.Path(os.path.join(da_dir, augment_type))
    return augment_dir


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
                            img_full_path = os.path.join("./dataset", ds, img_path_in_class)
                            # Dictionary {dataset: {class: image}...}
                            if class_name not in datasets[ds]:
                                datasets[ds][class_name] = {img_full_path}
                            datasets[ds][class_name].add(img_full_path)


# def addGaussianBlur():
#     augment_dir = da_dir
#     augment_dir = makeAugPath("gaussian-noise", augment_dir)
#     getTestData()
#     for ds_name, ds_set in datasets.items():
#         for class_name in ds_set.keys():
#             for img_full_path in ds_set[class_name]:
#                 # Show original image
#                 # img = mpimg.imread(img_full_path)
#                 # imgplot = plt.imshow(img)
#                 # plt.show()

#                 augmented_path = pathlib.Path(os.path.join(augment_dir, ds_name))
#                 original = cv2.imread(img_full_path, cv2.IMREAD_UNCHANGED)
#                 # Apply Gaussian blur
#                 augmented = cv2.GaussianBlur(original, (5, 5), cv2.BORDER_DEFAULT)

#                 # Insert to data augmentation folder
#                 augmented_path = os.path.join(augmented_path, class_name)
#                 if not os.path.exists(augmented_path):
#                     os.makedirs(augmented_path)
#                 augmented_path = os.path.join(augmented_path, img_full_path.split("/")[-1])
#                 cv2.imwrite(augmented_path, augmented)

#                 # Display augmented image
#                 # img = mpimg.imread(augmented_path)
#                 # imgplot = plt.imshow(img)
#                 # plt.show()

def augmentData(augment_type):
    augment_dir = da_dir
    augment_dir = makeAugPath(augment_type, augment_dir)
    getTestData()
    for ds_name, ds_set in datasets.items():
        for class_name in ds_set.keys():
            for img_full_path in ds_set[class_name]:
                # Show original image
                # img = mpimg.imread(img_full_path)
                # imgplot = plt.imshow(img)
                # plt.show()

                augmented_path = pathlib.Path(os.path.join(augment_dir, ds_name))
                original = cv2.imread(img_full_path, cv2.IMREAD_UNCHANGED)
                
                # Apply augmentation
                if augment_type == "gaussian-noise":
                    augmented = cv2.GaussianBlur(original, (5, 5), cv2.BORDER_DEFAULT)
                elif augment_type == "rotation":
                    transform = Compose([A.Rotate(limit=90, p=1)])
                    augmented = transform(image=original)['image']
                elif augment_type == "translation":
                    transform = Compose([A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0, rotate_limit=0, p=1)])
                    augmented = transform(image=original)['image']

                # Insert to data augmentation folder
                augmented_path = os.path.join(augmented_path, class_name)
                if not os.path.exists(augmented_path):
                    os.makedirs(augmented_path)
                augmented_path = os.path.join(augmented_path, img_full_path.split("/")[-1])
                cv2.imwrite(augmented_path, augmented)

                # Display augmented image
                # img = mpimg.imread(augmented_path)
                # imgplot = plt.imshow(img)
                # plt.show()
                                
                

if __name__ == '__main__':
    augmentData(augment_type = "rotation")

