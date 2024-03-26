import albumentations as A
import cv2
from albumentations.core.composition import Compose
import os

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    return image

def save_image(image, image_path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
    cv2.imwrite(image_path, image)

def apply_advanced_transformations(root_dir):
    transform = Compose([
        A.Rotate(limit=90, p=1),
    ])
    
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg"):
                file_path = os.path.join(subdir, file)
                image = load_image(file_path)
                augmented = transform(image=image)['image']
                save_image(augmented, file_path.replace('.jpg', '_augmented.jpg'))
                
                
if __name__ == '__main__':
    apply_advanced_transformations('dataset/')
