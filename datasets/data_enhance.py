import albumentations as a
from PIL import Image
import numpy as np
from albumentations import transforms
from albumentations.core.composition import OneOf



# 定义数据增强函数
transform = a.Compose([
    transforms.HueSaturationValue(p=0.7),
    transforms.RandomBrightnessContrast(p=0.7),
    transforms.GaussNoise(var_limit=(50, 100), p=0.7),
    a.ISONoise(color_shift=(0.2, 0.9), intensity=(0.2, 0.9), always_apply=False, p=0.7),
    a.ToGray(p=0.5)
])

def augmented_img(img):
    img_np = np.array(img)
    augmented_image_np = transform(image=img_np)['image']
    augmented_img = Image.fromarray(augmented_image_np)
    # print("---已增强")
    return augmented_img