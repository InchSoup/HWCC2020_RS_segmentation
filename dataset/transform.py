import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    # reszie
    A.OneOf([
        A.Resize(512, 512),
        A.RandomResizedCrop(512, 512)
    ]),

    # 旋转
    A.OneOf([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5)
    ]),

    #A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    #A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.5),
    A.RandomBrightnessContrast(),
    A.GaussNoise(p=0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])