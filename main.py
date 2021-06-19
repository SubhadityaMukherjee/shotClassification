import glob
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
#  import torchsnooper as sn

from helpers import *
from model import *
from train import *

# Helper module support
from lightningaddon import *

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
# DEFINE EVERYTHING
lg = outLogger()

main_path = "/media/hdd/Datasets/shotclassification/trailer/"
batch_size = 128
num_classes = 5
img_size = 128
n_epochs = 5
max_preprocessed = 1000

# Define network
enet = EfficientNet.from_pretrained(
            "efficientnet-b3", num_classes=num_classes
        )
in_features = enet._fc.in_features
enet._fc = nn.Linear(in_features, num_classes)

train_transforms=[
                A.RandomResizedCrop(img_size, img_size, p=1.0),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                A.CoarseDropout(p=0.5),
                A.Cutout(p=0.5),
                ToTensorV2(p=1.0),
            ]

valid_transforms = [
                A.CenterCrop(img_size, img_size, p=1.0),
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ],


# PREPROCESS
lg("start preprocessing")
#  preprocess_data(main_path,max_preprocessed)
lg("end preprocessing")

# CHOOSE WHERE THE FILES ARE
all_ims = glob.glob(main_path + "/*/*.jpg")
lg(f"Len all : f{len(all_ims)}")

# CONVERT TO DATAFRAME FOR STRATIFY
df, label_map = create_from_dict(all_ims, create_label=create_label)
lg("Created dataframe")

# LOAD
lg("Loading data")
dm = ImDataModule(
    df,
    batch_size=batch_size,
    num_classes=num_classes,
    img_size=img_size,
    data_dir=main_path,
    train_transforms = train_transforms,
    valid_transforms = valid_transforms

)
class_ids = dm.setup()

# PASS MODEL
model = LitModel(num_classes,model = enet, learning_rate=1e-4)
count_parameters(model.model)

# RUN TRAINING
logger = CSVLogger("logs", name="eff-5")
run_training(n_epochs, model, dm, logger=logger)
