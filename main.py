import glob
import os
import albumentations as A
#  import torchsnooper as sn

from helpers import *
from model import *
from train import *
from efficientnet_pytorch import EfficientNet

# Helper module support
from lightningaddon import *

os.environ["TORCH_HOME"] = "/media/hdd/Datasets/"
# DEFINE EVERYTHING
lg = outLogger()

main_path = "/media/hdd/Datasets/shotclassification/trailer/"
batch_size = 512
num_classes = 5
img_size = 128
n_epochs = 5
max_preprocessed = 5000

# Define network
# enet = EfficientNet.from_pretrained(
#             "efficientnet-b3", num_classes=num_classes
#         )
# enet = EfficientNet.from_name("efficientnet-b3")
# in_features = enet._fc.in_features
# enet._fc = nn.Sequential(
#     nn.Linear(in_features, num_classes),
#     nn.Softmax(dim = 1))


enet = xresnet34(c_out= num_classes)
in_features = enet[10].in_features
enet[10] = nn.Sequential(
    nn.Linear(in_features, num_classes),
    nn.Softmax(dim = 1))


# PREPROCESS
lg("start preprocessing")
# preprocess_data(Path("/media/hdd/Datasets/shotclassification/"),max_preprocessed)
lg("end preprocessing")

# CHOOSE WHERE THE FILES ARE
all_ims = glob.glob(main_path + "/*/*.jpg")
lg(f"Len all : {len(all_ims)}")

# CONVERT TO DATAFRAME FOR STRATIFY
df, label_map = create_from_dict(all_ims, create_label=create_label)
lg("Created dataframe")

# subset
df= df.loc[:1000]

# +
# LOAD
lg("Loading data")
dm = ImDataModule(
    df,
    batch_size=batch_size,
    num_classes=num_classes,
    img_size=img_size,
    data_dir=main_path,
)

class_ids = dm.setup()
# -

# PASS MODEL
model = LitModel(num_classes,model = enet, learning_rate=1e-4)

count_parameters(model.model, show_table=False)

visualize_model(model)

# +
# freeze_to(model.model, 9)

# +
# unfreeze_to(model.model, 9)
# -

total_layer_state(model)

# RUN TRAINING
logger = CSVLogger("logs", name="xres34-no-aug")
trained = run_training(n_epochs, model, dm, logger=logger)

get_last_log("xres34-no-aug")


