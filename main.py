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
batch_size = 128
num_classes = 5
img_size = 128
n_epochs = 1
max_preprocessed = 1000

# Define network
enet = EfficientNet.from_pretrained(
            "efficientnet-b3", num_classes=num_classes
        )
in_features = enet._fc.in_features
enet._fc = nn.Linear(in_features, num_classes)

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

dm.train_dataloader()

# PASS MODEL
model = LitModel(num_classes,model = enet, learning_rate=1e-4)

count_parameters(model.model, show_table=False)


def visualize_model(model, inp_size=[1, 3, 64, 64], device="cuda:0"):
    model = model.to(device)
    model.eval()
    """
    Use hiddenlayer to visualize a model
    """
    return hl.build_graph(model, torch.zeros(inp_size).to(device))



visualize_model(model)

# +
# freeze_to(model.model, 9)

# +
# unfreeze_to(model.model, 9)
# -

total_layer_state(model)

# RUN TRAINING
logger = CSVLogger("logs", name="eff-5")
trained = run_training(n_epochs, model, dm, logger=logger)

# !cat "logs/eff-5/version_18/metrics.csv"


