import json
from pathlib import Path

import cv2
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from tqdm import tqdm


def create_label(x):
    return str(Path(Path(x).name.split("_")[-1]).stem)


# - since all data is in images..
def conv_to_frames(fname, fpath, label):

    vidcap = cv2.VideoCapture(str(fname))
    #     print(fname)
    success, image = vidcap.read()
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
        #         print(f"{fname.parent}/frame:{count}_{label}.jpg")
        cv2.imwrite(
            f"{fname.parent}/frame:{count}_{label}.jpg", image
        )  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1


class ImageClassDs(Dataset):
    def __init__(
        self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms=None
    ):
        self.df = df
        self.imfolder = imfolder
        self.train = train
        self.transforms = transforms

    def __getitem__(self, index):
        im_path = self.df.iloc[index]["image_id"]
        x = cv2.imread(im_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        if self.transforms:
            x = self.transforms(image=x)["image"]

        y = self.df.iloc[index]["label"]
        return {
            "x": x,
            "y": y,
        }

    def __len__(self):
        return len(self.df)


def preprocess_data(fpath, max_l=1000):
    with open(fpath / "v1_full_trailer.json", "r") as f:
        label_json = json.load(f)
    key_lis = list(label_json.keys())

    label_dic = {}
    for key in key_lis:
        inner = label_json[key]
        in_keys = list(inner.keys())
        out_k = [
            (fpath / f"trailer/{key}/shot_{k}.mp4", inner[k]["scale"]["label"])
            for k in in_keys
        ]
        for x in out_k:
            label_dic[x[0]] = x[1]

    key_df = pd.DataFrame(label_dic, index=[1]).T.reset_index()
    key_df.columns = ["fpath", "label"]
    key_df.to_csv(fpath / "labels.csv")
    list_to_run = [(x[1], x[2]) for x in key_df.itertuples()]

    for i in tqdm(list_to_run[:max_l], total=max_l):
        run_splitter(i)


def create_from_dict(all_ims, create_label):

    df = pd.DataFrame.from_dict(
        {x: create_label(x) for x in all_ims}, orient="index"
    ).reset_index()

    df.columns = ["image_id", "label"]

    print(df.head())

    temp = preprocessing.LabelEncoder()
    df["label"] = temp.fit_transform(df.label.values)

    label_map = {i: l for i, l in enumerate(temp.classes_)}

    df.label.nunique()

    df.label.value_counts()

    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    stratify = StratifiedKFold(n_splits=5)
    for i, (t_idx, v_idx) in enumerate(
        stratify.split(X=df.image_id.values, y=df.label.values)
    ):
        df.loc[v_idx, "kfold"] = i
        df.to_csv("train_folds.csv", index=False)

    return df, label_map
