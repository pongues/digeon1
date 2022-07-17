import os

import torch
import torchvision
from torch.utils.data import Dataset

labelstrs = (
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
    "american_bulldog",
    "american_pit_bull_terrier",
    "basset_hound",
    "beagle",
    "boxer",
    "chihuahua",
    "english_cocker_spaniel",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "miniature_pinscher",
    "newfoundland",
    "pomeranian",
    "pug",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "staffordshire_bull_terrier",
    "wheaten_terrier",
    "yorkshire_terrier",
)


class CustomImageDataset(Dataset):
    img_dir: str
    size: str

    def __init__(self, list_fn: str, img_dir: str):
        self.img_dir = img_dir
        with open(list_fn, "r") as f:
            self.images = list(
                line.split("\t") for line in f.read().split("\n"))
        self.size = len(self.images) - 1

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        label = labelstrs.index(self.images[index][0])
        img_path = os.path.join(self.img_dir, self.images[index][1])
        image = torchvision.io.read_image(
            img_path, torchvision.io.ImageReadMode.RGB
        ).to(torch.float32)
        return image, label
