#!/usr/bin/env python3

import argparse

import torch
import torchvision

from mydata import labelstrs
import resnet

parser = argparse.ArgumentParser()
parser.add_argument("model", help="File path of the model")
parser.add_argument("images", nargs="*", help="File path of image to be classified")
args = parser.parse_args()

device = "cpu"

try: ROOT_DIR
except NameError: ROOT_DIR = ""

images_iter = (torch.unsqueeze(torchvision.io.read_image(fn, torchvision.io.ImageReadMode.RGB).to(torch.float32).detach(), 0) for fn in args.images)
images = torch.cat(list(images_iter))

model = torch.load(args.model)

outputs = model(images)
labels = torch.argmax(outputs, dim=1)
label_names = list(labelstrs[label] for label in labels)

print(label_names)
