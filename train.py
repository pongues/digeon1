#!/usr/bin/env python3

import argparse

import torch
import torch.nn as nn

from mydata import CustomImageDataset
import resnet

from torch.utils.data import DataLoader


def train(dataloader, model, criterion, optimizer):
    running_loss = 0.0
    count = len(dataloader)
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"[{running_loss / count:.5f}]")


def test(dataloader, model, criterion):
    size = 0
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            nc = (outputs.argmax(1) == labels).type(torch.float).sum().item()
            size += len(inputs)
            correct += nc
    loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%,")
    print(f"Avg loss: {loss:>8f} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-im", help="initial model file")
    parser.add_argument("-om", required=True, help="output model model prefix")
    parser.add_argument("-bs", type=int, default=128, help="batch size")
    parser.add_argument("-tr", type=float, default=0.8,
                        help="parcentage of training data")
    parser.add_argument("-lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("-es", type=int, default=500, help="number of epoch")
    args = parser.parse_args()

    print(args)

    device = "cpu"

    try:
        ROOT_DIR
    except NameError:
        ROOT_DIR = ""

    all_data = CustomImageDataset(
        ROOT_DIR + "list.jpg.txt", ROOT_DIR + "resized_images_224"
    )
    all_len = len(all_data)
    train_len = int(all_len * args.tr)
    test_len = all_len - train_len
    train_data, test_data = torch.utils.data.random_split(
        all_data, [train_len, test_len]
    )

    train_dataloader = DataLoader(
        train_data, batch_size=args.bs, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(test_data, batch_size=args.bs)

    if args.im is None:
        model = resnet.resnet34(64, 37).to(device)
    else:
        model = torch.load(args.im).to(device)

    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for t in range(args.es):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, criterion, optimizer)
        test(test_dataloader, model, criterion)
        torch.save(model, args.om + str(t + 1) + ".pth")
    print("Done!")
