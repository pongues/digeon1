#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
from mydata import CustomImageDataset
import resnet

from torch.utils.data import DataLoader

device = "cpu"

try: ROOT_DIR
except NameError: ROOT_DIR = ""

all_data = CustomImageDataset(ROOT_DIR + "list.jpg.txt", ROOT_DIR + "resized_images_224")
all_len = len(all_data)
train_len = all_len * 4 // 5
test_len = all_len - train_len
train_data, test_data = torch.utils.data.random_split(all_data, [train_len, test_len])

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

model = resnet.resnet34(64, 37).to(device)
print(model)


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
		
		print(f"{loss.item():.5f}")
	print(f"[{running_loss / count:.5f}]")


def test(dataloader, model, criterion):
	size = 0
	count = len(dataloader)
	model.eval()
	loss, correct = 0, 0
	with torch.no_grad():
		for i, (inputs, labels) in enumerate(dataloader):
			outputs = model(inputs)
			loss += criterion(outputs, labels).item()
			n_correct = (outputs.argmax(1) == labels).type(torch.float).sum().item()
			size += len(inputs)
			correct += n_correct
			print(n_correct)
	loss /= size
	correct /= size
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 500
for t in range(epochs):
	print(str(t))
	print(f"Epoch {t+1}\n-------------------------------")
	train(train_dataloader, model, criterion, optimizer)
	test(test_dataloader, model, criterion)
	torch.save(model, ROOT_DIR + str(t+1) + "model.pth")
print("Done!")
