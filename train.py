import tqdm
from preprocessing import exctract_json_data, define_categories
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets
from Dataloader.WSASL_Load import MyCustomDataset
from Model.CNN_Vanilla_frame_classification import Net
from torch.utils.data import random_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = MyCustomDataset('labels_100')
dataset_size = (len(dataset))

val_size = int(np.floor(dataset_size * 0.1))
train_size = int(dataset_size - val_size)
trainset, validset = random_split(dataset, [train_size, val_size])
dataloader_train = DataLoader(trainset, batch_size=20, shuffle=True, num_workers=2)
dataloader_val = DataLoader(validset, batch_size=20, shuffle=True, num_workers=2)

net = Net()
net = net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = criterion.to(device)

def accuracy(ys, ts):
    y = torch.argmax(ys, dim = 1)
    x = ts
    print("t:", x)
    print("y:", y)
    correct = 0
    for i in range(len(y)):
        if y[i] == x[i]:
            correct += 1
    return correct/len(y)



for epoch in range(2):  # loop over the dataset multiple times

    net.train()
    training_loss = 0.0
    for i,(inputs, labels) in enumerate(dataloader_train):
        # get the inputs; data is a list of [inputs, labels]
        inputs = inputs.view(-1, 1, 256, 256)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        
        outputs = net(inputs)
        value, index = (torch.max(outputs,0))
        value, index = (torch.max(labels,0))
        preds = torch.max(outputs, 1)[1]
        loss = criterion(outputs, torch.LongTensor(labels))
        loss.backward()
        training_loss += loss.item()
        optimizer.step()

        print(f"Training phase, Epoch: {epoch}. Loss: {training_loss/(i+1)}. Accuracy: {accuracy(outputs,labels)}.")

    net.eval()
    valError = 0
    for i, (inputs,labels) in enumerate(dataloader_val):
        with torch.no_grad():
            inputs = inputs.view(-1,1,256,256)
            inputs = inputs.to(device)
            labels = labels.to(device)
            prediction = net(inputs)
            loss = criterion(prediction, labels)

            valError += loss.item()
        print(f"Validation phase, Epoch: {epoch}. Loss: {valError/(i+1)}. Accuracy: {accuracy(outputs,labels)}.")

print('Finished Training')
