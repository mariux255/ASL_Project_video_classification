import tqdm
import numpy as np
import os
import cv2
import csv
import torch
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets
from Dataloader.WSASL_Videos_load import MyCustomDataset
from Model.CNN_Vanilla_frame_classification import Net
from Model.C3D_model import C3D
from Model.cnnlstm import ConvLSTM
from Model.pytorch_i3d import InceptionI3d
from torch.utils.data import random_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
buffer_size = 16
dataset = MyCustomDataset('labels_100',json_file_path="/Users/mjo/Desktop/WLASL/WLASL_v0.3.json", video_file_path="/Users/mjo/Desktop/WLASL/WLASL2000", frame_location="/Users/mjo/Desktop/WLASL/Processed_data/")
dataset_size = (len(dataset))

val_size = int(np.floor(dataset_size * 0.1))
train_size = int(dataset_size - val_size)
trainset, validset = random_split(dataset, [train_size, val_size])
dataloader_train = DataLoader(trainset, batch_size=20, shuffle=True, num_workers=2)
dataloader_val = DataLoader(validset, batch_size=20, shuffle=True, num_workers=2)

net = ConvLSTM(
        num_classes=100,
        latent_dim=512,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )
net = net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion = criterion.to(device)

def accuracy(ys, ts):
    y = torch.argmax(ys, dim = 1)
    x = ts
    correct = 0
    for i in range(len(y)):
        if y[i] == x[i]:
            correct += 1
    return correct/len(y)

now = datetime.now()
filename = "{}".format(now.strftime("%H:%M:%S") + ".csv")
title = ['{}'.format(net)]
headers = ['ID', 'Type','Epoch','Loss','Accuracy']
with open(filename,'w') as csvfile:
    Identification = 1
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(headers)
    for epoch in range(2):  # loop over the dataset multiple times

        net.train()
        training_loss = 0.0
        running_acc = 0
        for i,(inputs, labels) in enumerate(dataloader_train):
            # get the inputs; data is a list of [inputs, labels]
            print(type(inputs[0][0]))
            inputs = inputs.view(-1,3,buffer_size,224,224)
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            
            outputs = net(inputs)
            value, index = (torch.max(outputs,0))
            value, index = (torch.max(labels,0))
            preds = torch.max(outputs, 1)[1]

            loss = criterion(outputs.view(20,100), torch.LongTensor(labels))
            loss.backward()
            training_loss += loss.item()
            optimizer.step()
            running_acc += accuracy(outputs,labels)
            if i % 30 == 0:
                csvwriter.writerow(['{}'.format(Identification),'{}'.format("Training"),'{}'.format(epoch),'{}'.format(training_loss/(i+1)),'{}'.format(running_acc/(i+1))])
                print(f"Training phase, Epoch: {epoch}. Loss: {training_loss/(i+1)}. Accuracy: {running_acc/(i+1)}.")
                Identification += 1
        net.eval()
        valError = 0
        running_acc = 0
        for i, (inputs,labels) in enumerate(dataloader_val):
            with torch.no_grad():
                inputs = inputs.view(-1,3,buffer_size,224,224)
                inputs = inputs.to(device)
                labels = labels.to(device)
                prediction = net(inputs)
                loss = criterion(prediction, labels)
                running_acc += accuracy(outputs,labels)
                valError += loss.item()
            if i % 15 == 0:
                csvwriter.writerow(['{}'.format(Identification),'{}'.format("Training"),'{}'.format(epoch),'{}'.format(valError/(i+1)),'{}'.format(running_acc/(i+1))])
                print(f"Training phase, Epoch: {epoch}. Loss: {training_loss/(i+1)}. Accuracy: {running_acc/(i+1)}.")
                Identification += 1
print('Finished Training')
