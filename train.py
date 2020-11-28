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
from Model.I3D_Pytorch import I3D
from torch.utils.data import random_split


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
buffer_size = 64
dataset = MyCustomDataset(category='labels_100')

#dataset = MyCustomDataset(category='labels_100',json_file_path="/home/marius/Documents/Projects/WLASL_v0.3.json", frame_location="/home/marius/Documents/Projects/Processed_data")

dataset_size = (len(dataset))

val_size = int(np.floor(dataset_size * 0.1))
train_size = int(dataset_size - val_size)
trainset, validset = random_split(dataset, [train_size, val_size])
dataloader_train = DataLoader(trainset, batch_size=30, shuffle=True, num_workers=4)
dataloader_val = DataLoader(validset, batch_size=30, shuffle=True, num_workers=4)

net = I3D()
#net.load_state_dict(torch.load('Model/rgb_imagenet.pkl'))
net.replace_logits(100)
net = nn.DataParallel(net)
net = net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay= 0.0000001)
criterion = criterion.to(device)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 3,gamma =0.1)
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
    for epoch in range(10):  # loop over the dataset multiple times

        net.train()
        training_loss = 0.0
        running_acc = 0
        for i,(inputs, labels) in enumerate(dataloader_train):
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.view(-1,3,buffer_size,224,224)
            inputs = inputs.float()

            # for j in range(len(inputs)):
            #     temp = inputs[j]
            #     temp = temp.permute(1,2,3,0)
            
            #     for h in range(len(temp)):
            #         img = temp[h]
            #         imgplot = plt.imshow(img)
            #         plt.show()

            labels = torch.LongTensor(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            #forward + backward + optimize
            
            outputs = net(inputs)
            rgb_score, rgb_logits = outputs
            outputs = rgb_logits
            #print(outputs.shape)
            #print(labels.shape)
            loss = criterion(outputs, labels)
            #print(loss)
            loss.backward()
            training_loss += loss.item()
            optimizer.step()
            #running_acc += accuracy(outputs,labels)
            _,predicted = torch.max(outputs.data,1)
            correct = (predicted == labels).sum().item()
            running_acc += (correct/(labels.size(0)))

            print(f"Training phase, Epoch: {epoch}. Loss: {training_loss/(i+1)}. Accuracy: {running_acc/(i+1)}.")
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
                outputs = net(inputs)
                rgb_score, rgb_logits = outputs
                prediction = rgb_logits
                loss = criterion(prediction, labels)
                _,predicted = torch.max(prediction.data,1)
                correct = (predicted == labels).sum().item()
                running_acc += (correct/(labels.size(0)))

                #running_acc += accuracy(outputs,labels)
                valError += loss.item()
            if i % 15 == 0:
                csvwriter.writerow(['{}'.format(Identification),'{}'.format("Validation"),'{}'.format(epoch),'{}'.format(valError/(i+1)),'{}'.format(running_acc/(i+1))])
                print(f"Validation phase, Epoch: {epoch}. Loss: {valError/(i+1)}. Accuracy: {running_acc/(i+1)}.")
                Identification += 1
        scheduler.step()
print('Finished Training')
