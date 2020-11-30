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
from functions import *


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
buffer_size = 64
dataset = MyCustomDataset(category='labels_100')

#dataset = MyCustomDataset(category='labels_2000',json_file_path="/home/marius/Documents/Projects/WLASL_v0.3.json", frame_location="/home/marius/Documents/Projects/Processed_data")

dataset_size = (len(dataset))

val_size = int(np.floor(dataset_size * 0.2))
train_size = int(dataset_size - val_size)
trainset, validset = random_split(dataset, [train_size, val_size])
dataloader_train = DataLoader(trainset, batch_size=20, shuffle=True, num_workers=4)
dataloader_val = DataLoader(validset, batch_size=20, shuffle=True, num_workers=4)
#net = InceptionI3d(num_classes=400)
#net = C3D(num_classes = 100, pretrained = False)
#net.load_state_dict(torch.load('Model/rgb_imagenet.pt'))
#net.replace_logits(100)
# Create model
# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.0       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=100).to(device)
#net = nn.DataParallel(net)
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
                  list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
                  list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                  list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                  list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())
#net = net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(crnn_params.parameters(), lr=1e-3)
criterion = criterion.to(device)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 10,gamma =0.1)
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
#title = ['{}'.format(net)]
headers = ['ID', 'Type','Epoch','Loss','Accuracy']
with open(filename,'w') as csvfile:
    Identification = 1
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(headers)
    for epoch in range(30):  # loop over the dataset multiple times

        cnn_encoder.train()
        rnn_decoder.train()
        training_loss = 0.0
        running_acc = 0
        for i,(inputs, labels) in enumerate(dataloader_train):

            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.view(-1,buffer_size,3,224,224)
            inputs = inputs.float()

            #for j in range(len(inputs)):
            #    temp = inputs[j]
            #    temp = temp.permute(1,2,3,0)
            #
            #    for h in range(len(temp)):
            #        img = temp[h]
            #        imgplot = plt.imshow(img)
            #        plt.show()

            labels = torch.LongTensor(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            #net.lstm.reset_hidden_state()            #forward + backward + optimize

            outputs = rnn_decoder(cnn_encoder(inputs))            #rgb_score, rgb_logits = outputs
            #outputs = rgb_logits
            #print(outputs.shape)
            #print(labels.shape)
            loss = criterion(outputs, labels)
            #print(loss)
            loss.backward()
            loss_value = loss.item()
            training_loss += loss_value
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
        cnn_encoder.eval()
        rnn_decoder.eval()
        valError = 0
        running_acc = 0
        for i, (inputs,labels) in enumerate(dataloader_val):
            with torch.no_grad():
                inputs = inputs.view(-1,buffer_size,3,224,224)
                inputs = inputs.float()
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = rnn_decoder(cnn_encoder(inputs))
                prediction = outputs
                #rgb_score, rgb_logits = outputs
                #prediction = rgb_logits
                loss = criterion(prediction, labels)
                _,predicted = torch.max(prediction.data,1)
                correct = (predicted == labels).sum().item()
                running_acc += (correct/(labels.size(0)))

                #running_acc += accuracy(outputs,labels)
                loss_value = loss.item()
                valError += loss_value
            if i % 10 == 0:
                csvwriter.writerow(['{}'.format(Identification),'{}'.format("Validation"),'{}'.format(epoch),'{}'.format(valError/(i+1)),'{}'.format(running_acc/(i+1))])
                print(f"Validation phase, Epoch: {epoch}. Loss: {valError/(i+1)}. Accuracy: {running_acc/(i+1)}.")
                Identification += 1
        scheduler.step()
print('Finished Training')
