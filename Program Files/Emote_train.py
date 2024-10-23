import torch
import os 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from deep_emotion import Deep_Emotion
from data_load import Plain_Dataset,eval_data_dataloader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(epochs,train_load,val_load,crit,optimize,device):
    for e in range(epochs):
        train_loss = 0
        val_loss = 0
        train_crt = 0
        val_crt = 0
        net.train()

        for data,lables in train_load:
            data,lables = data.to(device),lables.to(device)
            optimize.zero_grad()
            outputs = net(data)
            loss = crit(outputs,lables)
            loss.backward()
            optimize.step()
            train_loss += loss.item()
            _,preds = torch.max(outputs,1)
            train_crt += torch.sum(preds == lables.data)

        net.eval()
        for data,lables in val_load:
            data,lables = data.to(device),lables.to(device)
            val_out = net(data)
            val_loss = crit(val_out,lables)
            valu_loss = val_loss.item()
            _,preds = torch.max(val_out,1)
            val_crt += torch.sum(preds == lables.data)

        train_loss = train_loss/len(train_dataset)
        train_acc = train_crt.double()/len(train_dataset)
        val_acc = val_crt.double()/len(val_dataset)
        print("Epoch: {}\nTraining Loss : {: .8f}\nValidation Loss : {: .8f}\nTraining Accuracy : {: .3f}".format(e+1,train_loss,val_loss,train_acc*100))
    
    torch.save(net.state_dict(),'deep_emotion-{}-{}-{}.pt'.format(epochs,batchsize,lr))



epochs = 100
lr = 0.005
batchsize = 128

net = Deep_Emotion()
net.to (device)
print(net)
traine = 'Program files\Train1.csv'
vali = "Program files\Train1.csv"
trainimg = "E:\Mini Project\Train1.csv" 
valiimg = "E:\Mini Project\Test_1.csv" 

trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
train_dataset = Plain_Dataset(csv_file = traine,img_dir = trainimg,datatype = 'Training_',transform =trans)
val_dataset = Plain_Dataset(csv_file = vali,img_dir = valiimg,datatype = 'val',transform =trans)
train_load = DataLoader(dataset = train_dataset,batch_size = batchsize,shuffle = True,num_workers=0)
val_load = DataLoader(dataset = val_dataset,batch_size = batchsize,shuffle = True,num_workers=0)

crit = nn.CrossEntropyLoss()
optimize = optim.Adam(net.parameters(),lr = lr)
train(epochs,train_load,val_load,crit,optimize,device)