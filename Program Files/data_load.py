import os 
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2


class Plain_Dataset(Dataset):
    def __init__(self, csv_file, img_dir, datatype='Training_', transform=None):
        self.csv_file = pd.read_csv(csv_file)  
        self.img_dir = img_dir                
        self.transform = transform            
        self.datatype = datatype              

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_path = self.csv_file.iloc[idx, 0]  
        label = int(self.csv_file.iloc[idx, 1])  
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image not found at {img_path}")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def eval_data_dataloader(csv_file,img_dir,datatype,sample_number,transform= None):
    
    if transform is None :
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    dataset = Plain_Dataset(csv_file=csv_file,img_dir = img_dir,datatype = datatype,transform = transform)

    label = dataset.__getitem__(sample_number)[1]
    print(label)
    imgg = dataset.__getitem__(sample_number)[0]
    imgnumpy = imgg.numpy()
    imgt = imgnumpy.squeeze()
    plt.imshow(imgt)
    plt.show()