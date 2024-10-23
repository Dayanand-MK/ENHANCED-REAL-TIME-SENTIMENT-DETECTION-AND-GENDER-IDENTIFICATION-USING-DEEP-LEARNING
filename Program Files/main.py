import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from deep_emotion import Deep_Emotion
from data_load import Plain_Dataset
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(epochs, train_loader, val_loader, criterion, optimizer, device):
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        train_correct = 0
        val_correct = 0
        net.train()

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        net.eval()
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            val_outputs = net(data)
            val_loss = criterion(val_outputs, labels)
            val_loss += val_loss.item()
            _, preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(preds == labels.data)

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct.double() / len(train_loader.dataset)
        val_acc = val_correct.double() / len(val_loader.dataset)
        print(f"Epoch {epoch+1}: Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
              f"Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
    
    torch.save(net.state_dict(), 'deep_emotion-{}-{}-{}.pt'.format(epochs, batchsize, lr))

epochs = 100
lr = 0.005
batchsize = 128

net = Deep_Emotion()
net.to(device)

train_csv = 'Train1.csv'
test_csv = 'Test_1.csv'
img_dir = "E:/Mini Project/CK+Dataset"

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = Plain_Dataset(csv_file=train_csv, img_dir=img_dir, datatype='Training', transform=trans)
val_dataset = Plain_Dataset(csv_file=test_csv, img_dir=img_dir, datatype='Validation', transform=trans)

train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

train(epochs, train_loader, val_loader, criterion, optimizer, device)