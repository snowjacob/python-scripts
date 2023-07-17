import torch
import torch.nn as nn
import torchvision as tv
import argparse
import numpy as np

# Author : Jacob Snow

parser = argparse.ArgumentParser(prog='PytorchCNN.py')
parser.add_argument('--train', type=str, action='store', required=True)
parser.add_argument('--dev', type=str, action='store', required=True)

args = parser.parse_args()

#Hyperparamaters
batch_size = 128

epochs = 15
dropout_rate = 0.745

lr = 0.0005
betas = (0.7707, 0.994)
eps = 4.722e-09
weight_decay = 0.0001296

kernal_size = 4
stride = 2

# Transformations & Normalization (Data Augmentation)
train_transforms = tv.transforms.Compose([
    tv.transforms.RandomRotation(45),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.RandomVerticalFlip(),
    tv.transforms.RandomCrop(256),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])

#No Augmentations on the dev set
dev_transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Loading
train = tv.datasets.ImageFolder(root=args.train, transform=train_transforms)
dev = tv.datasets.ImageFolder(root=args.dev, transform=dev_transforms)

# Model
# Inputs are images that are 256x256
class Net(nn.Module):
    def __init__(self, dropout_rate, kernal_size, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernal_size, stride)
        self.prelu1 = nn.PReLU(8)
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernal_size, stride)
        self.prelu2 = nn.PReLU(16)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernal_size, stride)
        self.conv3 = nn.Conv2d(16, 32, kernal_size, stride)
        self.prelu3 = nn.PReLU(32)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernal_size, stride)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.LazyLinear(64)
        self.prelu4 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 32)
        self.prelu5 = nn.PReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.batch_norm2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.batch_norm3(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.prelu4(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.prelu5(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Loss and optimizer
criterion = nn.BCELoss()

#Train Testing

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dev, batch_size=batch_size, shuffle=True)

def train_one_epoch(model, train_loader, criterion, optimizer):
    total_train_acc = 0
    train_loss = 0
    train_acc = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = torch.flatten(model(images)).float()
        loss = criterion(outputs, labels.float())
        loss.backward()
        train_loss += loss.detach().item()
        optimizer.step()
        train_acc = (outputs.round() == labels.float().round()).float().sum().item()
        train_acc = train_acc * 100 / labels.size(0)
        total_train_acc += train_acc
    return train_loss / len(train_loader), total_train_acc / len(train_loader)

def train_model():
    model = Net(dropout_rate, kernal_size, stride).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    return model


#Dev Testing

def test_model(model, dev_loader, criterion):
    model.eval()
    total_dev_acc = 0
    dev_loss = 0
    dev_acc = 0
    for images, labels in dev_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = torch.flatten(model(images)).float()
        loss = criterion(outputs, labels.float())
        dev_loss += loss.item()
        dev_acc = (outputs.round() == labels.float().round()).float().sum().item()
        dev_acc = dev_acc * 100 / labels.size(0)
        total_dev_acc += dev_acc
    return dev_loss / len(dev_loader), total_dev_acc / len(dev_loader)

# Predictions using the model (assuming we saved the above model as model.pt)

script_file = '.../project_data/script.txt'

model = torch.load('model.pt').to(device)

with open(script_file, 'r') as f:
    script_lines = f.readlines()

test_files = [line.strip() for line in script_lines]

def predict(file_path):
    image = tv.io.read_image('.../project_data/' + file_path)
    image = image.unsqueeze(0).float()
    image = image.to(device)
    output = torch.flatten(model(image)).float()
    return output.item()

predictions = []
for file_path in test_files:
    prediction = predict(file_path)
    predictions.append(prediction)

predictions = np.array(predictions, dtype=np.float32)

np.save('task4_predictions.npy', predictions)

