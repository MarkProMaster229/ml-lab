from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ["HF_DATASETS_USE_XET"] = "0"
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

ds = load_dataset(
    "parquet",
    data_files={
        "train": [
            "/home/chelovek/Загрузки/cats_vs_dogs/train-00000-of-00002.parquet",
            "/home/chelovek/Загрузки/cats_vs_dogs/train-00001-of-00002.parquet"
        ]
    }
)
split_ds = ds["train"].train_test_split(test_size=0.2, seed=42)

train_dataset = split_ds['train']
test_dataset = split_ds['test']
print(train_dataset[0].keys())


def transform_fn(example):
    img = example["image"]
    if img.mode != "RGB":
        img = img.convert("RGB")
    example["image"] = transform(img)
    
    example["labels"] = example["labels"]
    return example



train_dataset = [transform_fn(ex) for ex in train_dataset]
test_dataset = [transform_fn(ex) for ex in test_dataset]

def collate_fn(batch):
    images = torch.stack([b['image'] for b in batch])
    labels = torch.tensor([b['labels'] for b in batch])
    return {'image': images, 'labels': labels}

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)






class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1Atten1 = nn.Conv2d(3, 32, 6, padding=1)
        self.conv1Atten2 = nn.Conv2d(3, 32, 6, padding=1)

        self.conv2mulyAtten1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2mulyAtten2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pooltwo  = nn.MaxPool2d(2,2)
        self.pooltwoDow = nn.MaxPool2d(2,2)

        self.pooConv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.pooConv2 = nn.Conv2d(64, 128, 3, padding=1)

        self.conv3mulyAtten1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3mulyAtten2 = nn.Conv2d(128, 256, 3, padding=1)

        self.finalyConvAtten = nn.Conv2d(512, 512, 3, padding=1)

        self.finalyPoolMax = nn.MaxPool2d(2,2)

        #до адаптивного пуллинга 62 на 62 ппц 
        self.attenInput = nn.Conv2d(512, 512, 3, padding=1)



        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))

        self.fc1 = nn.Linear(512*16*16 , 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x, y):

        #print(f"x channels: {x.shape[1]}")
        #print(f"y channels: {y.shape[1]}")
        #print(f"Total after concat: {x.shape[1] + y.shape[1]}")
        #print(x.shape)
        x = F.relu(self.conv1Atten1(x))#64
        y = F.relu(self.conv1Atten2(y))#64

        x = F.relu(self.conv2mulyAtten1(x))#32
        y = F.relu(self.conv2mulyAtten2(y))#32

        x = self.pooltwo(F.relu(self.pooConv1(x)))#16
        y = self.pooltwoDow(F.relu(self.pooConv2(y)))#16


        x = F.relu(self.conv3mulyAtten1(x))#8
        y = F.relu(self.conv3mulyAtten2(y))#8


        finaly = torch.cat([x,y], dim=1)
        x = F.relu(self.finalyConvAtten(finaly))#4

        x = self.finalyPoolMax(F.relu(self.attenInput(x)))

        x = self.adaptive_pool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    





train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=46, shuffle=True, collate_fn=collate_fn
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=46, shuffle=False, collate_fn=collate_fn
)
train_losses = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)#???

for epoch in range(8):
    running_loss = 0
    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        batch_size, channels, height, width = images.shape
        images_top = images[:, :, :height//2, :]
        images_bottom = images[:, :, height//2:, :]
        
        outputs = model(images_top, images_bottom)
        
        loss = criterion(outputs, labels)
        

        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    
from safetensors.torch import save_file
#save_file(model.state_dict(), "cnn_letters.safetensors")
save_file(model.state_dict(), "HOUSEPETS.safetensors")

import matplotlib.pyplot as plt
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.savefig("training_loss15UP.png")

