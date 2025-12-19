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

# меня вынудили сделать не так
#class CNN(nn.Module):
#    def __init__(self):
#        super(CNN, self).__init__()
#        self.conv1 = nn.Conv2d(1,64,kernel_size=2,padding=1)
#        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
#        self.conv3 = nn.Conv2d(128, 256, kernel_size=8, padding=1)
#        self.conv4 = nn.Conv2d(256, 512, kernel_size=6, padding=1)
#        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
#        self.fc1 = nn.Linear(1024*8*8, 512)
#        self.fc2 = nn.Linear(512, 256)
#        self.fc3 = nn.Linear(256, 128)
#        self.fc4 = nn.Linear(128, 26)
#    def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = F.relu(self.conv2(x))
#        x = F.max_pool2d(x,8)
#        x = F.relu(self.conv3(x))
#        x = F.relu(self.conv4(x))
#        x = F.max_pool2d(x,4)
#        x = F.relu(self.conv5(x))

#        x = x.view(x.size(0), -1)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = F.relu(self.fc3(x))
#        x = self.fc4(x)
#
#        return x

# а вот так(( но это скучнее
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128*64*64, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    



train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=160, shuffle=True, collate_fn=collate_fn
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=160, shuffle=False, collate_fn=collate_fn
)
train_losses = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)#???

for epoch in range(6):
    running_loss = 0
    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
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

