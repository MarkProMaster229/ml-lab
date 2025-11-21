#работает на основе experiment2
#задача - увеличить сходимость, сохраняя малый размер модели
#может попробывать для задачи классификации 26 букв делать не огромное кол - во слоев?
#модель весом в 6 мб имеет точность Accuracy: 22/26 = 84.62% ! 

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file
import string
from datasets import load_dataset
torch.set_num_threads(12)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

ds = load_dataset("Leeps/Fonts-Individual-Letters", split="train")

alphabet = [chr(i) for i in range(65, 91)]

split_ds = ds.train_test_split(test_size=0.2, seed=42)

train_dataset = split_ds['train']
test_dataset = split_ds['test']

from PIL import Image
import numpy as np
import re
alphabet = [chr(i) for i in range(65, 91)]

from PIL import Image, ImageOps

def transform_fn(example):
    img = example["image"]
    if isinstance(img, list):
        img = np.array(img, dtype=np.uint8)
        while img.ndim > 2:
            img = img[0]
        img = Image.fromarray(img)
    
    img = img.convert("L")
    
    if np.mean(np.array(img)) < 128:
        img = ImageOps.invert(img)
    
    example["image"] = transform(img)
    
    text_list = example["text"]
    if isinstance(text_list, list):
        text = text_list[0]
    else:
        text = text_list
    
    match = re.search(r"TOK '(.?)'", text)
    if match:
        symbol = match.group(1).upper()
        if symbol in alphabet:
            example["label"] = alphabet.index(symbol)
        else:
            return None
    else:
        return None
    
    return example


from torch.utils.data import DataLoader

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    images = torch.stack([b['image'] for b in batch])
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    return {'image': images, 'label': labels}

train_dataset = [ex for ex in train_dataset if transform_fn(ex) is not None]
test_dataset = [ex for ex in test_dataset if transform_fn(ex) is not None]

print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,32, kernel_size=2,padding=1)
        self.conv2 = nn.Conv2d(32,64, kernel_size=4,padding=1)
        self.conv3 = nn.Conv2d(64,128, kernel_size=6,padding=1)
        
        self.fc1 = nn.Linear(128*29*29,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 26)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))#64 
        x = F.relu(self.conv2(x))#32
        x = F.max_pool2d(x,2)#16
        x = F.relu(self.conv3(x))#16
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=360, shuffle=True, collate_fn=collate_fn
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=160, shuffle=False, collate_fn=collate_fn
)
train_losses = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# TODO локальный минимум достигается на ~25 эпохе
for epoch in range(35):
    running_loss = 0
    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

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
save_file(model.state_dict(), "/content/drive/MyDrive/cnn_lettersNew.safetensors")

import matplotlib.pyplot as plt
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid(True)
plt.savefig("training_loss15UP.png")

