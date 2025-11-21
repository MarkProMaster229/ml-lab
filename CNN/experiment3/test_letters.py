import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file
import string

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,32, kernel_size=2,padding=1)
        self.conv2 = nn.Conv2d(32,64, kernel_size=4,padding=1)
        self.conv3 = nn.Conv2d(64,128, kernel_size=6,padding=1)
        
        self.fc1 = nn.Linear(128*5*5,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 26)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))#64 
        x = F.relu(self.conv2(x))#32
        x = F.max_pool2d(x,8)#16
        x = F.relu(self.conv3(x))#16
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = CNN()
weights = load_file("/home/chelovek/Загрузки/cnn_lettersNew (2).safetensors")
model.load_state_dict(weights)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dir = "/home/chelovek/Изображения/words"
letters = list(string.ascii_uppercase)

correct = 0
total = 0

for letter in letters:
    path = os.path.join(test_dir, f"{letter}.png")

    if not os.path.exists(path):
        print(f"[WARN] Нет файла {letter}.png")
        continue

    img = Image.open(path).convert("L")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        pred_idx = output.argmax(dim=1).item()
        pred_letter = letters[pred_idx]

    total += 1
    if pred_letter == letter:
        correct += 1
        print(f"{letter}: OK  → {pred_letter}")
    else:
        print(f"{letter}: FAIL → {pred_letter}")

print("\n---------------------------")
print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
