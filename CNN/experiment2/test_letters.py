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
        self.conv1 = nn.Conv2d(1,64,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(128, 256,kernel_size=3,padding=1)
        self.pool = nn.AdaptiveAvgPool2d((16,16))
        self.fc1 = nn.Linear(256*16*16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 26)
        
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,4)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

model = CNN()
weights = load_file("/home/chelovek/Загрузки/cnn_lettersNew.safetensors")
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
