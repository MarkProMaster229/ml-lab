import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from safetensors.torch import load_file

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1Atten1 = nn.Conv2d(3, 32, 6, padding=1)
        self.conv1Atten2 = nn.Conv2d(3, 32, 6, padding=1)
        self.conv2mulyAtten1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2mulyAtten2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pooltwo = nn.MaxPool2d(2,2)
        self.pooltwoDow = nn.MaxPool2d(2,2)
        self.pooConv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.pooConv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3mulyAtten1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3mulyAtten2 = nn.Conv2d(128, 256, 3, padding=1)
        self.finalyConvAtten = nn.Conv2d(512, 512, 3, padding=1)
        self.finalyPoolMax = nn.MaxPool2d(2,2)
        self.attenInput = nn.Conv2d(512, 512, 3, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.fc1 = nn.Linear(512*16*16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x, y):
        x = F.relu(self.conv1Atten1(x))
        y = F.relu(self.conv1Atten2(y))
        x = F.relu(self.conv2mulyAtten1(x))
        y = F.relu(self.conv2mulyAtten2(y))
        x = self.pooltwo(F.relu(self.pooConv1(x)))
        y = self.pooltwoDow(F.relu(self.pooConv2(y)))
        x = F.relu(self.conv3mulyAtten1(x))
        y = F.relu(self.conv3mulyAtten2(y))
        finaly = torch.cat([x, y], dim=1)
        x = F.relu(self.finalyConvAtten(finaly))
        x = self.finalyPoolMax(F.relu(self.attenInput(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def robust_load(image_path, target_size=(128, 128)):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    
    if img.mode in ('RGBA', 'LA', 'P'):
        if img.mode == 'P':
            img = img.convert('RGBA')
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = bg
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    return img.resize(target_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
state_dict = load_file("/home/chelovek/exper/HOUSEPETS.safetensors")
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_image(image_path):
    image = robust_load(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    batch, channels, height, width = image_tensor.shape
    images_top = image_tensor[:, :, :height//2, :]
    images_bottom = image_tensor[:, :, height//2:, :]
    
    with torch.no_grad():
        outputs = model(images_top, images_bottom)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    class_names = ["Кошка", "Собака"]
    print(f"Предсказание: {class_names[predicted_class]}")
    print(f"Вероятности: Кошка {probabilities[0]:.3f}, Собака {probabilities[1]:.3f}")
    return predicted_class

predict_image("/home/chelovek/Музыка/iris3.jpg")