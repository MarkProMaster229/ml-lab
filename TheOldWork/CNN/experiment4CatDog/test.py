from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from cnn_model import CNN
from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = "/home/chelovek/Музыка/standsrt.jpeg"
class_names = ["dog", "cat"]

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

model = CNN()

state_dict = load_file("/home/chelovek/exper/HOUSEPETS.safetensors")
model.load_state_dict(state_dict)

model.to(device)
model.eval()

img = Image.open(image_path).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(img)
    probs = F.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

print(f"Predicted class: {class_names[pred_class]} with probability {probs[0, pred_class]:.4f}")
