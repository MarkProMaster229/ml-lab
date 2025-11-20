import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from safetensors.torch import load_file
import string
import onnx

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,64,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(1024*8*8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 26)
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
model = CNN()
state_dict = torch.load("weights.pth", map_location="cpu")
model.load_state_dict(state_dict)
#eval() отключает dropout, batchnorm и т.п., чтобы граф был детерминированным.
model.eval()
dummy_input = torch.randn(1,1,64,64)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    export_params=True,
    do_constant_folding=True,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
#----------------------------------------------------------------------------------------------------------------------------------

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX модель валидна!")
import onnx
from onnxsim import simplify

# Путь к твоему графу ONNX (model.onnx)
graph_file = "model.onnx"

# Загружаем граф
onnx_model = onnx.load(graph_file)

# Применяем упрощение и объединение весов
model_simp, check = simplify(onnx_model)

if check:
    out_file = "model_single.onnx"
    onnx.save(model_simp, out_file)
    print(f" Успешно создан единый ONNX-файл: {out_file}")
else:
    print(" ONNX simplifier не смог проверить модель.")


#--------------------------------------------------------------------------------------------------------------------
import onnx
import os

# Пути к файлам
graph_file = "model.onnx"  # твой граф
output_file = "model_single.onnx"  # новый объединённый файл

# Проверяем, что файл графа существует
if not os.path.exists(graph_file):
    raise FileNotFoundError(f"{graph_file} не найден!")

# Загружаем граф
onnx_model = onnx.load(graph_file)

# На этом этапе все ссылки на веса, которые PyTorch положил в .data, уже доступны через graph.initializer
# Если PyTorch разделил веса в .data, их PyTorch уже связал с графом при экспорте,
# поэтому onnx.load() подхватывает их

# Сохраняем в новый файл
onnx.save(onnx_model, output_file)

print(f"✅ Успешно создан единый ONNX-файл: {output_file}")
print("Теперь файл готов для Android или ONNX Runtime.")
