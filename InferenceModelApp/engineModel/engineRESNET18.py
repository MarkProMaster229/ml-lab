from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import gc
from flask import Flask, request, jsonify, render_template_string, render_template
from flask_cors import CORS

#this file include engine only

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


#CNN engine
#CNN letter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from huggingface_hub import hf_hub_download


class EngineRESNET18:
    """
    Единый класс для инференса ResNet18.
    Вся архитектура внутри как вложенные классы.
    """
    
    # ========================================================================
    # ВЛОЖЕННЫЕ КЛАССЫ АРХИТЕКТУРЫ
    # ========================================================================
    
    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    class ResNet18(nn.Module):
        def __init__(self, num_classes=64, in_channels=1, BasicBlock=None):
            super().__init__()
            
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, 
                                   padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            self.layer1 = self._make_layer(64, 64, blocks=2, stride=1, BasicBlock=BasicBlock)
            self.layer2 = self._make_layer(64, 128, blocks=2, stride=2, BasicBlock=BasicBlock)
            self.layer3 = self._make_layer(128, 256, blocks=2, stride=2, BasicBlock=BasicBlock)
            self.layer4 = self._make_layer(256, 512, blocks=2, stride=2, BasicBlock=BasicBlock)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
            
            self._init_weights()
        
        def _make_layer(self, in_channels, out_channels, blocks, stride, BasicBlock):
            layers = []
            layers.append(BasicBlock(in_channels, out_channels, stride))
            for _ in range(1, blocks):
                layers.append(BasicBlock(out_channels, out_channels, stride=1))
            return nn.Sequential(*layers)
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        
    def __init__(
        self, 
        repo_id: str = "MarkProMaster229/experimental_models",
        config_path: str = "letterResNet18/config.json",
        weights_path: str = "letterResNet18/cnn_model_epoch_8.pth"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Скачиваем config.json
        print(f"📥 Загрузка конфига: {repo_id}/{config_path}")
        config_file = hf_hub_download(repo_id=repo_id, filename=config_path)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 2. Создаём модель
        self.model = self.ResNet18(
            num_classes=self.config["num_classes"],
            in_channels=self.config["input_channels"],
            BasicBlock=self.BasicBlock
        ).to(self.device)
        
        # 3. Грузим веса
        print(f"📥 Загрузка весов: {repo_id}/{weights_path}")
        weights_file = hf_hub_download(repo_id=repo_id, filename=weights_path)
        state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        
        # 4. Трансформации
        self.transform = transforms.Compose([
            transforms.Resize((self.config["img_size"], self.config["img_size"])),
            transforms.Grayscale(num_output_channels=self.config["input_channels"]),
            transforms.ToTensor(),
            transforms.Normalize(
                self.config["normalize"]["mean"],
                self.config["normalize"]["std"]
            )
        ])
        
        # 5. Маппинг меток
        self.id2label = {int(k): v for k, v in self.config["id2label"].items()}
        
        self.model.eval()
        print(f"✅ Engine загружен")
        print(f"   Модель: {self.config.get('model_type', 'ResNet18')}")
        print(f"   Классов: {self.config['num_classes']}")
        print(f"   Устройство: {self.device}")
    
    def predict(self, image_input):
        self.model.eval()
        
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        probs = F.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        
        label = self.id2label.get(pred_class.item(), f"class_{pred_class.item()}")
        
        return label, confidence.item()

engine = EngineRESNET18(
)

letter, conf = engine.predict("/home/chelovek/Загрузки/4618_G.jpg")
print(f"{letter} ({conf:.2%})")
