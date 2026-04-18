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
#------------------------------------------------------------------------------------

# mobilenet_engine.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from huggingface_hub import hf_hub_download


class EngineMobileNetV2:
    """
    Единый класс для инференса MobileNetV2.
    Вся архитектура внутри как вложенные классы.
    """
    class ConvBNReLU(nn.Sequential):
        def __init__(self, in_c, out_c, kernel, stride):
            padding = (kernel - 1) // 2
            super().__init__(
                nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU6(inplace=True)
            )

    class InvertedResidual(nn.Module):
        def __init__(self, in_c, out_c, stride, expand_ratio, ConvBNReLU):
            super().__init__()
            
            hidden = int(in_c * expand_ratio)
            self.use_residual = stride == 1 and in_c == out_c
            
            layers = []
            
            if expand_ratio != 1:
                layers.append(ConvBNReLU(in_c, hidden, 1, 1))
            
            layers += [
                nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(hidden, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c)
            ]
            
            self.block = nn.Sequential(*layers)
        
        def forward(self, x):
            if self.use_residual:
                return x + self.block(x)
            return self.block(x)

    class MobileNetV2(nn.Module):
        def __init__(self, num_classes=64, in_channels=1, width_mult=1.0, 
                     ConvBNReLU=None, InvertedResidual=None):
            super().__init__()
            
            def c(ch):
                return int(ch * width_mult)
            
            self.stem = ConvBNReLU(in_channels, c(32), 3, 2)
            
            self.blocks = nn.Sequential(
                InvertedResidual(c(32), c(16), 1, 1, ConvBNReLU),
                
                InvertedResidual(c(16), c(24), 2, 6, ConvBNReLU),
                InvertedResidual(c(24), c(24), 1, 6, ConvBNReLU),
                
                InvertedResidual(c(24), c(32), 2, 6, ConvBNReLU),
                InvertedResidual(c(32), c(32), 1, 6, ConvBNReLU),
                InvertedResidual(c(32), c(32), 1, 6, ConvBNReLU),
                
                InvertedResidual(c(32), c(64), 2, 6, ConvBNReLU),
                InvertedResidual(c(64), c(64), 1, 6, ConvBNReLU),
                InvertedResidual(c(64), c(64), 1, 6, ConvBNReLU),
                InvertedResidual(c(64), c(64), 1, 6, ConvBNReLU),
                
                InvertedResidual(c(64), c(96), 1, 6, ConvBNReLU),
                InvertedResidual(c(96), c(96), 1, 6, ConvBNReLU),
                InvertedResidual(c(96), c(96), 1, 6, ConvBNReLU),
                
                InvertedResidual(c(96), c(160), 2, 6, ConvBNReLU),
                InvertedResidual(c(160), c(160), 1, 6, ConvBNReLU),
                InvertedResidual(c(160), c(160), 1, 6, ConvBNReLU),
                
                InvertedResidual(c(160), c(320), 1, 6, ConvBNReLU)
            )
            
            self.last = ConvBNReLU(c(320), c(1280), 1, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(c(1280), num_classes)
        
        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            x = self.last(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    # ========================================================================
    # ОСНОВНОЙ КЛАСС ENGINE
    # ========================================================================
    
    def __init__(
        self, 
        repo_id: str = "MarkProMaster229/experimental_models",
        config_path: str = "MobileNetV2/config.json",
        weights_path: str = "MobileNetV2/cnn_model_epoch_15.pth"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Скачиваем config.json
        print(f"📥 Загрузка конфига: {repo_id}/{config_path}")
        config_file = hf_hub_download(repo_id=repo_id, filename=config_path)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 2. Создаём модель
        self.model = self.MobileNetV2(
            num_classes=self.config["num_classes"],
            in_channels=self.config["input_channels"],
            width_mult=self.config.get("width_mult", 1.0),
            ConvBNReLU=self.ConvBNReLU,
            InvertedResidual=self.InvertedResidual
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
        print(f"✅ Engine (MobileNetV2) загружен")
        print(f"   Модель: {self.config.get('model_type', 'MobileNetV2')}")
        print(f"   Классов: {self.config['num_classes']}")
        print(f"   Width multiplier: {self.config.get('width_mult', 1.0)}")
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
    
engineMobile = EngineMobileNetV2()
letter, conf = engineMobile.predict("/home/chelovek/Загрузки/4618_G.jpg")
print(f"{letter} ({conf:.2%})")