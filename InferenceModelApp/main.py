from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import gc
from flask import Flask, request, jsonify, render_template_string, render_template
from flask_cors import CORS

#this file include engine only
class Engine:
    def __init__(self):

        repo_name = "MarkProMaster229/experimental_models"
        lora = "loraForArchkit/loraForArch4"

        self.tokenizer = AutoTokenizer.from_pretrained("katanemo/Arch-Router-1.5B")

        base_model = AutoModelForCausalLM.from_pretrained(
            "katanemo/Arch-Router-1.5B",
            device_map="cpu",
            torch_dtype=torch.float32
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            repo_name,
            subfolder=lora
        )

        self.model.eval()
        print("Model loaded")

    def generate(self, prompt):
        prompt = f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# BERT family
class EngineRoBert:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

        repo_name = "MarkProMaster229/experimental_models"
        lora = "ForRoberta_models/loraForROBERTA_epoch4"

        base_model = AutoModelForSequenceClassification.from_pretrained(
            "FacebookAI/xlm-roberta-base",
            num_labels=3,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            repo_name,
            subfolder=lora
        )

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, prompt):
        self.model.eval()

        inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1).max().item()
        print(self.id2label[pred], confidence)

        return self.id2label[pred], confidence

class DistilBert:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        repo_name = "MarkProMaster229/experimental_models"
        lora = "distil_Bert/loraForDistil_Bert_epoch17"

        base_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert/distilbert-base-uncased",
            num_labels=3,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model = PeftModel.from_pretrained(base_model, repo_name, subfolder=lora)
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
    def predict(self, promt):
        
        self.model.eval()
        inputs = self.tokenizer(
        promt,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

        confidence = torch.softmax(logits, dim=-1).max().item()

        return self.id2label[predicted_class], confidence
    

class BaseBert:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        repo_name = "MarkProMaster229/experimental_models"
        lora = "loraBERTVanila/loraForBERT7"
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "google-bert/bert-base-uncased",
            num_labels=3,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model = PeftModel.from_pretrained(base_model, repo_name, subfolder=lora)
    def predict(self, promt):
        self.model.eval()

        inputs = self.tokenizer(
            promt,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

        id2label = {0: "negative", 1: "neutral", 2: "positive"}
        confidence = torch.softmax(logits, dim=-1).max().item()

        return id2label[predicted_class], confidence

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
#-------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from huggingface_hub import hf_hub_download
import numpy as np
import os
class EngineU_net:
    """
    Единый класс для инференса U-Net подобной модели сегментации.
    6 входных каналов (срезы КТ), 6 выходных (маски).
    """
    
    # ========================================================================
    # ВЛОЖЕННЫЕ КЛАССЫ АРХИТЕКТУРЫ
    # ========================================================================
    
    class DiceLoss(nn.Module):
        def __init__(self, smooth=1e-6):
            super().__init__()
            self.smooth = smooth

        def forward(self, predict, target):
            predict = torch.sigmoid(predict)
            predict = predict.view(-1)
            target = target.view(-1)
            intersection = (predict * target).sum()
            dice = (2.*intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth)
            return 1 - dice

    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()
            self.dice = EngineU_net.DiceLoss()

        def forward(self, pred, target):
            return self.bce(pred, target) + self.dice(pred, target)

    class UNetSegmentation(nn.Module):
        """U-Net подобная модель для сегментации с 6 входными и 6 выходными каналами"""
        
        def __init__(self, in_channels=6, out_channels=6):
            super().__init__()
            
            # ЭНКОДЕР
            # Блок 1
            self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
            self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            # Блок 2
            self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            # Блок 3
            self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

            # Блок 4
            self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

            # Блок 5 (bottleneck)
            self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

            # ДЕКОДЕР
            self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec4_1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
            self.dec4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            
            self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec3_1 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
            self.dec3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            
            self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec2_1 = nn.Conv2d(192, 64, kernel_size=3, padding=1)
            self.dec2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            
            self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
            self.dec1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
            self.dec1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            
            self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            # Энкодер
            x = self.relu(self.conv1_1(x))
            x = self.relu(self.conv1_2(x))
            skip1 = x
            x = self.pool1(x)
            
            x = self.relu(self.conv2_1(x))
            x = self.relu(self.conv2_2(x))
            skip2 = x
            x = self.pool2(x)
            
            x = self.relu(self.conv3_1(x))
            x = self.relu(self.conv3_2(x))
            x = self.relu(self.conv3_3(x))
            skip3 = x
            x = self.pool3(x)
            
            x = self.relu(self.conv4_1(x))
            x = self.relu(self.conv4_2(x))
            x = self.relu(self.conv4_3(x))
            skip4 = x
            x = self.pool4(x)
            
            # Bottleneck
            x = self.relu(self.conv5_1(x))
            x = self.relu(self.conv5_2(x))
            x = self.relu(self.conv5_3(x))
            
            # Декодер
            x = self.upconv4(x)
            x = torch.cat([x, skip4], dim=1)
            x = self.relu(self.dec4_1(x))
            x = self.relu(self.dec4_2(x))
            
            x = self.upconv3(x)
            x = torch.cat([x, skip3], dim=1)
            x = self.relu(self.dec3_1(x))
            x = self.relu(self.dec3_2(x))
            
            x = self.upconv2(x)
            x = torch.cat([x, skip2], dim=1)
            x = self.relu(self.dec2_1(x))
            x = self.relu(self.dec2_2(x))
            
            x = self.upconv1(x)
            x = torch.cat([x, skip1], dim=1)
            x = self.relu(self.dec1_1(x))
            x = self.relu(self.dec1_2(x))
            
            out = self.final_conv(x)
            return out

    # ========================================================================
    # ОСНОВНОЙ КЛАСС ENGINE
    # ========================================================================
    
    def __init__(
        self, 
        repo_id: str = "MarkProMaster229/experimental_models",
        config_path: str = "vanila2dEmbol/config.json",
        weights_path: str = "vanila2dEmbol/cnn_model_epoch_4.pth",
        n_slices: int = 6
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_slices = n_slices
        
        # 1. Скачиваем config.json
        print(f"📥 Загрузка конфига: {repo_id}/{config_path}")
        config_file = hf_hub_download(repo_id=repo_id, filename=config_path)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 2. Создаём модель
        self.model = self.UNetSegmentation(
            in_channels=self.config.get("in_channels", 6),
            out_channels=self.config.get("out_channels", 6)
        ).to(self.device)
        
        # 3. Грузим веса
        print(f"📥 Загрузка весов: {repo_id}/{weights_path}")
        weights_file = hf_hub_download(repo_id=repo_id, filename=weights_path)
        state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print(f"✅ Engine (U-Net Segmentation) загружен")
        print(f"   Модель: {self.config.get('model_type', 'UNetSegmentation')}")
        print(f"   Входных каналов: {self.config.get('in_channels', 6)}")
        print(f"   Выходных каналов: {self.config.get('out_channels', 6)}")
        print(f"   Устройство: {self.device}")
    
    def _window_ct(self, pixel_array, slope=1, intercept=0):
        """Применение КТ-окна к одному срезу"""
        image = pixel_array.astype(np.float32) * slope + intercept
        img = np.clip(image, -250, 450)
        return (img - (-250)) / 700
    
    def predict_from_dicom_series(self, dcm_dir: str):
        """
        Предсказание для серии DICOM-файлов из папки.
        
        Args:
            dcm_dir: путь к папке с DICOM-файлами (.dcm)
            
        Returns:
            numpy.ndarray: маска сегментации формы (H, W, 6)
        """
        import pydicom
        
        dcm_files = sorted([f for f in os.listdir(dcm_dir) if f.endswith('.dcm')])
        
        if len(dcm_files) < self.n_slices:
            raise ValueError(f"Нужно минимум {self.n_slices} срезов, найдено {len(dcm_files)}")
        
        # Берём центральные срезы
        start_idx = (len(dcm_files) - self.n_slices) // 2
        
        slices = []
        for i in range(start_idx, start_idx + self.n_slices):
            dcm_path = os.path.join(dcm_dir, dcm_files[i])
            ds = pydicom.dcmread(dcm_path)
            
            intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
            slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
            
            windowed = self._window_ct(ds.pixel_array, slope, intercept)
            slices.append(windowed)
        
        # Стек: (6, H, W)
        input_tensor = torch.tensor(np.stack(slices, axis=0), dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)  # (1, 6, H, W)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.sigmoid(output)  # в вероятности 0-1
        
        # (1, 6, H, W) -> (H, W, 6)
        mask = output.squeeze(0).cpu().numpy()
        mask = np.transpose(mask, (1, 2, 0))
        
        return mask
    
    def predict_from_numpy(self, ct_volume: np.ndarray):
        """
        Предсказание для numpy-массива срезов КТ.
        
        Args:
            ct_volume: numpy-массив формы (n_slices, H, W)
            
        Returns:
            numpy.ndarray: маска сегментации формы (H, W, 6)
        """
        if ct_volume.shape[0] != self.n_slices:
            raise ValueError(f"Ожидалось {self.n_slices} срезов, получено {ct_volume.shape[0]}")
        
        input_tensor = torch.tensor(ct_volume, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.sigmoid(output)
        
        mask = output.squeeze(0).cpu().numpy()
        mask = np.transpose(mask, (1, 2, 0))
        
        return mask
    
    def predict_binary_masks(self, ct_volume: np.ndarray, threshold: float = 0.5):
        """
        Возвращает бинарные маски по порогу.
        
        Returns:
            dict: словарь с масками для каждого класса
        """
        prob_mask = self.predict_from_numpy(ct_volume)
        binary_mask = (prob_mask > threshold).astype(np.uint8)
        
        return {
            "probabilities": prob_mask,
            "binary": binary_mask,
            "threshold": threshold
        }
        
    def visualize_sequence(self, dcm_dir: str, mat_path: str = None, threshold: float = 0.5):
        import matplotlib.pyplot as plt
        import pydicom
        import scipy.io
        
        # Загружаем Ground Truth
        if mat_path and os.path.exists(mat_path):
            gt_mask_original = scipy.io.loadmat(mat_path)['Mask']  # (H, W, N)
            has_gt = True
        else:
            gt_mask_original = None
            has_gt = False
        
        # Загружаем DICOM
        dcm_files = sorted([f for f in os.listdir(dcm_dir) if f.endswith('.dcm')])
        start_idx = (len(dcm_files) - self.n_slices) // 2
        
        ct_slices_original = []
        for i in range(start_idx, start_idx + self.n_slices):
            dcm_path = os.path.join(dcm_dir, dcm_files[i])
            ds = pydicom.dcmread(dcm_path)
            intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
            slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
            windowed = self._window_ct(ds.pixel_array, slope, intercept)
            ct_slices_original.append(windowed)
        
        # Текущее состояние
        ct_slices = [s.copy() for s in ct_slices_original]
        gt_mask = gt_mask_original.copy() if has_gt else None
        
        def apply_transform(transform_type):
            nonlocal ct_slices, gt_mask, pred_mask, pred_binary
            
            if transform_type == 'rotate':
                ct_slices = [np.rot90(s, 2) for s in ct_slices]
                if has_gt:
                    gt_mask = np.rot90(gt_mask, 2, axes=(0,1))  # Трансформируем ВСЮ 3D маску
                print("🔄 Применён ПОВОРОТ 180°")
            elif transform_type == 'shift':
                ct_slices = [np.roll(s, 50, axis=1) for s in ct_slices]
                if has_gt:
                    gt_mask = np.roll(gt_mask, 50, axis=1)  # Трансформируем ВСЮ 3D маску
                print("➡️ Применён СДВИГ ВПРАВО на 50px")
            elif transform_type == 'flip':
                ct_slices = [np.fliplr(s) for s in ct_slices]
                if has_gt:
                    gt_mask = np.fliplr(gt_mask)  # Трансформируем ВСЮ 3D маску
                print("🪞 Применёно ЗЕРКАЛО")
            elif transform_type == 'reset':
                ct_slices = [s.copy() for s in ct_slices_original]
                gt_mask = gt_mask_original.copy() if has_gt else None
                print("🔄 Сброс к оригиналу")
            
            # Пересчитываем предсказания
            ct_volume = np.stack(ct_slices, axis=0)
            pred_mask = self.predict_from_numpy(ct_volume)
            pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        # Начальное предсказание
        ct_volume = np.stack(ct_slices, axis=0)
        pred_mask = self.predict_from_numpy(ct_volume)
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        # Визуализация
        fig, axes = plt.subplots(1, 3 if has_gt else 2, figsize=(15, 5))
        if not has_gt:
            axes = [axes[0], axes[1]]
        
        colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1)]
        
        current_idx = 0
        current_transform = "ОРИГИНАЛ"
        
        def update_display(idx):
            for ax in axes:
                ax.clear()
            
            # КТ
            axes[0].imshow(ct_slices[idx], cmap='gray')
            axes[0].set_title(f'КТ срез {idx+1}/6 [{current_transform}]')
            axes[0].axis('off')
            
            # Ground Truth
            if has_gt and gt_mask is not None:
                gt_slice_idx = start_idx + idx
                if gt_slice_idx < gt_mask.shape[2]:
                    gt_slice = gt_mask[:, :, gt_slice_idx]
                    
                    gt_overlay = np.zeros((*ct_slices[idx].shape, 3))
                    ct_norm = (ct_slices[idx] - ct_slices[idx].min()) / (ct_slices[idx].max() - ct_slices[idx].min() + 1e-8)
                    gt_overlay[:, :, 0] = ct_norm
                    gt_overlay[:, :, 1] = ct_norm
                    gt_overlay[:, :, 2] = ct_norm
                    
                    for c in range(1, 6):
                        if c <= gt_slice.max():
                            class_mask = (gt_slice == c)
                            if class_mask.sum() > 0:
                                for ch in range(3):
                                    gt_overlay[:, :, ch] = np.where(
                                        class_mask,
                                        gt_overlay[:, :, ch] * 0.5 + colors[c-1][ch] * 0.5,
                                        gt_overlay[:, :, ch]
                                    )
                    axes[1].imshow(gt_overlay)
                    axes[1].set_title(f'GT (трансформирована)')
                axes[1].axis('off')
            
            # Предсказание
            pred_ax = axes[2] if has_gt else axes[1]
            pred_overlay = np.zeros((*ct_slices[idx].shape, 3))
            ct_norm = (ct_slices[idx] - ct_slices[idx].min()) / (ct_slices[idx].max() - ct_slices[idx].min() + 1e-8)
            pred_overlay[:, :, 0] = ct_norm
            pred_overlay[:, :, 1] = ct_norm
            pred_overlay[:, :, 2] = ct_norm
            
            for c in range(1, 6):
                if pred_binary[:, :, c].sum() > 0:
                    for ch in range(3):
                        pred_overlay[:, :, ch] = np.where(
                            pred_binary[:, :, c] > 0,
                            pred_overlay[:, :, ch] * 0.5 + colors[c-1][ch] * 0.5,
                            pred_overlay[:, :, ch]
                        )
            pred_ax.imshow(pred_overlay)
            pred_ax.set_title(f'Pred (th={threshold})')
            pred_ax.axis('off')
            
            fig.suptitle(f'← → навигация | T:поворот | S:сдвиг | F:зеркало | R:сброс', fontsize=12)
            fig.canvas.draw()
        
        def on_key(event):
            nonlocal current_idx, current_transform
            if event.key == 'left':
                current_idx = max(0, current_idx - 1)
                update_display(current_idx)
            elif event.key == 'right':
                current_idx = min(5, current_idx + 1)
                update_display(current_idx)
            elif event.key == 't':
                current_transform = "ПОВОРОТ 180°"
                apply_transform('rotate')
                update_display(current_idx)
            elif event.key == 's':
                current_transform = "СДВИГ ВПРАВО"
                apply_transform('shift')
                update_display(current_idx)
            elif event.key == 'f':
                current_transform = "ЗЕРКАЛО"
                apply_transform('flip')
                update_display(current_idx)
            elif event.key == 'r':
                current_transform = "ОРИГИНАЛ"
                apply_transform('reset')
                update_display(current_idx)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        update_display(0)
        
        print("\n" + "="*60)
        print("🎮 УПРАВЛЕНИЕ:")
        print("   ← →  : переключение срезов")
        print("   T    : ПОВОРОТ 180°")
        print("   S    : СДВИГ ВПРАВО")
        print("   F    : ЗЕРКАЛО")
        print("   R    : СБРОС")
        print("="*60 + "\n")
        
        plt.show()
    @staticmethod
    def demo():
        """
        Демо: проходит по пациентам 33, 34, 35 с интерактивной визуализацией
        """
        import kagglehub
        import os
        import glob
        
        print("📥 Загрузка датасета...")
        dataset_path = kagglehub.dataset_download("andrewmvd/pulmonary-embolism-in-ct-images")
        
        # Ищем все папки с DICOM
        all_patients = {}
        for root, dirs, files in os.walk(dataset_path):
            dcm_files = [f for f in files if f.endswith('.dcm')]
            if len(dcm_files) > 100:
                patient_id = os.path.basename(root)
                all_patients[patient_id] = root
        
        # Сортируем и выбираем 33, 34, 35 (если есть)
        target_patients = ['PAT033', 'PAT034', 'PAT035']
        selected = {}
        for pid in target_patients:
            if pid in all_patients:
                selected[pid] = all_patients[pid]
            else:
                # Ищем любые другие, если нет точных совпадений
                matching = [p for p in all_patients.keys() if pid in p or pid.replace('0', '') in p]
                if matching:
                    selected[pid] = all_patients[matching[0]]
        
        if not selected:
            # Берём первых трёх попавшихся
            for i, (pid, path) in enumerate(list(all_patients.items())[:3]):
                selected[f"пациент_{i+1}"] = path
        
        print(f"🤖 Загрузка модели...")
        engine = EngineU_net()
        
        for patient_name, dcm_dir in selected.items():
            patient_id = os.path.basename(dcm_dir)
            
            # Ищем .mat файл
            mat_path = None
            search_paths = [
                os.path.join(dataset_path, f"{patient_id}.mat"),
                os.path.join(os.path.dirname(dcm_dir), f"{patient_id}.mat"),
                os.path.join(dataset_path, "GroundTruth", f"{patient_id}.mat"),
            ]
            
            for path in search_paths:
                if os.path.exists(path):
                    mat_path = path
                    break
            
            if mat_path is None:
                mat_files = glob.glob(os.path.join(dataset_path, "**", "*.mat"), recursive=True)
                for mf in mat_files:
                    if patient_id in mf:
                        mat_path = mf
                        break
            
            print("\n" + "="*60)
            print(f"📂 ПАЦИЕНТ: {patient_id}")
            print("="*60)
            print(f"📁 DICOM: {dcm_dir}")
            if mat_path:
                print(f"📋 Разметка: {mat_path}")
            else:
                print(f"⚠️ Разметка не найдена")
            
            print(f"\n🎮 УПРАВЛЕНИЕ:")
            print("   ← → срезы | T:поворот 180° | S:сдвиг | F:зеркало | R:сброс")
            print("   Закройте окно для перехода к следующему пациенту\n")
            
            engine.visualize_sequence(
                dcm_dir=dcm_dir, 
                mat_path=mat_path, 
                threshold=0.5
            )
            
            # Спрашиваем, продолжать ли
            if patient_name != list(selected.keys())[-1]:
                response = input(f"\n✅ Пациент {patient_id} просмотрен. Продолжить? (Y/n): ")
                if response.lower() == 'n':
                    print("👋 Демо завершено.")
                    return
        
        print("\n🎉 Все пациенты просмотрены!")
y = EngineU_net()

y.demo()
#-------------------------------------------------------------------------------------------------
#AttentionCNNModel
# attention_unet_engine.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from huggingface_hub import hf_hub_download
import pydicom
import matplotlib.pyplot as plt
import scipy.io


class EngineAttentionUNet:
    """
    Движок для Attention U-Net сегментации.
    6 входных каналов (срезы КТ), 6 выходных (маски).
    """
    
    # ========================================================================
    # ВЛОЖЕННЫЕ КЛАССЫ АРХИТЕКТУРЫ (без изменений)
    # ========================================================================
    
    class DiceLoss(nn.Module):
        def __init__(self, smooth=1e-6):
            super().__init__()
            self.smooth = smooth

        def forward(self, predict, target):
            predict = torch.sigmoid(predict)
            predict = predict.view(-1)
            target = target.view(-1)
            intersection = (predict * target).sum()
            dice = (2.*intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth)
            return 1 - dice

    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()
            self.dice = EngineAttentionUNet.DiceLoss()

        def forward(self, pred, target):
            return self.bce(pred, target) + self.dice(pred, target)

    class AttentionBlock(nn.Module):
        def __init__(self, F_g, F_l, F_int):
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
            )
            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(F_int)
            )
            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi

    class AttentionUNet(nn.Module):
        def __init__(self, in_channels=6, out_channels=6):
            super().__init__()
            
            self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
            self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
            self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
            self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

            AttentionBlock = EngineAttentionUNet.AttentionBlock
            self.att4 = AttentionBlock(F_g=256, F_l=512, F_int=256)
            self.att3 = AttentionBlock(F_g=128, F_l=256, F_int=128)
            self.att2 = AttentionBlock(F_g=64, F_l=128, F_int=64)
            self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)

            self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec4_1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
            self.dec4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            
            self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec3_1 = nn.Conv2d(384, 128, kernel_size=3, padding=1)
            self.dec3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            
            self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec2_1 = nn.Conv2d(192, 64, kernel_size=3, padding=1)
            self.dec2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            
            self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
            self.dec1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
            self.dec1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            
            self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.relu(self.conv1_1(x))
            x = self.relu(self.conv1_2(x))
            skip1 = x
            x = self.pool1(x)
            
            x = self.relu(self.conv2_1(x))
            x = self.relu(self.conv2_2(x))
            skip2 = x
            x = self.pool2(x)
            
            x = self.relu(self.conv3_1(x))
            x = self.relu(self.conv3_2(x))
            x = self.relu(self.conv3_3(x))
            skip3 = x
            x = self.pool3(x)
            
            x = self.relu(self.conv4_1(x))
            x = self.relu(self.conv4_2(x))
            x = self.relu(self.conv4_3(x))
            skip4 = x
            x = self.pool4(x)
            
            x = self.relu(self.conv5_1(x))
            x = self.relu(self.conv5_2(x))
            x = self.relu(self.conv5_3(x))
            
            x = self.upconv4(x)
            skip4 = self.att4(g=x, x=skip4)
            x = torch.cat([x, skip4], dim=1)
            x = self.relu(self.dec4_1(x))
            x = self.relu(self.dec4_2(x))
            
            x = self.upconv3(x)
            skip3 = self.att3(g=x, x=skip3)
            x = torch.cat([x, skip3], dim=1)
            x = self.relu(self.dec3_1(x))
            x = self.relu(self.dec3_2(x))
            
            x = self.upconv2(x)
            skip2 = self.att2(g=x, x=skip2)
            x = torch.cat([x, skip2], dim=1)
            x = self.relu(self.dec2_1(x))
            x = self.relu(self.dec2_2(x))
            
            x = self.upconv1(x)
            skip1 = self.att1(g=x, x=skip1)
            x = torch.cat([x, skip1], dim=1)
            x = self.relu(self.dec1_1(x))
            x = self.relu(self.dec1_2(x))
            
            out = self.final_conv(x)
            return out

    # ========================================================================
    # ОСНОВНОЙ КЛАСС ENGINE
    # ========================================================================
    
    def __init__(
        self, 
        repo_id: str = "MarkProMaster229/experimental_models",
        config_path: str = "ModelAttention2dEmbol/config.json",
        weights_path: str = "ModelAttention2dEmbol/cnn_model_epoch_9.pth",
        n_slices: int = 6
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_slices = n_slices
        
        print(f"📥 Загрузка конфига: {repo_id}/{config_path}")
        config_file = hf_hub_download(repo_id=repo_id, filename=config_path)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.model = self.AttentionUNet(
            in_channels=self.config.get("in_channels", 6),
            out_channels=self.config.get("out_channels", 6)
        ).to(self.device)
        
        print(f"📥 Загрузка весов: {repo_id}/{weights_path}")
        weights_file = hf_hub_download(repo_id=repo_id, filename=weights_path)
        state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print(f"✅ Engine (Attention U-Net) загружен")
        print(f"   Устройство: {self.device}")
    
    def _window_ct(self, pixel_array, slope=1, intercept=0):
        image = pixel_array.astype(np.float32) * slope + intercept
        img = np.clip(image, -250, 450)
        return (img - (-250)) / 700
    
    def predict_from_numpy(self, ct_volume: np.ndarray):
        if ct_volume.shape[0] != self.n_slices:
            raise ValueError(f"Ожидалось {self.n_slices} срезов, получено {ct_volume.shape[0]}")
        
        input_tensor = torch.tensor(ct_volume, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.sigmoid(output)
        
        mask = output.squeeze(0).cpu().numpy()
        mask = np.transpose(mask, (1, 2, 0))
        return mask
    
    def visualize_sequence(self, dcm_dir: str, mat_path: str = None, threshold: float = 0.5):
        import pydicom
        import scipy.io
        
        if mat_path and os.path.exists(mat_path):
            gt_mask_original = scipy.io.loadmat(mat_path)['Mask']
            has_gt = True
        else:
            gt_mask_original = None
            has_gt = False
        
        dcm_files = sorted([f for f in os.listdir(dcm_dir) if f.endswith('.dcm')])
        start_idx = (len(dcm_files) - self.n_slices) // 2
        
        ct_slices_original = []
        for i in range(start_idx, start_idx + self.n_slices):
            dcm_path = os.path.join(dcm_dir, dcm_files[i])
            ds = pydicom.dcmread(dcm_path)
            intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
            slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
            windowed = self._window_ct(ds.pixel_array, slope, intercept)
            ct_slices_original.append(windowed)
        
        ct_slices = [s.copy() for s in ct_slices_original]
        gt_mask = gt_mask_original.copy() if has_gt else None
        
        def apply_transform(transform_type):
            nonlocal ct_slices, gt_mask, pred_mask, pred_binary
            
            if transform_type == 'rotate':
                ct_slices = [np.rot90(s, 2) for s in ct_slices]
                if has_gt:
                    gt_mask = np.rot90(gt_mask, 2, axes=(0,1))
                print("🔄 ПОВОРОТ 180°")
            elif transform_type == 'shift':
                ct_slices = [np.roll(s, 50, axis=1) for s in ct_slices]
                if has_gt:
                    gt_mask = np.roll(gt_mask, 50, axis=1)
                print("➡️ СДВИГ ВПРАВО")
            elif transform_type == 'flip':
                ct_slices = [np.fliplr(s) for s in ct_slices]
                if has_gt:
                    gt_mask = np.fliplr(gt_mask)
                print("🪞 ЗЕРКАЛО")
            elif transform_type == 'reset':
                ct_slices = [s.copy() for s in ct_slices_original]
                gt_mask = gt_mask_original.copy() if has_gt else None
                print("🔄 СБРОС")
            
            ct_volume = np.stack(ct_slices, axis=0)
            pred_mask = self.predict_from_numpy(ct_volume)
            pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        ct_volume = np.stack(ct_slices, axis=0)
        pred_mask = self.predict_from_numpy(ct_volume)
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        fig, axes = plt.subplots(1, 3 if has_gt else 2, figsize=(15, 5))
        if not has_gt:
            axes = [axes[0], axes[1]]
        
        colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1)]
        current_idx = 0
        current_transform = "ОРИГИНАЛ"
        
        def update_display(idx):
            for ax in axes:
                ax.clear()
            
            axes[0].imshow(ct_slices[idx], cmap='gray')
            axes[0].set_title(f'КТ срез {idx+1}/6 [{current_transform}]')
            axes[0].axis('off')
            
            if has_gt and gt_mask is not None:
                gt_slice_idx = start_idx + idx
                if gt_slice_idx < gt_mask.shape[2]:
                    gt_slice = gt_mask[:, :, gt_slice_idx]
                    gt_overlay = np.zeros((*ct_slices[idx].shape, 3))
                    ct_norm = (ct_slices[idx] - ct_slices[idx].min()) / (ct_slices[idx].max() - ct_slices[idx].min() + 1e-8)
                    gt_overlay[:, :, 0] = ct_norm
                    gt_overlay[:, :, 1] = ct_norm
                    gt_overlay[:, :, 2] = ct_norm
                    
                    for c in range(1, 6):
                        if c <= gt_slice.max():
                            class_mask = (gt_slice == c)
                            if class_mask.sum() > 0:
                                for ch in range(3):
                                    gt_overlay[:, :, ch] = np.where(
                                        class_mask,
                                        gt_overlay[:, :, ch] * 0.5 + colors[c-1][ch] * 0.5,
                                        gt_overlay[:, :, ch]
                                    )
                    axes[1].imshow(gt_overlay)
                    axes[1].set_title(f'GT')
                axes[1].axis('off')
            
            pred_ax = axes[2] if has_gt else axes[1]
            pred_overlay = np.zeros((*ct_slices[idx].shape, 3))
            ct_norm = (ct_slices[idx] - ct_slices[idx].min()) / (ct_slices[idx].max() - ct_slices[idx].min() + 1e-8)
            pred_overlay[:, :, 0] = ct_norm
            pred_overlay[:, :, 1] = ct_norm
            pred_overlay[:, :, 2] = ct_norm
            
            for c in range(1, 6):
                if pred_binary[:, :, c].sum() > 0:
                    for ch in range(3):
                        pred_overlay[:, :, ch] = np.where(
                            pred_binary[:, :, c] > 0,
                            pred_overlay[:, :, ch] * 0.5 + colors[c-1][ch] * 0.5,
                            pred_overlay[:, :, ch]
                        )
            pred_ax.imshow(pred_overlay)
            pred_ax.set_title(f'Pred (th={threshold})')
            pred_ax.axis('off')
            
            fig.suptitle(f'← → срезы | T:поворот | S:сдвиг | F:зеркало | R:сброс', fontsize=12)
            fig.canvas.draw()
        
        def on_key(event):
            nonlocal current_idx, current_transform
            if event.key == 'left':
                current_idx = max(0, current_idx - 1)
                update_display(current_idx)
            elif event.key == 'right':
                current_idx = min(5, current_idx + 1)
                update_display(current_idx)
            elif event.key == 't':
                current_transform = "ПОВОРОТ 180°"
                apply_transform('rotate')
                update_display(current_idx)
            elif event.key == 's':
                current_transform = "СДВИГ ВПРАВО"
                apply_transform('shift')
                update_display(current_idx)
            elif event.key == 'f':
                current_transform = "ЗЕРКАЛО"
                apply_transform('flip')
                update_display(current_idx)
            elif event.key == 'r':
                current_transform = "ОРИГИНАЛ"
                apply_transform('reset')
                update_display(current_idx)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        update_display(0)
        
        print("\n" + "="*60)
        print("🎮 ← → срезы | T:поворот | S:сдвиг | F:зеркало | R:сброс")
        print("="*60 + "\n")
        
        plt.show()

    @staticmethod
    def demo():
        """
        Демо: качает датасет с Kaggle, загружает модель, 
        запускает интерактивную визуализацию для пациентов 33, 34, 35
        """
        import kagglehub
        import glob
        
        print("📥 Загрузка датасета с Kaggle...")
        dataset_path = kagglehub.dataset_download("andrewmvd/pulmonary-embolism-in-ct-images")
        
        # Ищем всех пациентов
        all_patients = {}
        for root, dirs, files in os.walk(dataset_path):
            dcm_files = [f for f in files if f.endswith('.dcm')]
            if len(dcm_files) > 100:
                patient_id = os.path.basename(root)
                all_patients[patient_id] = root
        
        # Выбираем 33, 34, 35
        target = ['PAT033', 'PAT034', 'PAT035']
        selected = {pid: all_patients[pid] for pid in target if pid in all_patients}
        
        if not selected:
            print("❌ Пациенты 33, 34, 35 не найдены")
            return
        
        print(f"🤖 Загрузка модели Attention U-Net...")
        engine = EngineAttentionUNet()
        
        for patient_name, dcm_dir in selected.items():
            patient_id = os.path.basename(dcm_dir)
            
            # Ищем .mat файл
            mat_path = None
            search_paths = [
                os.path.join(dataset_path, f"{patient_id}.mat"),
                os.path.join(os.path.dirname(dcm_dir), f"{patient_id}.mat"),
                os.path.join(dataset_path, "GroundTruth", f"{patient_id}.mat"),
            ]
            
            for path in search_paths:
                if os.path.exists(path):
                    mat_path = path
                    break
            
            if mat_path is None:
                mat_files = glob.glob(os.path.join(dataset_path, "**", "*.mat"), recursive=True)
                for mf in mat_files:
                    if patient_id in mf:
                        mat_path = mf
                        break
            
            print("\n" + "="*60)
            print(f"📂 ПАЦИЕНТ: {patient_id}")
            print("="*60)
            print(f"📁 DICOM: {dcm_dir}")
            if mat_path:
                print(f"📋 Разметка: {mat_path}")
            else:
                print(f"⚠️ Разметка не найдена")
            
            print(f"\n🎮 УПРАВЛЕНИЕ:")
            print("   ← → срезы | T:поворот | S:сдвиг | F:зеркало | R:сброс")
            print("   Закройте окно для перехода к следующему\n")
            
            engine.visualize_sequence(dcm_dir=dcm_dir, mat_path=mat_path, threshold=0.5)
            
            if patient_name != list(selected.keys())[-1]:
                response = input(f"\n✅ {patient_id} просмотрен. Продолжить? (Y/n): ")
                if response.lower() == 'n':
                    print("👋 Демо завершено.")
                    return
        
        print("\n🎉 Все пациенты просмотрены!")

TheTestAttentionU_net = EngineAttentionUNet()
TheTestAttentionU_net.demo()

#-----------------------------------------------------------------------------------
#3D U-Net с VGG-подобным энкодером
# unet3d_engine.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt


class EngineUNet3D:
    """
    Движок для 3D U-Net сегментации.
    1 входной канал (объём 6×H×W), 1 выходной канал (маска 6×H×W).
    """
    
    # ========================================================================
    # ВЛОЖЕННЫЕ КЛАССЫ АРХИТЕКТУРЫ
    # ========================================================================
    
    class DiceLoss(nn.Module):
        def __init__(self, smooth=1e-6):
            super().__init__()
            self.smooth = smooth

        def forward(self, predict, target):
            predict = torch.sigmoid(predict)
            predict = predict.view(-1)
            target = target.view(-1)
            intersection = (predict * target).sum()
            dice = (2.*intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth)
            return 1 - dice

    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()
            self.dice = EngineUNet3D.DiceLoss()

        def forward(self, pred, target):
            return self.bce(pred, target) + self.dice(pred, target)

    class UNet3D(nn.Module):
        def __init__(self, in_channels=1, out_channels=1):
            super().__init__()
            
            # ЭНКОДЕР
            self.conv1_1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
            self.conv1_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv2_1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
            self.conv2_2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(3, 2, 2))

            self.conv3_1 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
            self.conv3_2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
            self.conv3_3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv4_1 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
            self.conv4_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
            self.conv4_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
            self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv5_1 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
            self.conv5_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
            self.conv5_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)

            # ДЕКОДЕР
            self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.dec4_1 = nn.Conv3d(768, 256, kernel_size=3, padding=1)
            self.dec4_2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
            
            self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.dec3_1 = nn.Conv3d(384, 128, kernel_size=3, padding=1)
            self.dec3_2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
            
            self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 2, 2), stride=(3, 2, 2))
            self.dec2_1 = nn.Conv3d(192, 64, kernel_size=3, padding=1)
            self.dec2_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
            
            self.upconv1 = nn.ConvTranspose3d(64, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            self.dec1_1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
            self.dec1_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
            
            self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.relu(self.conv1_1(x))
            x = self.relu(self.conv1_2(x))
            skip1 = x
            x = self.pool1(x)
            
            x = self.relu(self.conv2_1(x))
            x = self.relu(self.conv2_2(x))
            skip2 = x
            x = self.pool2(x)
            
            x = self.relu(self.conv3_1(x))
            x = self.relu(self.conv3_2(x))
            x = self.relu(self.conv3_3(x))
            skip3 = x
            x = self.pool3(x)
            
            x = self.relu(self.conv4_1(x))
            x = self.relu(self.conv4_2(x))
            x = self.relu(self.conv4_3(x))
            skip4 = x
            x = self.pool4(x)
            
            x = self.relu(self.conv5_1(x))
            x = self.relu(self.conv5_2(x))
            x = self.relu(self.conv5_3(x))
            
            x = self.upconv4(x)
            x = torch.cat([x, skip4], dim=1)
            x = self.relu(self.dec4_1(x))
            x = self.relu(self.dec4_2(x))
            
            x = self.upconv3(x)
            x = torch.cat([x, skip3], dim=1)
            x = self.relu(self.dec3_1(x))
            x = self.relu(self.dec3_2(x))
            
            x = self.upconv2(x)
            x = torch.cat([x, skip2], dim=1)
            x = self.relu(self.dec2_1(x))
            x = self.relu(self.dec2_2(x))
            
            x = self.upconv1(x)
            x = torch.cat([x, skip1], dim=1)
            x = self.relu(self.dec1_1(x))
            x = self.relu(self.dec1_2(x))
            
            out = self.final_conv(x)
            return out.squeeze(1)

    # ========================================================================
    # ОСНОВНОЙ КЛАСС ENGINE
    # ========================================================================
    
    def __init__(
        self, 
        repo_id: str = "MarkProMaster229/experimental_models",
        config_path: str = "3DVanillaCNNEmbol/config.json",
        weights_path: str = "3DVanillaCNNEmbol/cnn_model_epoch_9.pth",
        n_slices: int = 6
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_slices = n_slices
        
        print(f"📥 Загрузка конфига: {repo_id}/{config_path}")
        config_file = hf_hub_download(repo_id=repo_id, filename=config_path)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.model = self.UNet3D(
            in_channels=self.config.get("in_channels", 1),
            out_channels=self.config.get("out_channels", 1)
        ).to(self.device)
        
        print(f"📥 Загрузка весов: {repo_id}/{weights_path}")
        weights_file = hf_hub_download(repo_id=repo_id, filename=weights_path)
        state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print(f"✅ Engine (3D U-Net) загружен")
        print(f"   Устройство: {self.device}")
    
    def _window_ct(self, pixel_array, slope=1, intercept=0):
        image = pixel_array.astype(np.float32) * slope + intercept
        img = np.clip(image, -250, 450)
        return (img - (-250)) / 700
    
    def predict_from_numpy(self, ct_volume: np.ndarray):
        """
        ct_volume: (6, H, W)
        Возвращает: (6, H, W) — бинарная маска
        """
        if ct_volume.shape[0] != self.n_slices:
            raise ValueError(f"Ожидалось {self.n_slices} срезов, получено {ct_volume.shape[0]}")
        
        # (6, H, W) -> (1, 1, 6, H, W)
        input_tensor = torch.tensor(ct_volume, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.sigmoid(output)
        
        # (1, 6, H, W) -> (6, H, W)
        mask = output.squeeze(0).cpu().numpy()
        return mask
    
    def visualize_sequence(self, dcm_dir: str, mat_path: str = None, threshold: float = 0.5):
        import pydicom
        import scipy.io
        
        if mat_path and os.path.exists(mat_path):
            gt_mask_original = scipy.io.loadmat(mat_path)['Mask']
            has_gt = True
        else:
            gt_mask_original = None
            has_gt = False
        
        dcm_files = sorted([f for f in os.listdir(dcm_dir) if f.endswith('.dcm')])
        start_idx = (len(dcm_files) - self.n_slices) // 2
        
        ct_slices_original = []
        for i in range(start_idx, start_idx + self.n_slices):
            dcm_path = os.path.join(dcm_dir, dcm_files[i])
            ds = pydicom.dcmread(dcm_path)
            intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
            slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
            windowed = self._window_ct(ds.pixel_array, slope, intercept)
            ct_slices_original.append(windowed)
        
        ct_slices = [s.copy() for s in ct_slices_original]
        gt_mask = gt_mask_original.copy() if has_gt else None
        
        def apply_transform(transform_type):
            nonlocal ct_slices, gt_mask, pred_mask, pred_binary
            
            if transform_type == 'rotate':
                ct_slices = [np.rot90(s, 2) for s in ct_slices]
                if has_gt:
                    gt_mask = np.rot90(gt_mask, 2, axes=(0,1))
                print("🔄 ПОВОРОТ 180°")
            elif transform_type == 'shift':
                ct_slices = [np.roll(s, 50, axis=1) for s in ct_slices]
                if has_gt:
                    gt_mask = np.roll(gt_mask, 50, axis=1)
                print("➡️ СДВИГ ВПРАВО")
            elif transform_type == 'flip':
                ct_slices = [np.fliplr(s) for s in ct_slices]
                if has_gt:
                    gt_mask = np.fliplr(gt_mask)
                print("🪞 ЗЕРКАЛО")
            elif transform_type == 'reset':
                ct_slices = [s.copy() for s in ct_slices_original]
                gt_mask = gt_mask_original.copy() if has_gt else None
                print("🔄 СБРОС")
            
            ct_volume = np.stack(ct_slices, axis=0)
            pred_mask = self.predict_from_numpy(ct_volume)
            pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        ct_volume = np.stack(ct_slices, axis=0)
        pred_mask = self.predict_from_numpy(ct_volume)
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        fig, axes = plt.subplots(1, 3 if has_gt else 2, figsize=(15, 5))
        if not has_gt:
            axes = [axes[0], axes[1]]
        
        colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1)]
        current_idx = 0
        current_transform = "ОРИГИНАЛ"
        
        def update_display(idx):
            for ax in axes:
                ax.clear()
            
            # КТ
            axes[0].imshow(ct_slices[idx], cmap='gray')
            axes[0].set_title(f'КТ срез {idx+1}/6 [{current_transform}]')
            axes[0].axis('off')
            
            # Ground Truth — КРАСНЫЙ
            if has_gt and gt_mask is not None:
                gt_slice_idx = start_idx + idx
                if gt_slice_idx < gt_mask.shape[2]:
                    gt_slice = gt_mask[:, :, gt_slice_idx]
                    gt_overlay = np.zeros((*ct_slices[idx].shape, 3))
                    ct_norm = (ct_slices[idx] - ct_slices[idx].min()) / (ct_slices[idx].max() - ct_slices[idx].min() + 1e-8)
                    gt_overlay[:, :, 0] = ct_norm
                    gt_overlay[:, :, 1] = ct_norm
                    gt_overlay[:, :, 2] = ct_norm
                    
                    class_mask = (gt_slice > 0)
                    if class_mask.sum() > 0:
                        for ch in range(3):
                            gt_overlay[:, :, ch] = np.where(
                                class_mask,
                                gt_overlay[:, :, ch] * 0.5 + [1, 0, 0][ch] * 0.5,  # КРАСНЫЙ
                                gt_overlay[:, :, ch]
                            )
                    axes[1].imshow(gt_overlay)
                    axes[1].set_title(f'GT (врач)')
                axes[1].axis('off')
            
            # Предсказание — ПУРПУРНЫЙ/ФИОЛЕТОВЫЙ
            pred_ax = axes[2] if has_gt else axes[1]
            pred_overlay = np.zeros((*ct_slices[idx].shape, 3))
            ct_norm = (ct_slices[idx] - ct_slices[idx].min()) / (ct_slices[idx].max() - ct_slices[idx].min() + 1e-8)
            pred_overlay[:, :, 0] = ct_norm
            pred_overlay[:, :, 1] = ct_norm
            pred_overlay[:, :, 2] = ct_norm
            
            if pred_binary[idx].sum() > 0:
                for ch in range(3):
                    pred_overlay[:, :, ch] = np.where(
                        pred_binary[idx] > 0,
                        pred_overlay[:, :, ch] * 0.5 + [1, 0, 1][ch] * 0.5,  # ПУРПУРНЫЙ (R=1, G=0, B=1)
                        pred_overlay[:, :, ch]
                    )
            pred_ax.imshow(pred_overlay)
            pred_ax.set_title(f'Pred (модель)')
            pred_ax.axis('off')
            
            fig.suptitle(f'← → срезы | T:поворот | S:сдвиг | F:зеркало | R:сброс', fontsize=12)
            fig.canvas.draw()
        
        def on_key(event):
            nonlocal current_idx, current_transform
            if event.key == 'left':
                current_idx = max(0, current_idx - 1)
                update_display(current_idx)
            elif event.key == 'right':
                current_idx = min(5, current_idx + 1)
                update_display(current_idx)
            elif event.key == 't':
                current_transform = "ПОВОРОТ 180°"
                apply_transform('rotate')
                update_display(current_idx)
            elif event.key == 's':
                current_transform = "СДВИГ ВПРАВО"
                apply_transform('shift')
                update_display(current_idx)
            elif event.key == 'f':
                current_transform = "ЗЕРКАЛО"
                apply_transform('flip')
                update_display(current_idx)
            elif event.key == 'r':
                current_transform = "ОРИГИНАЛ"
                apply_transform('reset')
                update_display(current_idx)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        update_display(0)
        
        print("\n" + "="*60)
        print("🎮 ← → срезы | T:поворот | S:сдвиг | F:зеркало | R:сброс")
        print("="*60 + "\n")
        
        plt.show()

    @staticmethod
    def demo():
        import kagglehub
        import glob
        
        print("📥 Загрузка датасета с Kaggle...")
        dataset_path = kagglehub.dataset_download("andrewmvd/pulmonary-embolism-in-ct-images")
        
        all_patients = {}
        for root, dirs, files in os.walk(dataset_path):
            dcm_files = [f for f in files if f.endswith('.dcm')]
            if len(dcm_files) > 100:
                patient_id = os.path.basename(root)
                all_patients[patient_id] = root
        
        target = ['PAT033', 'PAT034', 'PAT035']
        selected = {pid: all_patients[pid] for pid in target if pid in all_patients}
        
        if not selected:
            print("❌ Пациенты 33, 34, 35 не найдены")
            return
        
        print(f"🤖 Загрузка модели 3D U-Net...")
        engine = EngineUNet3D()
        
        for patient_name, dcm_dir in selected.items():
            patient_id = os.path.basename(dcm_dir)
            
            mat_path = None
            search_paths = [
                os.path.join(dataset_path, f"{patient_id}.mat"),
                os.path.join(os.path.dirname(dcm_dir), f"{patient_id}.mat"),
                os.path.join(dataset_path, "GroundTruth", f"{patient_id}.mat"),
            ]
            
            for path in search_paths:
                if os.path.exists(path):
                    mat_path = path
                    break
            
            if mat_path is None:
                mat_files = glob.glob(os.path.join(dataset_path, "**", "*.mat"), recursive=True)
                for mf in mat_files:
                    if patient_id in mf:
                        mat_path = mf
                        break
            
            print("\n" + "="*60)
            print(f"📂 ПАЦИЕНТ: {patient_id}")
            print("="*60)
            print(f"📁 DICOM: {dcm_dir}")
            if mat_path:
                print(f"📋 Разметка: {mat_path}")
            else:
                print(f"⚠️ Разметка не найдена")
            
            print(f"\n🎮 УПРАВЛЕНИЕ:")
            print("   ← → срезы | T:поворот | S:сдвиг | F:зеркало | R:сброс")
            print("   Закройте окно для перехода к следующему\n")
            
            engine.visualize_sequence(dcm_dir=dcm_dir, mat_path=mat_path, threshold=0.5)
            
            if patient_name != list(selected.keys())[-1]:
                response = input(f"\n✅ {patient_id} просмотрен. Продолжить? (Y/n): ")
                if response.lower() == 'n':
                    print("👋 Демо завершено.")
                    return
        
        print("\n🎉 Все пациенты просмотрены!")
    def evaluate_invariance(self, dcm_dir: str, mat_path: str = None, threshold: float = 0.5):
        """
        Количественная оценка пространственной инвариантности модели.
        Возвращает точные цифры: пиксели, проценты, IoU.
        """
        import pydicom
        import scipy.io
        from scipy.ndimage import rotate, shift as scipy_shift
        
        print("\n" + "="*70)
        print("📊 КОЛИЧЕСТВЕННАЯ ОЦЕНКА ПРОСТРАНСТВЕННОЙ ИНВАРИАНТНОСТИ")
        print("="*70)
        
        # Загружаем данные
        dcm_files = sorted([f for f in os.listdir(dcm_dir) if f.endswith('.dcm')])
        start_idx = (len(dcm_files) - self.n_slices) // 2
        
        ct_slices = []
        for i in range(start_idx, start_idx + self.n_slices):
            dcm_path = os.path.join(dcm_dir, dcm_files[i])
            ds = pydicom.dcmread(dcm_path)
            intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
            slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
            windowed = self._window_ct(ds.pixel_array, slope, intercept)
            ct_slices.append(windowed)
        
        ct_original = np.stack(ct_slices, axis=0)
        
        # Базовое предсказание
        pred_original = self.predict_from_numpy(ct_original)
        binary_original = (pred_original > threshold).astype(np.uint8)
        total_pixels_original = binary_original.sum()
        
        print(f"\n📌 ОРИГИНАЛ: {total_pixels_original} активированных пикселей (100%)")
        
        # Словарь с результатами
        results = {}
        
        # ========== ТЕСТ 1: ПОВОРОТ 180° ==========
        ct_rotated = np.stack([np.rot90(s, 2) for s in ct_slices], axis=0)
        pred_rotated = self.predict_from_numpy(ct_rotated)
        binary_rotated = (pred_rotated > threshold).astype(np.uint8)
        
        # Поворачиваем предсказание обратно для сравнения
        pred_rotated_back = np.stack([np.rot90(pred_rotated[i], 2) for i in range(6)], axis=0)
        binary_rotated_back = (pred_rotated_back > threshold).astype(np.uint8)
        
        pixels_rotated = binary_rotated.sum()
        pixels_rotated_back = binary_rotated_back.sum()
        
        # IoU с оригиналом
        intersection = (binary_original & binary_rotated_back).sum()
        union = (binary_original | binary_rotated_back).sum()
        iou = intersection / union if union > 0 else 0
        
        results['rotate'] = {
            'name': 'Поворот 180°',
            'pixels_after': pixels_rotated,
            'pixels_after_back': pixels_rotated_back,
            'retention': (pixels_rotated_back / total_pixels_original * 100) if total_pixels_original > 0 else 0,
            'iou': iou * 100
        }
        
        # ========== ТЕСТ 2: СДВИГ (несколько значений) ==========
        shift_values = [10, 25, 50, 75, 100]
        results['shifts'] = []
        
        for shift_px in shift_values:
            ct_shifted = np.stack([np.roll(s, shift_px, axis=1) for s in ct_slices], axis=0)
            pred_shifted = self.predict_from_numpy(ct_shifted)
            binary_shifted = (pred_shifted > threshold).astype(np.uint8)
            
            # Сдвигаем обратно
            pred_shifted_back = np.stack([np.roll(pred_shifted[i], -shift_px, axis=1) for i in range(6)], axis=0)
            binary_shifted_back = (pred_shifted_back > threshold).astype(np.uint8)
            
            pixels_shifted = binary_shifted.sum()
            pixels_shifted_back = binary_shifted_back.sum()
            
            intersection = (binary_original & binary_shifted_back).sum()
            union = (binary_original | binary_shifted_back).sum()
            iou = intersection / union if union > 0 else 0
            
            results['shifts'].append({
                'pixels': shift_px,
                'pixels_after': pixels_shifted,
                'pixels_after_back': pixels_shifted_back,
                'retention': (pixels_shifted_back / total_pixels_original * 100) if total_pixels_original > 0 else 0,
                'iou': iou * 100
            })
        
        # ========== ТЕСТ 3: ЗЕРКАЛО ==========
        ct_flipped = np.stack([np.fliplr(s) for s in ct_slices], axis=0)
        pred_flipped = self.predict_from_numpy(ct_flipped)
        binary_flipped = (pred_flipped > threshold).astype(np.uint8)
        
        pred_flipped_back = np.stack([np.fliplr(pred_flipped[i]) for i in range(6)], axis=0)
        binary_flipped_back = (pred_flipped_back > threshold).astype(np.uint8)
        
        pixels_flipped = binary_flipped.sum()
        pixels_flipped_back = binary_flipped_back.sum()
        
        intersection = (binary_original & binary_flipped_back).sum()
        union = (binary_original | binary_flipped_back).sum()
        iou = intersection / union if union > 0 else 0
        
        results['flip'] = {
            'name': 'Зеркало',
            'pixels_after': pixels_flipped,
            'pixels_after_back': pixels_flipped_back,
            'retention': (pixels_flipped_back / total_pixels_original * 100) if total_pixels_original > 0 else 0,
            'iou': iou * 100
        }
        
        # ========== ВЫВОД РЕЗУЛЬТАТОВ ==========
        print("\n" + "-"*70)
        print("📈 РЕЗУЛЬТАТЫ ТЕСТОВ:")
        print("-"*70)
        
        r = results['rotate']
        print(f"\n🔄 ПОВОРОТ 180°:")
        print(f"   Активаций: {r['pixels_after']} px")
        print(f"   После обратного поворота: {r['pixels_after_back']} px")
        print(f"   Сохранение: {r['retention']:.1f}%")
        print(f"   IoU с оригиналом: {r['iou']:.1f}%")
        
        print(f"\n➡️ СДВИГ ВПРАВО:")
        print(f"   {'Сдвиг':<8} {'Активаций':<12} {'После возврата':<14} {'Сохранение':<12} {'IoU':<8}")
        print(f"   {'-'*60}")
        for s in results['shifts']:
            print(f"   {s['pixels']:3d} px   {s['pixels_after']:5d} px     {s['pixels_after_back']:5d} px        {s['retention']:5.1f}%      {s['iou']:5.1f}%")
        
        r = results['flip']
        print(f"\n🪞 ЗЕРКАЛО:")
        print(f"   Активаций: {r['pixels_after']} px")
        print(f"   После обратного отражения: {r['pixels_after_back']} px")
        print(f"   Сохранение: {r['retention']:.1f}%")
        print(f"   IoU с оригиналом: {r['iou']:.1f}%")
        
        # ========== ИТОГОВАЯ ТАБЛИЦА ДЛЯ СТАТЬИ ==========
        print("\n" + "="*70)
        print("📋 СВОДНАЯ ТАБЛИЦА ДЛЯ ПУБЛИКАЦИИ:")
        print("="*70)
        print(f"""
        Модель: 3D U-Net (VGG)
        Оригинальных активаций: {total_pixels_original} px
        
        ┌─────────────────┬──────────────┬──────────────┬─────────────┐
        │ Трансформация   │ Сохранение   │ IoU          │ Статус      │
        ├─────────────────┼──────────────┼──────────────┼─────────────┤
        │ Поворот 180°    │ {results['rotate']['retention']:5.1f}%       │ {results['rotate']['iou']:5.1f}% 
        │ Сдвиг 10px      │ {results['shifts'][0]['retention']:5.1f}%       │ {results['shifts'][0]['iou']:5.1f}%
        │ Сдвиг 25px      │ {results['shifts'][1]['retention']:5.1f}%       │ {results['shifts'][1]['iou']:5.1f}%
        │ Сдвиг 50px      │ {results['shifts'][2]['retention']:5.1f}%       │ {results['shifts'][2]['iou']:5.1f}%
        │ Сдвиг 75px      │ {results['shifts'][3]['retention']:5.1f}%       │ {results['shifts'][3]['iou']:5.1f}%
        │ Сдвиг 100px     │ {results['shifts'][4]['retention']:5.1f}%       │ {results['shifts'][4]['iou']:5.1f}%
        │ Зеркало         │ {results['flip']['retention']:5.1f}%       │ {results['flip']['iou']:5.1f}% 
        └─────────────────┴──────────────┴──────────────┴─────────────┘
        """)
        
        # Для 2D моделей — отдельная строка
        print("\n📋 ДЛЯ СРАВНЕНИЯ (2D модели):")
        
        return results

#потом поправлю
if __name__ == "__main__":
    EngineUNet3D.demo()
    engine = EngineUNet3D()
    results = engine.evaluate_invariance(
        dcm_dir="/home/chelovek/Рабочий стол/PATFully/PAT034",
        mat_path="/home/chelovek/Рабочий стол/PATFully/PAT034.mat",
        threshold=0.5
    )
#----------------------------------------------------------------------------------
#3D Attention U-Net с VGG-подобным энкодером.

class EngineAttentionUNet3D:
    """
    Движок для 3D Attention U-Net сегментации.
    1 входной канал (объём 6×H×W), 1 выходной канал (маска 6×H×W).
    """
    
    # ========================================================================
    # ВЛОЖЕННЫЕ КЛАССЫ АРХИТЕКТУРЫ
    # ========================================================================
    
    class DiceLoss(nn.Module):
        def __init__(self, smooth=1e-6):
            super().__init__()
            self.smooth = smooth

        def forward(self, predict, target):
            predict = torch.sigmoid(predict)
            predict = predict.view(-1)
            target = target.view(-1)
            intersection = (predict * target).sum()
            dice = (2.*intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth)
            return 1 - dice

    class CombinedLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()
            self.dice = EngineAttentionUNet3D.DiceLoss()

        def forward(self, pred, target):
            return self.bce(pred, target) + self.dice(pred, target)

    class AttentionBlock3D(nn.Module):
        """3D версия Attention Gate с автоматическим согласованием размеров"""
        def __init__(self, F_g, F_l, F_int):
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm3d(F_int)
            )
            self.W_x = nn.Sequential(
                nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm3d(F_int)
            )
            self.psi = nn.Sequential(
                nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm3d(1),
                nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            if g.shape[2:] != x.shape[2:]:
                g = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=False)
            
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi

    class VGG16_UNet_3D(nn.Module):
        def __init__(self, in_channels=1, out_channels=1):
            super().__init__()
            
            AttentionBlock3D = EngineAttentionUNet3D.AttentionBlock3D
            
            # ЭНКОДЕР
            self.conv1_1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
            self.conv1_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

            self.conv2_1 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
            self.conv2_2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

            self.conv3_1 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
            self.conv3_2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
            self.conv3_3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
            self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv4_1 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
            self.conv4_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
            self.conv4_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
            self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv5_1 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
            self.conv5_2 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
            self.conv5_3 = nn.Conv3d(512, 512, kernel_size=3, padding=1)

            # ATTENTION GATES
            self.att4 = AttentionBlock3D(F_g=256, F_l=512, F_int=256)
            self.att3 = AttentionBlock3D(F_g=128, F_l=256, F_int=128)
            self.att2 = AttentionBlock3D(F_g=64, F_l=128, F_int=64)
            self.att1 = AttentionBlock3D(F_g=64, F_l=64, F_int=32)

            # ДЕКОДЕР
            self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.dec4_1 = nn.Conv3d(768, 256, kernel_size=3, padding=1)
            self.dec4_2 = nn.Conv3d(256, 256, kernel_size=3, padding=1)

            self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            self.dec3_1 = nn.Conv3d(384, 128, kernel_size=3, padding=1)
            self.dec3_2 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

            self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 2, 2), stride=(3, 2, 2))
            self.dec2_1 = nn.Conv3d(192, 64, kernel_size=3, padding=1)
            self.dec2_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

            self.upconv1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
            self.dec1_1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
            self.dec1_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

            self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.relu(self.conv1_1(x))
            x = self.relu(self.conv1_2(x))
            skip1 = x
            x = self.pool1(x)

            x = self.relu(self.conv2_1(x))
            x = self.relu(self.conv2_2(x))
            skip2 = x
            x = self.pool2(x)

            x = self.relu(self.conv3_1(x))
            x = self.relu(self.conv3_2(x))
            x = self.relu(self.conv3_3(x))
            skip3 = x
            x = self.pool3(x)

            x = self.relu(self.conv4_1(x))
            x = self.relu(self.conv4_2(x))
            x = self.relu(self.conv4_3(x))
            skip4 = x
            x = self.pool4(x)

            x = self.relu(self.conv5_1(x))
            x = self.relu(self.conv5_2(x))
            x = self.relu(self.conv5_3(x))

            x = self.upconv4(x)
            skip4 = self.att4(g=x, x=skip4)
            x = torch.cat([x, skip4], dim=1)
            x = self.relu(self.dec4_1(x))
            x = self.relu(self.dec4_2(x))

            x = self.upconv3(x)
            skip3 = self.att3(g=x, x=skip3)
            x = torch.cat([x, skip3], dim=1)
            x = self.relu(self.dec3_1(x))
            x = self.relu(self.dec3_2(x))

            x = self.upconv2(x)
            skip2 = self.att2(g=x, x=skip2)
            x = torch.cat([x, skip2], dim=1)
            x = self.relu(self.dec2_1(x))
            x = self.relu(self.dec2_2(x))

            x = self.upconv1(x)
            skip1 = self.att1(g=x, x=skip1)
            x = torch.cat([x, skip1], dim=1)
            x = self.relu(self.dec1_1(x))
            x = self.relu(self.dec1_2(x))

            out = self.final_conv(x)
            return out.squeeze(1)

    # ========================================================================
    # ОСНОВНОЙ КЛАСС ENGINE
    # ========================================================================
    
    def __init__(
        self, 
        repo_id: str = "MarkProMaster229/experimental_models",
        config_path: str = "VGG16_UNet_3DAttentionEmbol/config.json",
        weights_path: str = "VGG16_UNet_3DAttentionEmbol/cnn_model_epoch_7.pth",
        n_slices: int = 6
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_slices = n_slices
        
        print(f"📥 Загрузка конфига: {repo_id}/{config_path}")
        config_file = hf_hub_download(repo_id=repo_id, filename=config_path)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.model = self.VGG16_UNet_3D(
            in_channels=self.config.get("in_channels", 1),
            out_channels=self.config.get("out_channels", 1)
        ).to(self.device)
        
        print(f"📥 Загрузка весов: {repo_id}/{weights_path}")
        weights_file = hf_hub_download(repo_id=repo_id, filename=weights_path)
        state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        print(f"✅ Engine (Attention 3D U-Net) загружен")
        print(f"   Устройство: {self.device}")
    
    def _window_ct(self, pixel_array, slope=1, intercept=0):
        image = pixel_array.astype(np.float32) * slope + intercept
        img = np.clip(image, -250, 450)
        return (img - (-250)) / 700
    
    def predict_from_numpy(self, ct_volume: np.ndarray):
        if ct_volume.shape[0] != self.n_slices:
            raise ValueError(f"Ожидалось {self.n_slices} срезов, получено {ct_volume.shape[0]}")
        
        input_tensor = torch.tensor(ct_volume, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.sigmoid(output)
        
        mask = output.squeeze(0).cpu().numpy()
        return mask
    
    def visualize_sequence(self, dcm_dir: str, mat_path: str = None, threshold: float = 0.5):
        import pydicom
        import scipy.io
        
        if mat_path and os.path.exists(mat_path):
            gt_mask_original = scipy.io.loadmat(mat_path)['Mask']
            has_gt = True
        else:
            gt_mask_original = None
            has_gt = False
        
        dcm_files = sorted([f for f in os.listdir(dcm_dir) if f.endswith('.dcm')])
        start_idx = (len(dcm_files) - self.n_slices) // 2
        
        ct_slices_original = []
        for i in range(start_idx, start_idx + self.n_slices):
            dcm_path = os.path.join(dcm_dir, dcm_files[i])
            ds = pydicom.dcmread(dcm_path)
            intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
            slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
            windowed = self._window_ct(ds.pixel_array, slope, intercept)
            ct_slices_original.append(windowed)
        
        ct_slices = [s.copy() for s in ct_slices_original]
        gt_mask = gt_mask_original.copy() if has_gt else None
        
        def apply_transform(transform_type):
            nonlocal ct_slices, gt_mask, pred_mask, pred_binary
            
            if transform_type == 'rotate':
                ct_slices = [np.rot90(s, 2) for s in ct_slices]
                if has_gt:
                    gt_mask = np.rot90(gt_mask, 2, axes=(0,1))
                print("🔄 ПОВОРОТ 180°")
            elif transform_type == 'shift':
                ct_slices = [np.roll(s, 50, axis=1) for s in ct_slices]
                if has_gt:
                    gt_mask = np.roll(gt_mask, 50, axis=1)
                print("➡️ СДВИГ ВПРАВО")
            elif transform_type == 'flip':
                ct_slices = [np.fliplr(s) for s in ct_slices]
                if has_gt:
                    gt_mask = np.fliplr(gt_mask)
                print("🪞 ЗЕРКАЛО")
            elif transform_type == 'reset':
                ct_slices = [s.copy() for s in ct_slices_original]
                gt_mask = gt_mask_original.copy() if has_gt else None
                print("🔄 СБРОС")
            
            ct_volume = np.stack(ct_slices, axis=0)
            pred_mask = self.predict_from_numpy(ct_volume)
            pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        ct_volume = np.stack(ct_slices, axis=0)
        pred_mask = self.predict_from_numpy(ct_volume)
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        fig, axes = plt.subplots(1, 3 if has_gt else 2, figsize=(15, 5))
        if not has_gt:
            axes = [axes[0], axes[1]]
        
        current_idx = 0
        current_transform = "ОРИГИНАЛ"
        
        def update_display(idx):
            for ax in axes:
                ax.clear()
            
            axes[0].imshow(ct_slices[idx], cmap='gray')
            axes[0].set_title(f'КТ срез {idx+1}/6 [{current_transform}]')
            axes[0].axis('off')
            
            if has_gt and gt_mask is not None:
                gt_slice_idx = start_idx + idx
                if gt_slice_idx < gt_mask.shape[2]:
                    gt_slice = gt_mask[:, :, gt_slice_idx]
                    gt_overlay = np.zeros((*ct_slices[idx].shape, 3))
                    ct_norm = (ct_slices[idx] - ct_slices[idx].min()) / (ct_slices[idx].max() - ct_slices[idx].min() + 1e-8)
                    gt_overlay[:, :, 0] = ct_norm
                    gt_overlay[:, :, 1] = ct_norm
                    gt_overlay[:, :, 2] = ct_norm
                    
                    class_mask = (gt_slice > 0)
                    if class_mask.sum() > 0:
                        for ch in range(3):
                            gt_overlay[:, :, ch] = np.where(
                                class_mask,
                                gt_overlay[:, :, ch] * 0.5 + [1, 0, 0][ch] * 0.5,  # КРАСНЫЙ
                                gt_overlay[:, :, ch]
                            )
                    axes[1].imshow(gt_overlay)
                    axes[1].set_title(f'GT (врач)')
                axes[1].axis('off')
            
            pred_ax = axes[2] if has_gt else axes[1]
            pred_overlay = np.zeros((*ct_slices[idx].shape, 3))
            ct_norm = (ct_slices[idx] - ct_slices[idx].min()) / (ct_slices[idx].max() - ct_slices[idx].min() + 1e-8)
            pred_overlay[:, :, 0] = ct_norm
            pred_overlay[:, :, 1] = ct_norm
            pred_overlay[:, :, 2] = ct_norm
            
            if pred_binary[idx].sum() > 0:
                for ch in range(3):
                    pred_overlay[:, :, ch] = np.where(
                        pred_binary[idx] > 0,
                        pred_overlay[:, :, ch] * 0.5 + [1, 0, 1][ch] * 0.5,  # ПУРПУРНЫЙ
                        pred_overlay[:, :, ch]
                    )
            pred_ax.imshow(pred_overlay)
            pred_ax.set_title(f'Pred (модель)')
            pred_ax.axis('off')
            
            fig.suptitle(f'← → срезы | T:поворот | S:сдвиг | F:зеркало | R:сброс', fontsize=12)
            fig.canvas.draw()
        
        def on_key(event):
            nonlocal current_idx, current_transform
            if event.key == 'left':
                current_idx = max(0, current_idx - 1)
                update_display(current_idx)
            elif event.key == 'right':
                current_idx = min(5, current_idx + 1)
                update_display(current_idx)
            elif event.key == 't':
                current_transform = "ПОВОРОТ 180°"
                apply_transform('rotate')
                update_display(current_idx)
            elif event.key == 's':
                current_transform = "СДВИГ ВПРАВО"
                apply_transform('shift')
                update_display(current_idx)
            elif event.key == 'f':
                current_transform = "ЗЕРКАЛО"
                apply_transform('flip')
                update_display(current_idx)
            elif event.key == 'r':
                current_transform = "ОРИГИНАЛ"
                apply_transform('reset')
                update_display(current_idx)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        update_display(0)
        
        print("\n" + "="*60)
        print("🎮 ← → срезы | T:поворот | S:сдвиг | F:зеркало | R:сброс")
        print("="*60 + "\n")
        
        plt.show()

    @staticmethod
    def demo():
        import kagglehub
        import glob
        
        print("📥 Загрузка датасета с Kaggle...")
        dataset_path = kagglehub.dataset_download("andrewmvd/pulmonary-embolism-in-ct-images")
        
        all_patients = {}
        for root, dirs, files in os.walk(dataset_path):
            dcm_files = [f for f in files if f.endswith('.dcm')]
            if len(dcm_files) > 100:
                patient_id = os.path.basename(root)
                all_patients[patient_id] = root
        
        target = ['PAT033', 'PAT034', 'PAT035']
        selected = {pid: all_patients[pid] for pid in target if pid in all_patients}
        
        if not selected:
            print("❌ Пациенты 33, 34, 35 не найдены")
            return
        
        print(f"🤖 Загрузка модели Attention 3D U-Net...")
        engine = EngineAttentionUNet3D()
        
        for patient_name, dcm_dir in selected.items():
            patient_id = os.path.basename(dcm_dir)
            
            mat_path = None
            search_paths = [
                os.path.join(dataset_path, f"{patient_id}.mat"),
                os.path.join(os.path.dirname(dcm_dir), f"{patient_id}.mat"),
                os.path.join(dataset_path, "GroundTruth", f"{patient_id}.mat"),
            ]
            
            for path in search_paths:
                if os.path.exists(path):
                    mat_path = path
                    break
            
            if mat_path is None:
                mat_files = glob.glob(os.path.join(dataset_path, "**", "*.mat"), recursive=True)
                for mf in mat_files:
                    if patient_id in mf:
                        mat_path = mf
                        break
            
            print("\n" + "="*60)
            print(f"📂 ПАЦИЕНТ: {patient_id}")
            print("="*60)
            print(f"📁 DICOM: {dcm_dir}")
            if mat_path:
                print(f"📋 Разметка: {mat_path}")
            else:
                print(f"⚠️ Разметка не найдена")
            
            print(f"\n🎮 УПРАВЛЕНИЕ:")
            print("   ← → срезы | T:поворот | S:сдвиг | F:зеркало | R:сброс")
            print("   Закройте окно для перехода к следующему\n")
            
            engine.visualize_sequence(dcm_dir=dcm_dir, mat_path=mat_path, threshold=0.5)
            
            if patient_name != list(selected.keys())[-1]:
                response = input(f"\n✅ {patient_id} просмотрен. Продолжить? (Y/n): ")
                if response.lower() == 'n':
                    print("👋 Демо завершено.")
                    return
        
        print("\n🎉 Все пациенты просмотрены!")
    def evaluate_invariance2(self, dcm_dir: str, mat_path: str = None, threshold: float = 0.5):
            """
            Количественная оценка пространственной инвариантности модели.
            Возвращает точные цифры: пиксели, проценты, IoU.
            """
            import pydicom
            import scipy.io
            from scipy.ndimage import rotate, shift as scipy_shift
            
            print("\n" + "="*70)
            print("📊 КОЛИЧЕСТВЕННАЯ ОЦЕНКА ПРОСТРАНСТВЕННОЙ ИНВАРИАНТНОСТИ")
            print("="*70)
            
            # Загружаем данные
            dcm_files = sorted([f for f in os.listdir(dcm_dir) if f.endswith('.dcm')])
            start_idx = (len(dcm_files) - self.n_slices) // 2
            
            ct_slices = []
            for i in range(start_idx, start_idx + self.n_slices):
                dcm_path = os.path.join(dcm_dir, dcm_files[i])
                ds = pydicom.dcmread(dcm_path)
                intercept = ds.RescaleIntercept if 'RescaleIntercept' in ds else 0
                slope = ds.RescaleSlope if 'RescaleSlope' in ds else 1
                windowed = self._window_ct(ds.pixel_array, slope, intercept)
                ct_slices.append(windowed)
            
            ct_original = np.stack(ct_slices, axis=0)
            
            # Базовое предсказание
            pred_original = self.predict_from_numpy(ct_original)
            binary_original = (pred_original > threshold).astype(np.uint8)
            total_pixels_original = binary_original.sum()
            
            print(f"\n📌 ОРИГИНАЛ: {total_pixels_original} активированных пикселей (100%)")
            
            # Словарь с результатами
            results = {}
            
            # ========== ТЕСТ 1: ПОВОРОТ 180° ==========
            ct_rotated = np.stack([np.rot90(s, 2) for s in ct_slices], axis=0)
            pred_rotated = self.predict_from_numpy(ct_rotated)
            binary_rotated = (pred_rotated > threshold).astype(np.uint8)
            
            # Поворачиваем предсказание обратно для сравнения
            pred_rotated_back = np.stack([np.rot90(pred_rotated[i], 2) for i in range(6)], axis=0)
            binary_rotated_back = (pred_rotated_back > threshold).astype(np.uint8)
            
            pixels_rotated = binary_rotated.sum()
            pixels_rotated_back = binary_rotated_back.sum()
            
            # IoU с оригиналом
            intersection = (binary_original & binary_rotated_back).sum()
            union = (binary_original | binary_rotated_back).sum()
            iou = intersection / union if union > 0 else 0
            
            results['rotate'] = {
                'name': 'Поворот 180°',
                'pixels_after': pixels_rotated,
                'pixels_after_back': pixels_rotated_back,
                'retention': (pixels_rotated_back / total_pixels_original * 100) if total_pixels_original > 0 else 0,
                'iou': iou * 100
            }
            
            # ========== ТЕСТ 2: СДВИГ (несколько значений) ==========
            shift_values = [10, 25, 50, 75, 100]
            results['shifts'] = []
            
            for shift_px in shift_values:
                ct_shifted = np.stack([np.roll(s, shift_px, axis=1) for s in ct_slices], axis=0)
                pred_shifted = self.predict_from_numpy(ct_shifted)
                binary_shifted = (pred_shifted > threshold).astype(np.uint8)
                
                # Сдвигаем обратно
                pred_shifted_back = np.stack([np.roll(pred_shifted[i], -shift_px, axis=1) for i in range(6)], axis=0)
                binary_shifted_back = (pred_shifted_back > threshold).astype(np.uint8)
                
                pixels_shifted = binary_shifted.sum()
                pixels_shifted_back = binary_shifted_back.sum()
                
                intersection = (binary_original & binary_shifted_back).sum()
                union = (binary_original | binary_shifted_back).sum()
                iou = intersection / union if union > 0 else 0
                
                results['shifts'].append({
                    'pixels': shift_px,
                    'pixels_after': pixels_shifted,
                    'pixels_after_back': pixels_shifted_back,
                    'retention': (pixels_shifted_back / total_pixels_original * 100) if total_pixels_original > 0 else 0,
                    'iou': iou * 100
                })
            
            # ========== ТЕСТ 3: ЗЕРКАЛО ==========
            ct_flipped = np.stack([np.fliplr(s) for s in ct_slices], axis=0)
            pred_flipped = self.predict_from_numpy(ct_flipped)
            binary_flipped = (pred_flipped > threshold).astype(np.uint8)
            
            pred_flipped_back = np.stack([np.fliplr(pred_flipped[i]) for i in range(6)], axis=0)
            binary_flipped_back = (pred_flipped_back > threshold).astype(np.uint8)
            
            pixels_flipped = binary_flipped.sum()
            pixels_flipped_back = binary_flipped_back.sum()
            
            intersection = (binary_original & binary_flipped_back).sum()
            union = (binary_original | binary_flipped_back).sum()
            iou = intersection / union if union > 0 else 0
            
            results['flip'] = {
                'name': 'Зеркало',
                'pixels_after': pixels_flipped,
                'pixels_after_back': pixels_flipped_back,
                'retention': (pixels_flipped_back / total_pixels_original * 100) if total_pixels_original > 0 else 0,
                'iou': iou * 100
            }
            
            # ========== ВЫВОД РЕЗУЛЬТАТОВ ==========
            print("\n" + "-"*70)
            print("📈 РЕЗУЛЬТАТЫ ТЕСТОВ:")
            print("-"*70)
            
            r = results['rotate']
            print(f"\n🔄 ПОВОРОТ 180°:")
            print(f"   Активаций: {r['pixels_after']} px")
            print(f"   После обратного поворота: {r['pixels_after_back']} px")
            print(f"   Сохранение: {r['retention']:.1f}%")
            print(f"   IoU с оригиналом: {r['iou']:.1f}%")
            
            print(f"\n➡️ СДВИГ ВПРАВО:")
            print(f"   {'Сдвиг':<8} {'Активаций':<12} {'После возврата':<14} {'Сохранение':<12} {'IoU':<8}")
            print(f"   {'-'*60}")
            for s in results['shifts']:
                print(f"   {s['pixels']:3d} px   {s['pixels_after']:5d} px     {s['pixels_after_back']:5d} px        {s['retention']:5.1f}%      {s['iou']:5.1f}%")
            
            r = results['flip']
            print(f"\n🪞 ЗЕРКАЛО:")
            print(f"   Активаций: {r['pixels_after']} px")
            print(f"   После обратного отражения: {r['pixels_after_back']} px")
            print(f"   Сохранение: {r['retention']:.1f}%")
            print(f"   IoU с оригиналом: {r['iou']:.1f}%")
            
            # ========== ИТОГОВАЯ ТАБЛИЦА ДЛЯ СТАТЬИ ==========
            print("\n" + "="*70)
            print("📋 СВОДНАЯ ТАБЛИЦА ДЛЯ ПУБЛИКАЦИИ:")
            print("="*70)
            print(f"""
            Модель: 3D U-Net (VGG)
            Оригинальных активаций: {total_pixels_original} px
            
            ┌─────────────────┬──────────────┬──────────────┬─────────────┐
            │ Трансформация   │ Сохранение   │ IoU          │ Статус      │
            ├─────────────────┼──────────────┼──────────────┼─────────────┤
            │ Поворот 180°    │ {results['rotate']['retention']:5.1f}%       │ {results['rotate']['iou']:5.1f}%
            │ Сдвиг 10px      │ {results['shifts'][0]['retention']:5.1f}%       │ {results['shifts'][0]['iou']:5.1f}%
            │ Сдвиг 25px      │ {results['shifts'][1]['retention']:5.1f}%       │ {results['shifts'][1]['iou']:5.1f}%
            │ Сдвиг 50px      │ {results['shifts'][2]['retention']:5.1f}%       │ {results['shifts'][2]['iou']:5.1f}%
            │ Сдвиг 75px      │ {results['shifts'][3]['retention']:5.1f}%       │ {results['shifts'][3]['iou']:5.1f}%
            │ Сдвиг 100px     │ {results['shifts'][4]['retention']:5.1f}%       │ {results['shifts'][4]['iou']:5.1f}% 
            │ Зеркало         │ {results['flip']['retention']:5.1f}%       │ {results['flip']['iou']:5.1f}%
            └─────────────────┴──────────────┴──────────────┴─────────────┘
            """)
            
            # Для 2D моделей — отдельная строка
            print("\n📋 ДЛЯ СРАВНЕНИЯ (2D модели):")
            
            return results


if __name__ == "__main__":
    EngineAttentionUNet3D.demo()
    AttentionUNet3D = EngineAttentionUNet3D()
    results = AttentionUNet3D.evaluate_invariance2(
        dcm_dir="/home/chelovek/Рабочий стол/PATFully/PAT034",
        mat_path="/home/chelovek/Рабочий стол/PATFully/PAT034.mat",
        threshold=0.5
    )

class Manager:
    def MyCollector(self, model):
        if hasattr(model, 'model'):
            model.model.cpu()
            del model.model
        if hasattr(model, 'tokenizer'):
            del model.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        del model
        gc.collect()

        gc.collect()

    #this Load-Use-Unload
    
    def ThisController(self, promt, MyMagicObject):
        # my magic logic
        # archAutoRegr
        if MyMagicObject == 1:
            model = Engine()
            result = model.generate(promt)
            self.MyCollector(model)
            return result
        #RoBert-xlm
        elif MyMagicObject == 2:
            model = EngineRoBert()
            result = model.predict(promt)
            self.MyCollector(model)
            return result

test3 = EngineRoBert()
test3.predict("настроение ")

#this test 
app = Flask(__name__)
CORS(app)

engine1 = Engine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.get_json()

    print("RAW DATA:", data)

    prompt = data.get("prompt", "")

    print("PROMPT:", prompt)

    result = engine1.generate(prompt)

    return jsonify({
        "response": result
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)