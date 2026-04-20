# se_resnet_classifier_engine.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import os
from huggingface_hub import hf_hub_download


class EngineSEResNetClassifier:
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

    class SEBlock(nn.Module):
        def __init__(self, channels, reduction_ratio=16):
            super().__init__()
            reduced_channels = max(1, channels // reduction_ratio)
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, reduced_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduced_channels, channels, kernel_size=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            scale = self.fc(x)
            return x * scale

    class SEBasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=16):
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

            self.se = EngineSEResNetClassifier.SEBlock(out_channels, reduction_ratio)

        def forward(self, x):
            identity = self.shortcut(x)
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
            out += identity
            out = self.se(out)
            out = F.relu(out, inplace=True)
            return out

    class SE_ResNet18(nn.Module):
        def __init__(self, num_classes=64, in_channels=1, reduction_ratio=16):
            super().__init__()
            
            SEBasicBlock = EngineSEResNetClassifier.SEBasicBlock
            
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, 
                                   padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(64, 64, blocks=2, stride=1, 
                                           reduction_ratio=reduction_ratio, 
                                           SEBasicBlock=SEBasicBlock)
            self.layer2 = self._make_layer(64, 128, blocks=2, stride=2, 
                                           reduction_ratio=reduction_ratio,
                                           SEBasicBlock=SEBasicBlock)
            self.layer3 = self._make_layer(128, 256, blocks=2, stride=2, 
                                           reduction_ratio=reduction_ratio,
                                           SEBasicBlock=SEBasicBlock)
            self.layer4 = self._make_layer(256, 512, blocks=2, stride=2, 
                                           reduction_ratio=reduction_ratio,
                                           SEBasicBlock=SEBasicBlock)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

            self._init_weights()

        def _make_layer(self, in_channels, out_channels, blocks, stride, 
                        reduction_ratio, SEBasicBlock):
            layers = []
            layers.append(SEBasicBlock(in_channels, out_channels, stride, reduction_ratio))
            for _ in range(1, blocks):
                layers.append(SEBasicBlock(out_channels, out_channels, stride=1, 
                                           reduction_ratio=reduction_ratio))
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
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    # ========================================================================
    # ОСНОВНОЙ КЛАСС ENGINE
    # ========================================================================
    
    def __init__(
        self,
        repo_id: str = "MarkProMaster229/experimental_models",
        config_path: str = "SeNetLetterAttentionModel/config.json",
        weights_path: str = "SeNetLetterAttentionModel/cnn_model_epoch_13.pth"
    ):
        self.device = torch.device("cpu")
        
        # 1. Скачиваем конфиг
        print(f"📥 Загрузка конфига: {repo_id}/{config_path}")
        config_file = hf_hub_download(repo_id=repo_id, filename=config_path)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 2. Создаём модель
        self.model = self.SE_ResNet18(
            num_classes=self.config.get('num_classes', 64),
            in_channels=self.config.get('input_channels', 1),
            reduction_ratio=self.config.get('reduction_ratio', 16)
        ).to(self.device)
        
        # 3. Грузим веса
        print(f"📥 Загрузка весов: {repo_id}/{weights_path}")
        weights_file = hf_hub_download(repo_id=repo_id, filename=weights_path)
        state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        
        # 4. Трансформации
        self.transform = transforms.Compose([
            transforms.Resize((self.config.get('img_size', 224), 
                              self.config.get('img_size', 224))),
            transforms.Grayscale(num_output_channels=self.config.get('input_channels', 1)),
            transforms.ToTensor(),
            transforms.Normalize(
                self.config.get('normalize', {}).get('mean', [0.5]),
                self.config.get('normalize', {}).get('std', [0.5])
            )
        ])
        
        # 5. Маппинг меток
        self.id2label = {}
        for k, v in self.config.get('id2label', {}).items():
            self.id2label[int(k)] = v
        
        # Если id2label пустой, используем стандартный алфавит
        if not self.id2label:
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:#'\"!?/()"
            self.id2label = {i: c for i, c in enumerate(alphabet)}
        
        self.model.eval()
        print(f"✅ Engine (SE-ResNet18) загружен")
        print(f"   Модель: {self.config.get('model_type', 'SE-ResNet18')}")
        print(f"   Классов: {self.config.get('num_classes', 64)}")
        print(f"   Устройство: {self.device}")
    
    def predict(self, image_input):
        """
        Предсказание класса буквы для одного изображения.
        
        Args:
            image_input: путь к файлу (str) или PIL.Image
            
        Returns:
            tuple: (буква, уверенность)
        """
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
        
        pred_class = pred_class.item()
        confidence = confidence.item()
        
        label = self.id2label.get(pred_class, f"class_{pred_class}")
        
        return label, confidence
    
    def predict_top_k(self, image_input, k: int = 3):
        """
        Возвращает топ-k предсказаний.
        """
        self.model.eval()
        
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        probs = F.softmax(outputs, dim=1)
        top_probs, top_classes = torch.topk(probs, k, dim=1)
        
        results = []
        for i in range(k):
            cls = top_classes[0, i].item()
            prob = top_probs[0, i].item()
            label = self.id2label.get(cls, f"class_{cls}")
            results.append((label, prob))
        
        return results
