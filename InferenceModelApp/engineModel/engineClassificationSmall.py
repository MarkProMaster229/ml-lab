import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download
import json
import os


class EngineTransformerClassifier:
    """
    Движок для Transformer-классификатора текста.
    3 класса: positive, negative, neutral
    """
    
    # ========================================================================
    # ВЛОЖЕННЫЕ КЛАССЫ АРХИТЕКТУРЫ
    # ========================================================================
    
    class TransformerBlock(nn.Module):
        def __init__(self, sizeVector=256, numHeads=8, dropout=0.5):
            super().__init__()
            self.ln1 = nn.LayerNorm(sizeVector)
            self.attn = nn.MultiheadAttention(sizeVector, numHeads, batch_first=True)
            self.dropout_attn = nn.Dropout(dropout)
            self.ln2 = nn.LayerNorm(sizeVector)
            self.ff = nn.Sequential(
                nn.Linear(sizeVector, sizeVector*4),
                nn.GELU(),
                nn.Linear(sizeVector*4, sizeVector),
                nn.Dropout(dropout)
            )
        
        def forward(self, x, attention_mask=None):
            key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
            h = self.ln1(x)
            attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
            x = x + self.dropout_attn(attn_out)
            x = x + self.ff(self.ln2(x))
            return x

    class TransformerClassifier(nn.Module):
        def __init__(self, vocabSize=120000, maxLen=100, sizeVector=256, 
                     numBlocks=4, numHeads=8, numClasses=3, dropout=0.5):
            super().__init__()
            self.token_emb = nn.Embedding(vocabSize, sizeVector)
            self.pos_emb = nn.Embedding(maxLen, sizeVector)
            self.layers = nn.ModuleList([
                EngineTransformerClassifier.TransformerBlock(
                    sizeVector=sizeVector, numHeads=numHeads, dropout=dropout
                )
                for _ in range(numBlocks)
            ])
            self.dropout = nn.Dropout(dropout)
            self.ln = nn.LayerNorm(sizeVector*2)
            self.classifier = nn.Linear(sizeVector*2, numClasses)

        def forward(self, x, attention_mask=None):
            B, T = x.shape
            tok = self.token_emb(x)
            pos = self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0).expand(B, T))
            h = tok + pos

            for layer in self.layers:
                h = layer(h, attention_mask)

            cls_token = h[:, 0, :]
            mean_pool = h.mean(dim=1)
            combined = torch.cat([cls_token, mean_pool], dim=1)
            combined = self.ln(self.dropout(combined))
            logits = self.classifier(combined)
            return logits

    # ========================================================================
    # ОСНОВНОЙ КЛАСС ENGINE
    # ========================================================================
    
    def __init__(
        self,
        repo_id: str = "MarkProMaster229/ClassificationSmall",
        config_path: str = "config.json",
        weights_path: str = "model_weights.pth",
        tokenizer_path: str = "tokenizer.json",
        vocab_path: str = "vocab.txt"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Скачиваем конфиг
        print(f"📥 Загрузка конфига: {repo_id}/{config_path}")
        config_file = hf_hub_download(repo_id=repo_id, filename=config_path)
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 2. Скачиваем токенизатор и vocab
        print(f"📥 Загрузка токенизатора...")
        tokenizer_file = hf_hub_download(repo_id=repo_id, filename=tokenizer_path)
        vocab_file = hf_hub_download(repo_id=repo_id, filename=vocab_path)
        
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            vocab_file=vocab_file
        )
        
        # 3. Создаём модель
        self.model = self.TransformerClassifier(
            vocabSize=self.config.get('vocabSize', 119547),
            maxLen=self.config.get('maxLen', 100),
            sizeVector=self.config.get('sizeVector', 256),
            numBlocks=self.config.get('numLayers', 4),
            numHeads=self.config.get('numHeads', 8),
            numClasses=self.config.get('numClasses', 3),
            dropout=self.config.get('dropout', 0.1)
        ).to(self.device)
        
        # 4. Грузим веса
        print(f"📥 Загрузка весов: {repo_id}/{weights_path}")
        weights_file = hf_hub_download(repo_id=repo_id, filename=weights_path)
        state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        
        # 5. Маппинг меток
        self.id2label = self.config.get('id2label', {
            0: "positive",
            1: "negative",
            2: "neutral"
        })
        
        print(f"✅ Engine (Transformer Classifier) загружен")
        print(f"   Устройство: {self.device}")
        print(f"   Классы: {self.id2label}")
    
    def predict(self, text: str, max_length: int = None):
        """
        Предсказание тональности текста.
        
        Args:
            text: строка для классификации
            max_length: максимальная длина (по умолчанию из конфига)
            
        Returns:
            tuple: (метка, уверенность)
        """
        if max_length is None:
            max_length = self.config.get('maxLen', 100)
        
        # Токенизация
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Инференс
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask)
        
        # Софтмакс и argmax
        probs = F.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        
        pred_class = pred_class.item()
        confidence = confidence.item()
        
        label = self.id2label.get(str(pred_class), f"class_{pred_class}")
        
        return label, confidence
    
    def predict(self, texts: list, max_length: int = None):
        results = []
        for text in texts:
            label, conf = self.predict(text, max_length)
            results.append((label, conf))
        return results