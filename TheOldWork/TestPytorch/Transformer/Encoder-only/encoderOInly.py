import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class TransformerBlock(nn.Module):
    def __init__(self, sizeVector=256, numHeads=8, dropout=0.1):
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

        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class TransformerRun(nn.Module):
    def __init__(self, vocabSize=120000, maxLen=100, sizeVector=256, numBlocks=4, numHeads=8, numClasses=3, dropout=0.1):
        super().__init__()

        self.token_emb = nn.Embedding(vocabSize, sizeVector)
        self.pos_emb = nn.Embedding(maxLen, sizeVector)

        self.cls_token = nn.Parameter(torch.randn(1, 1, sizeVector))

        self.layers = nn.ModuleList([
            TransformerBlock(sizeVector=sizeVector, numHeads=numHeads, dropout=dropout)
            for _ in range(numBlocks)
        ])

        self.ln = nn.LayerNorm(sizeVector * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(sizeVector * 2, numClasses)

    def forward(self, x, attention_mask=None):
        B, T = x.shape

        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0).expand(B, T))
        h = tok + pos

        # add CLS
        cls = self.cls_token.expand(B, 1, -1)
        h = torch.cat([cls, h], dim=1)

        for layer in self.layers:
            h = layer(h, attention_mask)

        cls_token = h[:, 0, :]
        mean_pool = h[:, 1:, :].mean(dim=1)

        combined = torch.cat([cls_token, mean_pool], dim=1)
        combined = self.dropout(self.ln(combined))
        logits = self.classifier(combined)
        return logits
