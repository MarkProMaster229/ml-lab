import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer
class TransformerBlock(nn.Module):
    def __init__(self, sizeVector=128, numHeads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(sizeVector)
        self.attn = nn.MultiheadAttention(sizeVector, numHeads, batch_first=True)
        self.ln2 = nn.LayerNorm(sizeVector)
        self.ff = nn.Sequential(
            nn.Linear(sizeVector, sizeVector * 4),
            nn.GELU(),
            nn.Linear(sizeVector * 4, sizeVector),
        )

    def forward(self, x, attMask=None):
        h = self.ln1(x)
        z, _ = self.attn(h, h, h, attn_mask=attMask)
        x = x + z

        h = self.ln2(x)
        z = self.ff(h)
        x = x + z

        return x


class TransformerRun(nn.Module):
    def __init__(self, vocabSize=119547, maxLong=256, sizeVector=128, block=2):
        super().__init__()
        self.maxLong = maxLong
        self.tokenEmbed = nn.Embedding(vocabSize, sizeVector)
        self.posEmbed = nn.Embedding(maxLong, sizeVector)
        self.ln_f = nn.LayerNorm(sizeVector)

        self.layers = nn.ModuleList([
            TransformerBlock(sizeVector=sizeVector, numHeads=4)
            for _ in range(block)
        ])

        self.lmHead = nn.Linear(sizeVector, vocabSize)
        mask = torch.triu(torch.full((maxLong, maxLong), float('-inf')), diagonal=1)
        self.register_buffer("attMask", mask)

    def forward(self, x):
        B, T = x.shape
        tok = self.tokenEmbed(x)
        pos = self.posEmbed(torch.arange(T, device=x.device)).unsqueeze(0)
        h = tok + pos
        attMask = self.attMask[:T, :T]
        for layer in self.layers:
            h = layer(h, attMask)
        h = self.ln_f(h)
        return self.lmHead(h)


# фильтрация повторов не ок(
def generate(model, start_tokens, length=50, device='cpu', temperature=1.0):

    model.eval()
    tokens = start_tokens.clone().to(device)
    for _ in range(length):
        with torch.no_grad():
            logits = model(tokens)
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)

            if tokens.size(1) > 0:
                last_token = tokens[:, -1].unsqueeze(-1)
                probs.scatter_(dim=-1, index=last_token, value=0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
    return tokens


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "/home/chelovek/exper/newtrainedModel/epoch_7/model_weights.pth"
    tokenizer_path = "/home/chelovek/exper/newtrainedModel/epoch_7"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model = TransformerRun()
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)

    start_text = "Привет, я "
    start_tokens = tokenizer.encode(start_text, return_tensors="pt").to(device)

    generated_tokens = generate(
        model,
        start_tokens,
        length=100,
        device=device,
        temperature=0.5
    )

    generated_text = tokenizer.decode(generated_tokens[0].tolist(), skip_special_tokens=True)
    print("Generated text:\n", generated_text)
