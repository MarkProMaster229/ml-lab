import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class GQAAttention(nn.Module):
    def __init__(self, d_model, n_heads_q, n_heads_kv):
        super().__init__()
        self.d_model = d_model
        self.n_heads_q = n_heads_q
        self.n_heads_kv = n_heads_kv
        self.head_dim = d_model // n_heads_q

        self.q_proj = nn.Linear(d_model, n_heads_q * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads_kv * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        
        q = self.q_proj(x) 
        k = self.k_proj(x) 
        v = self.v_proj(x) 
        
        q = q.view(B, T, self.n_heads_q, self.head_dim).transpose(1, 2)   
        k = k.view(B, T, self.n_heads_kv, self.head_dim).transpose(1, 2)  
        v = v.view(B, T, self.n_heads_kv, self.head_dim).transpose(1, 2)  
        
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        
        return self.out_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads_q, n_heads_kv, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GQAAttention(d_model, n_heads_q, n_heads_kv)
        
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln_1(x)))
        x = x + self.drop(self.mlp(self.ln_2(x)))
        return x

class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,    
            d_state=16,         
            d_conv=4,           
            expand=2,           
        )
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mamba(self.ln(x)))


class CustomHybridGPT(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_heads_q_my, n_heads_kv_my, dropout_prob=0.1):
        super().__init__()
        

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout_prob)
        
        self.layer_1_mamba = MambaBlock(d_model, dropout_prob)
        
        self.layer_2_transformer = TransformerBlock(d_model, n_heads_q_my, n_heads_kv_my, dropout_prob)
        
        self.layer_3_mamba = MambaBlock(d_model, dropout_prob)
        self.layer_4_mamba = MambaBlock(d_model, dropout_prob)

        self.layer_5_transformer = TransformerBlock(d_model, n_heads_q_my, n_heads_kv_my, dropout_prob)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        B, T = token_ids.size()
        
        pos = torch.arange(0, T, dtype=torch.long, device=token_ids.device)
        
        x = self.token_embedding(token_ids) + self.position_embedding(pos)
        x = self.emb_dropout(x)
    
        x = self.layer_1_mamba(x)        # Шаг 1: Мамба0
        x = self.layer_2_transformer(x)  # Шаг 2: Первое QKV внимание
        x = self.layer_3_mamba(x)        # Шаг 3: Мамба1
        x = self.layer_4_mamba(x)        # Шаг 4: Мамба2
        x = self.layer_5_transformer(x)  # Шаг 5: Итоговое QKV представление
        
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        
        return logits


if __name__ == "__main__":
    model = CustomHybridGPT(
        vocab_size=1000, 
        d_model=512, 
        max_len=1024,  
        n_heads_q_my=16,
        n_heads_kv_my=4,
        dropout_prob=0.1
    ).cuda()
    
    dummy_input = torch.randint(0, 1000, (2, 8)).cuda()
    
    logits = model(dummy_input)
    print("Размер выходных логитов:", logits.shape) # Ожидается [2, 8, 1000]
