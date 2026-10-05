import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.utils.checkpoint import checkpoint


class GQAAttention(nn.Module):
    def __init__(self, d_model, n_heads_q, n_heads_kv, head_dim=None, dropout=0.0):
        super().__init__()
        if head_dim is None:
            head_dim = d_model // n_heads_q
        
        self.n_heads_q = n_heads_q
        self.n_heads_kv = n_heads_kv
        self.head_dim = head_dim
        self.inner_dim_q = n_heads_q * head_dim
        self.inner_dim_kv = n_heads_kv * head_dim
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, self.inner_dim_q, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim_kv, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim_kv, bias=False)
        self.out_proj = nn.Linear(self.inner_dim_q, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        n_rep = self.n_heads_q // self.n_heads_kv
        
        q = self.q_proj(x).view(B, T, self.n_heads_q, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads_kv, self.head_dim).transpose(1, 2)
        
        q = q.contiguous().view(B, self.n_heads_kv, n_rep, T, self.head_dim)
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)
        
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.reshape(B, self.n_heads_q, T, self.head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, self.inner_dim_q)
        return self.out_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads_q, n_heads_kv, head_dim=None, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GQAAttention(d_model, n_heads_q, n_heads_kv, head_dim=head_dim, dropout=dropout)
        
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

class TransformerBlockBig(nn.Module):
    def __init__(self, d_model, n_heads_q, n_heads_kv, head_dim=None, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GQAAttention(d_model, n_heads_q, n_heads_kv, head_dim=head_dim, dropout=dropout)
        
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 6 * d_model),
            nn.GELU(),
            nn.Linear(6 * d_model, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln_1(x)))
        x = x + self.drop(self.mlp(self.ln_2(x)))
        return x

class AttnMambaBlock(nn.Module):
    def __init__(self, d_model, n_heads_q, n_heads_kv, head_dim=None, 
                 dropout=0.1, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GQAAttention(d_model, n_heads_q, n_heads_kv, 
                                 head_dim=head_dim, dropout=dropout)
        
        self.ln_2 = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln_1(x)))
        x = x + self.drop(self.mamba(self.ln_2(x)))
        return x

class FinalBlock(nn.Module):
    def __init__(self, d_model, n_heads_q, n_heads_kv, head_dim=None, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GQAAttention(d_model, n_heads_q, n_heads_kv, head_dim=head_dim, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
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
    def __init__(self, vocab_size, d_model, max_len, n_heads_q_my, n_heads_kv_my,
                 head_dim=None, dropout_prob=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout_prob)
        
        self.layer_1_mamba = MambaBlock(d_model, dropout_prob)
        
        self.layer_2_transformer = TransformerBlockBig(
            d_model, n_heads_q_my, n_heads_kv_my, head_dim=head_dim, dropout=dropout_prob
        )
        
        self.layer_3_mamba = MambaBlock(d_model, dropout_prob)
        self.layer_4_mamba = MambaBlock(d_model, dropout_prob)

        self.layer_5_AttnMambaBlock = AttnMambaBlock(
            d_model, n_heads_q_my, n_heads_kv_my, head_dim=head_dim, dropout=dropout_prob
        )
        self.layer_6_mamba = MambaBlock(d_model, dropout_prob)
        self.layer_7_mamba = MambaBlock(d_model, dropout_prob)
        
        self.layer_8_FinalBlock = FinalBlock(
            d_model, n_heads_q_my, n_heads_kv_my, head_dim=head_dim, dropout=dropout_prob
        )

        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        B, T = token_ids.size()
        
        pos = torch.arange(0, T, dtype=torch.long, device=token_ids.device)
        
        x = self.token_embedding(token_ids) + self.position_embedding(pos)
        x = self.emb_dropout(x)
        
        x = self.layer_1_mamba(x)
        x = checkpoint(self.layer_2_transformer, x, use_reentrant=False)
        x = self.layer_3_mamba(x)
        x = self.layer_4_mamba(x)
        x = checkpoint(self.layer_5_AttnMambaBlock, x, use_reentrant=False)
        x = self.layer_6_mamba(x)
        x = checkpoint(self.layer_7_mamba, x, use_reentrant=False)
        x = self.layer_8_FinalBlock(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits


def build_model(vocab_size, config=None):
    cfg = config or {}
    return CustomHybridGPT(
        vocab_size=vocab_size,
        d_model=cfg.get("d_model", 512),
        max_len=cfg.get("max_len", 1024),
        n_heads_q_my=cfg.get("n_heads_q", 16),
        n_heads_kv_my=cfg.get("n_heads_kv", 4),
        head_dim=cfg.get("head_dim", None),
        dropout_prob=cfg.get("dropout", 0.1),
    )


if __name__ == "__main__":
    model = CustomHybridGPT(
        vocab_size=1000,
        d_model=512,
        max_len=1024,
        n_heads_q_my=16,
        n_heads_kv_my=4,
        head_dim=64,
        dropout_prob=0.1,
    ).cuda()

    dummy_input = torch.randint(0, 1000, (2, 8)).cuda()
    logits = model(dummy_input)
    print("Размер выходных логитов:", logits.shape)