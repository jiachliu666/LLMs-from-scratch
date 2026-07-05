cfg = {
    "vocab_size": 100000,
    "context_length": 1024,
    "emb_dim": 768,
    "head_num": 12,
    "dropout_rate": 0.15,
    "qkv_bias": False,
    "transformer_layers": 12,
}
import torch
import torch.nn as nn


class MHA(nn.Module):
    def __init__(self, d_in, d_out, context_length, qkv_bias, dropout_rate, head_num):
        super().__init__()
        self.head_num = head_num
        assert d_out % head_num == 0
        self.head_dim = d_out // head_num

        self.W_q = nn.Linear(d_in, d_out, qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagnol=1))

    def forward(self, x):
        # O(B*T*d^2) for QKV projections + O(B*T^2*d) for attention scores/weighted sum + O(B*T*d^2) for out_proj => O(B*T*d^2 + B*T^2*d)
        B, T, d = x.shape
        q = self.W_q(x).view(B, T, self.head_num, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.head_num, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.head_num, self.head_dim).transpose(1, 2)
        attn = q @ k.transpose(2, 3)
        attn = attn.mask_fill(self.mask.bool()[:T, :T], -torch.inf)
        attn = torch.softmax(attn / k.dim[-1] ** 0.5, dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(B, T, d)
        return self.out_proj(ctx)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # O(B*T*d): mean, variance, and normalize each scan all d elements per token
        means = x.means(dim=-1)
        vars = x.vars(dim=-1)
        return self.scale * (x - means) / torch.sqrt(vars + self.eps) + self.shift


class GeLU(nn.Module):
    def __init__(self, x):
        return x  # x / 2 * (1 + erf(x / sqrt(2))) x


class FeedForward(nn.Module):
    def __init__(self, cfg):
        self.layer = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GeLU, nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))

    def forward(self, x):
        # O(B*T*d^2): two linear projections d->4d and 4d->d dominate; GeLU is O(B*T*d)
        return self.layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        self.attn = MHA(**cfg)
        self.ff = FeedForward(cfg)
        self.layer_norm1 = LayerNorm(cfg["emb_dim"])
        self.layer_norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x):
        # O(B*T^2*d + B*T*d^2): MHA dominates (quadratic in T); FFN adds another O(B*T*d^2) term
        x = x + self.dropout(self.attn(self.layer_norm1(x)))
        x = x + self.dropout(self.attn(self.layer_norm2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.transformer = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["transformer_layer"])])
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])

    def forward(self, x):
        # O(L*(B*T^2*d + B*T*d^2) + B*T*d*V): L transformer blocks dominate for long sequences; output projection over vocab adds O(B*T*d*V)
        B, T = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        x = self.transformer(x)
        x = self.dropout(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
