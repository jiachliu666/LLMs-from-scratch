"""
Ch04: Implementing a GPT Model from Scratch — 互动学习文件
==========================================================
运行方式:
    python -m pytest learn_ch04.py -v
    python -m pytest learn_ch04.py -v -s        # 显示 print 输出
    python -m pytest learn_ch04.py -v -k demo   # 只跑演示

知识地图（Google MLE 面试级别）:
  Section1: GPT 配置 & 整体架构
  Section2: LayerNorm — 为什么用？和 BatchNorm 的区别？
  Section3: GELU 激活函数 — 为什么不用 ReLU？
  Section4: FeedForward 网络 — 4x 扩展的意义
  Section5: 残差连接（Shortcut）— 解决梯度消失
  Section6: TransformerBlock — Pre-Norm 架构
  Section7: GPTModel 完整架构 — 参数量 & Weight Tying
  Section8: 文本生成（Greedy Decoding）
  Section9: 面试题集中考察
"""

import unittest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 共享配置
# ─────────────────────────────────────────────────────────────────────────────
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

# 小配置，用于快速测试（不需要加载完整 GPT-2）
GPT_CONFIG_SMALL = {
    "vocab_size": 100,
    "context_length": 16,
    "emb_dim": 32,
    "n_heads": 4,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "n_layers": 2,
}


# ─────────────────────────────────────────────────────────────────────────────
# 模型组件定义（从 notebook 复制，供测试使用）
# ─────────────────────────────────────────────────────────────────────────────
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 残差连接 1：attention

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 残差连接 2：FFN
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape  # [2, 6]
        tok_embeds = self.tok_emb(in_idx)  # [2, 6, 32]
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ═════════════════════════════════════════════════════════════════════════════
# 第 1 节  GPT 配置 & 整体架构
# ═════════════════════════════════════════════════════════════════════════════
class Section1_GPTConfig(unittest.TestCase):
    """
    【概念】GPT-2 Small (124M) 的超参数：
      vocab_size=50257   BPE 词表大小
      context_length=1024 最大序列长度（位置编码的上限）
      emb_dim=768        每个 token 的向量维度
      n_heads=12         多头注意力的头数
      n_layers=12        TransformerBlock 的堆叠层数
      drop_rate=0.1      10% 的 dropout 防过拟合
      qkv_bias=False     Q/K/V 投影不加 bias（现代 LLM 的惯例）

    【面试考点】
    - 为什么 vocab_size=50257？GPT-2 BPE 词表，256 字节 + 合并 = 50000 + 特殊 token
    - head_dim = emb_dim / n_heads = 768/12 = 64
    - 参数量计算：Embedding + n_layers * TransformerBlock + output head
    """

    def test_demo_01_config_values(self):
        """GPT-2 small 的关键超参数"""
        cfg = GPT_CONFIG_124M
        self.assertEqual(cfg["vocab_size"], 50257)
        self.assertEqual(cfg["emb_dim"], 768)
        self.assertEqual(cfg["n_heads"], 12)
        self.assertEqual(cfg["n_layers"], 12)

    def test_demo_02_head_dim(self):
        """head_dim = emb_dim / n_heads = 64"""
        cfg = GPT_CONFIG_124M
        head_dim = cfg["emb_dim"] // cfg["n_heads"]
        self.assertEqual(head_dim, 64)

    def test_demo_03_output_shape(self):
        """GPT 输出 logits 形状：[batch, seq_len, vocab_size]"""
        torch.manual_seed(0)
        model = GPTModel(GPT_CONFIG_SMALL)
        model.eval()
        idx = torch.randint(0, 100, (2, 8))
        logits = model(idx)
        self.assertEqual(logits.shape, torch.Size([2, 8, 100]))

    def test_ex_01_input_output_same_seq_len(self):
        """
        练习 1.1: GPT 的输入和输出序列长度相同。
        输入 [2, 6]，输出 logits 的前两维是 [2, 6]。
        """
        torch.manual_seed(0)
        model = GPTModel(GPT_CONFIG_SMALL)
        model.eval()
        idx = torch.randint(0, 100, (2, 6))
        logits = model(idx)

        # TODO
        expected_shape = torch.Size([2, 6, 100])
        # raise NotImplementedError("TODO 1.1")
        self.assertEqual(logits.shape, expected_shape)

    def test_ex_02_emb_dim_divisible_by_n_heads(self):
        """
        练习 1.2: emb_dim 必须能被 n_heads 整除，否则 MultiHeadAttention 报错。
        验证 768 % 12 == 0。
        """
        cfg = GPT_CONFIG_124M
        # TODO
        remainder = GPT_CONFIG_124M["emb_dim"] % GPT_CONFIG_124M["n_heads"]
        # raise NotImplementedError("TODO 1.2")
        self.assertEqual(remainder, 0)


# ═════════════════════════════════════════════════════════════════════════════
# 第 2 节  Layer Normalization
# ═════════════════════════════════════════════════════════════════════════════
class Section2_LayerNorm(unittest.TestCase):
    """
    【概念】LayerNorm 对每个样本的特征维度归一化：
      norm_x = (x - mean) / sqrt(var + eps)
      output = scale * norm_x + shift

    与 BatchNorm 的区别：
      BatchNorm: 跨样本（batch 维）归一化，依赖 batch size
      LayerNorm: 跨特征（feature 维）归一化，batch size=1 也能用

    【面试考点】
    - 为什么 LLM 用 LayerNorm 不用 BatchNorm？
      序列长度可变，batch size 可能为 1（推理时），BatchNorm 效果差
    - eps 的作用：防止方差为 0 时除以零
    - unbiased=False：用 n 而不是 n-1 计算方差（与 GPT-2 训练一致）
    - scale/shift 是可训练参数，让模型恢复归一化前的表达能力
    """

    def setUp(self):
        self.ln = LayerNorm(emb_dim=4)
        # 初始化：scale=1, shift=0，所以初始状态就是纯归一化
        torch.manual_seed(0)
        self.x = torch.randn(2, 4)  # [batch=2, features=4]

    def test_demo_01_output_mean_near_zero(self):
        """LayerNorm 后每个样本的均值接近 0"""
        out = self.ln(self.x)
        means = out.mean(dim=-1)
        self.assertTrue(torch.allclose(means, torch.zeros(2), atol=1e-5))

    def test_demo_02_output_var_near_one(self):
        """LayerNorm 后每个样本的方差接近 1"""
        out = self.ln(self.x)
        vars_ = out.var(dim=-1, unbiased=False)
        self.assertTrue(torch.allclose(vars_, torch.ones(2), atol=1e-3))

    def test_demo_03_scale_shift_are_trainable(self):
        """scale 和 shift 是可训练参数"""
        param_names = [n for n, _ in self.ln.named_parameters()]
        self.assertIn("scale", param_names)
        self.assertIn("shift", param_names)

    def test_demo_04_initial_scale_ones_shift_zeros(self):
        """初始 scale=1（不改变大小），shift=0（不改变位置）"""
        self.assertTrue(torch.all(self.ln.scale == 1.0))
        self.assertTrue(torch.all(self.ln.shift == 0.0))

    def test_demo_05_eps_prevents_div_zero(self):
        """常数输入（方差为 0）时，eps 防止除以零"""
        x_const = torch.ones(2, 4)  # 所有元素相同，方差=0
        out = self.ln(x_const)  # 不应该报错
        self.assertFalse(torch.isnan(out).any())

    def test_demo_06_independent_across_samples(self):
        """LayerNorm 对每个样本独立归一化，修改一个样本不影响另一个"""
        x1 = torch.randn(2, 4)
        x2 = x1.clone()
        x2[0] *= 100  # 只改第 0 个样本

        out1 = self.ln(x1)
        out2 = self.ln(x2)

        # 第 1 个样本不受影响
        self.assertTrue(torch.allclose(out1[1], out2[1], atol=1e-5))
        # 第 0 个样本受影响
        self.assertFalse(torch.allclose(out1[0], out2[0]))

    def test_ex_01_manual_layer_norm(self):
        """
        练习 2.1: 手动实现 LayerNorm（不用 nn.LayerNorm），
                  验证结果与 LayerNorm 类一致。

        公式: norm_x = (x - mean) / sqrt(var + eps)
        """
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        ln = LayerNorm(emb_dim=4)

        # TODO: 手动计算

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        manual_out = (x - mean) / torch.sqrt(var + 1e-5)

        # raise NotImplementedError("TODO 2.1")

        expected = ln(x)
        self.assertTrue(torch.allclose(manual_out, expected, atol=1e-5))

    def test_ex_02_layernorm_vs_batchnorm_behavior(self):
        """
        练习 2.2: 【面试题】验证 LayerNorm 和 BatchNorm 归一化方向不同。

        LayerNorm:  每行（样本）内部归一化 → dim=-1
        BatchNorm:  每列（特征）跨样本归一化 → dim=0

        对 [2, 4] 的输入：
          LayerNorm 后，每行均值≈0
          BatchNorm 后，每列均值≈0
        """
        torch.manual_seed(0)
        x = torch.randn(2, 4)

        ln_out = LayerNorm(4)(x)
        bn_out = nn.BatchNorm1d(4)(x)

        # LayerNorm：行均值为 0
        self.assertTrue(torch.allclose(ln_out.mean(dim=-1), torch.zeros(2), atol=1e-5))
        # BatchNorm：列均值为 0
        self.assertTrue(torch.allclose(bn_out.mean(dim=0), torch.zeros(4), atol=1e-5))


# ═════════════════════════════════════════════════════════════════════════════
# 第 3 节  GELU 激活函数
# ═════════════════════════════════════════════════════════════════════════════
class Section3_GELU(unittest.TestCase):
    """
    【概念】GELU (Gaussian Error Linear Unit)：
      GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))

    vs ReLU(x) = max(0, x)

    【面试考点】
    - 为什么 LLM 用 GELU 不用 ReLU？
      1. GELU 在 x<0 时有非零梯度（约 -0.75 处除外），避免"神经元死亡"
      2. GELU 是平滑函数，ReLU 在 x=0 处不可微
      3. 实验上 GELU 在 NLP 任务中表现更好（BERT, GPT 都用 GELU）
    - GELU(0) = 0，GELU(x→+∞) ≈ x，GELU(x→-∞) ≈ 0
    """

    def setUp(self):
        self.gelu = GELU()
        self.relu = nn.ReLU()

    def test_demo_01_gelu_at_zero(self):
        """GELU(0) = 0"""
        self.assertAlmostEqual(self.gelu(torch.tensor(0.0)).item(), 0.0, places=5)

    def test_demo_02_gelu_positive_approx_identity(self):
        """GELU(x) ≈ x for large positive x"""
        x = torch.tensor(10.0)
        self.assertAlmostEqual(self.gelu(x).item(), x.item(), delta=0.01)

    def test_demo_03_gelu_negative_nonzero(self):
        """GELU 对负数有非零输出（与 ReLU 不同）"""
        x = torch.tensor(-1.0)
        gelu_out = self.gelu(x).item()
        relu_out = self.relu(x).item()
        self.assertEqual(relu_out, 0.0)  # ReLU 直接截断
        self.assertNotEqual(gelu_out, 0.0)  # GELU 有非零值（约 -0.16）

    def test_demo_04_relu_hard_zero_for_negative(self):
        """ReLU 对所有负数输出精确的 0"""
        x = torch.linspace(-5, -0.01, 100)
        self.assertTrue((self.relu(x) == 0).all())

    def test_demo_05_gelu_smooth_near_zero(self):
        """GELU 在 0 附近是平滑的（相邻点的值连续变化）"""
        x1 = torch.tensor(-0.01)
        x2 = torch.tensor(0.01)
        # GELU 值连续，差值很小
        diff = abs(self.gelu(x1).item() - self.gelu(x2).item())
        self.assertLess(diff, 0.02)

    def test_ex_01_gelu_vs_relu_for_small_negative(self):
        """
        练习 3.1: x=-0.5 时，GELU 和 ReLU 的输出分别是多少？
        验证 GELU(-0.5) < 0（负数）而 ReLU(-0.5) = 0
        """
        x = torch.tensor(-0.5)
        import math

        # TODO
        gelu_out = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / 3.1415926) * (x + 0.044715 * x**3)))
        relu_out = max(0, x)
        # raise NotImplementedError("TODO 3.1")

        self.assertLess(gelu_out, 0.0)
        self.assertEqual(relu_out, 0.0)

    def test_ex_02_gelu_output_shape(self):
        """
        练习 3.2: GELU 是 element-wise 操作，输出形状和输入相同。
        """
        x = torch.randn(2, 4, 768)
        out = self.gelu(x)

        # TODO
        expected_shape = torch.Size([2, 4, 768])
        # raise NotImplementedError("TODO 3.2")
        self.assertEqual(out.shape, expected_shape)


# ═════════════════════════════════════════════════════════════════════════════
# 第 4 节  FeedForward 网络
# ═════════════════════════════════════════════════════════════════════════════
class Section4_FeedForward(unittest.TestCase):
    """
    【概念】Transformer 中的 FFN（Position-wise Feed-Forward Network）：
      x → Linear(d, 4d) → GELU → Linear(4d, d) → output

    关键特点：
      - 先扩张 4 倍再压缩回来（4x expansion）
      - 输入输出形状相同
      - 每个位置独立计算（position-wise）

    【面试考点】
    - 为什么扩张 4 倍？
      中间层更宽 → 更大的特征空间 → 更强的非线性表达能力
      实验上 4x 是 transformer 的经验最优值
    - FFN 的作用：注意力层学习 token 间的关系，FFN 学习 token 内部的特征变换
    - 参数量：2 * emb_dim * 4 * emb_dim = 8 * emb_dim²
      GPT-2: 8 * 768² = 4,718,592 per layer
    """

    def setUp(self):
        self.cfg = GPT_CONFIG_SMALL
        self.ffn = FeedForward(self.cfg)

    def test_demo_01_output_shape_preserved(self):
        """FFN 输入输出形状完全相同：[batch, T, emb_dim]"""
        x = torch.randn(2, 6, self.cfg["emb_dim"])
        out = self.ffn(x)
        self.assertEqual(out.shape, x.shape)

    def test_demo_02_hidden_dim_is_4x(self):
        """中间层维度是 4 * emb_dim"""
        # layers[0] 是第一个 Linear
        w = self.ffn.layers[0].weight
        self.assertEqual(w.shape[0], 4 * self.cfg["emb_dim"])  # out_features = 4*emb_dim

    def test_demo_03_position_wise_independent(self):
        """FFN 对每个位置独立计算：改变一个位置不影响其他位置"""
        x1 = torch.randn(1, 4, self.cfg["emb_dim"])
        x2 = x1.clone()
        x2[0, 2, :] = torch.randn(self.cfg["emb_dim"])  # 只改位置 2

        out1 = self.ffn(x1)
        out2 = self.ffn(x2)

        self.assertTrue(torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5))
        self.assertTrue(torch.allclose(out1[0, 1], out2[0, 1], atol=1e-5))
        self.assertFalse(torch.allclose(out1[0, 2], out2[0, 2]))

    def test_demo_04_parameter_count(self):
        """FFN 参数量 = 2 * emb_dim * 4 * emb_dim（bias=True 时还要加 bias）"""
        emb = self.cfg["emb_dim"]
        # Linear(emb, 4*emb) weights + Linear(4*emb, emb) weights
        # + 2 bias vectors
        expected = emb * 4 * emb + 4 * emb + 4 * emb * emb + emb
        total = sum(p.numel() for p in self.ffn.parameters())
        self.assertEqual(total, expected)

    def test_ex_01_ffn_shape_with_gpt2_config(self):
        """
        练习 4.1: 用 GPT-2 配置（emb_dim=768）的 FFN，
                  输入 [2, 10, 768]，输出形状是？
        """
        ffn = FeedForward({"emb_dim": 768})
        x = torch.randn(2, 10, 768)
        out = ffn(x)

        # TODO
        expected_shape = torch.Size([2, 10, 768])
        # raise NotImplementedError("TODO 4.1")
        self.assertEqual(out.shape, expected_shape)

    def test_ex_02_intermediate_dimension(self):
        """
        练习 4.2: emb_dim=768 时，FFN 中间层的维度是多少？
        """
        # TODO
        intermediate_dim = 768 * 4
        # raise NotImplementedError("TODO 4.2")
        self.assertEqual(intermediate_dim, 3072)


# ═════════════════════════════════════════════════════════════════════════════
# 第 5 节  残差连接（Shortcut Connection）
# ═════════════════════════════════════════════════════════════════════════════
class Section5_ResidualConnection(unittest.TestCase):
    """
    【概念】残差连接（Residual/Shortcut Connection）：
      output = layer(x) + x

    原来（无残差）：
      x → Layer1 → Layer2 → Layer3 → ... → output

    有残差：
      x → Layer1 → + → Layer2 → + → ...
          ↑_________|   ↑___________|
            直接跳过       直接跳过

    【面试考点】
    - 为什么残差连接能解决梯度消失？
      反向传播时，梯度可以通过"跳过"路径直接流回早期层：
      ∂L/∂x = ∂L/∂(layer(x)+x) = ∂L/∂output * (∂layer(x)/∂x + 1)
      "+1" 保证即使 ∂layer/∂x 很小，梯度也能流动
    - 残差连接让深层网络等价于"很多浅层网络的集成"
    - 要求：x 和 layer(x) 形状相同才能相加
    """

    def test_demo_01_gradient_vanishing_without_shortcut(self):
        """
        无残差连接时，早期层的梯度远小于末层（梯度消失）
        """
        torch.manual_seed(123)
        layer_sizes = [3, 3, 3, 3, 3, 1]

        # 无残差网络
        layers = nn.ModuleList([nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i + 1]), GELU()) for i in range(len(layer_sizes) - 1)])

        x = torch.tensor([[1.0, 0.0, -1.0]])
        out = x
        for layer in layers:
            out = layer(out)

        loss = nn.MSELoss()(out, torch.tensor([[0.0]]))
        loss.backward()

        grads = [layers[i][0].weight.grad.abs().mean().item() for i in range(5)]
        # 早期层梯度 < 末层梯度（梯度随层数增加而增大，从末层向前看）
        self.assertLess(grads[0], grads[4])

    def test_demo_02_shortcut_alleviates_vanishing(self):
        """
        有残差连接时，各层梯度更均衡（早期层梯度不再极小）
        """
        torch.manual_seed(123)
        d = 3

        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(d, d)
                self.act = GELU()

            def forward(self, x):
                return x + self.act(self.linear(x))

        layers = nn.ModuleList([ResBlock() for _ in range(5)])
        final = nn.Linear(d, 1)

        x = torch.tensor([[1.0, 0.0, -1.0]])
        out = x
        for layer in layers:
            out = layer(out)
        out = final(out)

        loss = nn.MSELoss()(out, torch.tensor([[0.0]]))
        loss.backward()

        grads = [layers[i].linear.weight.grad.abs().mean().item() for i in range(5)]
        # 有残差时，第 0 层的梯度不应该极小
        self.assertGreater(grads[0], 1e-4)

    def test_demo_03_residual_requires_same_shape(self):
        """残差连接要求 x 和 layer(x) 形状相同"""
        x = torch.randn(2, 4, 32)
        layer_same = nn.Linear(32, 32)  # 输出形状相同
        layer_diff = nn.Linear(32, 64)  # 输出形状不同

        # 形状相同：可以相加
        out_same = layer_same(x) + x
        self.assertEqual(out_same.shape, x.shape)

        # 形状不同：会报错
        with self.assertRaises(RuntimeError):
            _ = layer_diff(x) + x

    def test_ex_01_residual_connection_math(self):
        """
        练习 5.1: 手动实现一步残差连接。
        给定 x=[1,2,3]，layer 输出 [0.1, 0.2, 0.3]，
        残差输出 = x + layer(x) = [1.1, 2.2, 3.3]
        """
        x = torch.tensor([[1.0, 2.0, 3.0]])
        layer_out = torch.tensor([[0.1, 0.2, 0.3]])

        # TODO
        residual_out = x + layer_out
        # raise NotImplementedError("TODO 5.1")

        expected = torch.tensor([[1.1, 2.2, 3.3]])
        self.assertTrue(torch.allclose(residual_out, expected))


# ═════════════════════════════════════════════════════════════════════════════
# 第 6 节  TransformerBlock（Pre-Norm 架构）
# ═════════════════════════════════════════════════════════════════════════════
class Section6_TransformerBlock(unittest.TestCase):
    """
    【概念】TransformerBlock 把所有组件组合在一起：

      输入 x
        │
        ├─ shortcut1 = x
        │
        x = LayerNorm1(x)      ← Pre-Norm（归一化在前）
        x = MultiHeadAttention(x)
        x = Dropout(x)
        x = x + shortcut1      ← 残差连接 1
        │
        ├─ shortcut2 = x
        │
        x = LayerNorm2(x)
        x = FeedForward(x)
        x = Dropout(x)
        x = x + shortcut2      ← 残差连接 2
        │
        输出 x（形状不变）

    【面试考点】
    - Pre-Norm vs Post-Norm：
      原始 Transformer 是 Post-Norm（LN 在残差连接之后），训练不稳定
      GPT-2 用 Pre-Norm（LN 在子层之前），训练更稳定，现代 LLM 的标准

    - 为什么两个 LayerNorm？
      attention 和 FFN 各自负责不同的变换，分开归一化更合理

    - TransformerBlock 的输入输出形状完全相同：[b, T, emb_dim]
      这样才能堆叠 n_layers 层
    """

    def setUp(self):
        self.cfg = GPT_CONFIG_SMALL
        torch.manual_seed(0)
        self.block = TransformerBlock(self.cfg)
        self.block.eval()

    def test_demo_01_output_shape_preserved(self):
        """TransformerBlock 输入输出形状完全相同"""
        x = torch.randn(2, 8, self.cfg["emb_dim"])
        out = self.block(x)
        self.assertEqual(out.shape, x.shape)

    def test_demo_02_pre_norm_order(self):
        """验证 Pre-Norm：norm1 在 attention 之前，norm2 在 FFN 之前"""
        # 检查 TransformerBlock 有 norm1 和 norm2
        self.assertTrue(hasattr(self.block, "norm1"))
        self.assertTrue(hasattr(self.block, "norm2"))
        self.assertTrue(hasattr(self.block, "att"))
        self.assertTrue(hasattr(self.block, "ff"))

    def test_demo_03_block_can_stack(self):
        """n 个 TransformerBlock 可以串联（输入输出形状一致）"""
        x = torch.randn(2, 8, self.cfg["emb_dim"])
        blocks = nn.Sequential(*[TransformerBlock(self.cfg) for _ in range(3)])
        blocks.eval()
        out = blocks(x)
        self.assertEqual(out.shape, x.shape)

    def test_demo_04_residual_connection_effect(self):
        """残差连接使输出不完全等于子层输出（加上了原始输入）"""
        x = torch.randn(1, 4, self.cfg["emb_dim"])

        # 只跑 attention（无残差）
        normed = self.block.norm1(x)
        att_out = self.block.att(normed)

        # 完整 block（有残差）
        full_out = self.block(x)

        # full_out ≠ att_out（因为还加了 FFN 和两次残差）
        self.assertFalse(torch.allclose(full_out, att_out))

    def test_ex_01_count_components(self):
        """
        练习 6.1: TransformerBlock 包含哪些子模块？
        验证有 att、ff、norm1、norm2、drop_shortcut 共 5 个。
        """
        child_names = [name for name, _ in self.block.named_children()]
        # TODO: 取消注释验证
        # raise NotImplementedError("TODO 6.1")
        self.assertIn("att", child_names)
        self.assertIn("ff", child_names)
        self.assertIn("norm1", child_names)
        self.assertIn("norm2", child_names)
        self.assertIn("drop_shortcut", child_names)

    def test_ex_02_pre_norm_vs_post_norm(self):
        """
        练习 6.2: 【面试题】Pre-Norm 和 Post-Norm 的区别。
        Pre-Norm:  x + sublayer(LN(x))   ← GPT-2，训练更稳定
        Post-Norm: LN(x + sublayer(x))   ← 原始 Transformer

        验证 GPT-2 的 TransformerBlock 使用 Pre-Norm（LN 在残差内部）。
        手动跑一步，检查第一步是 norm1 还是 att。
        """
        x = torch.randn(1, 4, self.cfg["emb_dim"])

        # Pre-Norm 顺序：先 norm，再 att，再残差
        normed_x = self.block.norm1(x)
        att_out = self.block.att(normed_x)
        out_prenorm = x + att_out  # 残差加的是原始 x

        # 验证 out_prenorm 是合法张量
        self.assertEqual(out_prenorm.shape, x.shape)
        self.assertFalse(torch.isnan(out_prenorm).any())


# ═════════════════════════════════════════════════════════════════════════════
# 第 7 节  GPTModel 完整架构 & 参数量
# ═════════════════════════════════════════════════════════════════════════════
class Section7_GPTModel(unittest.TestCase):
    """
    【概念】完整 GPT 架构：

      token_ids [b, T]
           ↓
      tok_emb + pos_emb → [b, T, emb_dim]
           ↓ Dropout
      TransformerBlock × n_layers
           ↓
      LayerNorm（final）
           ↓
      Linear(emb_dim, vocab_size)  → logits [b, T, vocab_size]

    参数量分析（GPT-2 small）：
      tok_emb:       50257 × 768 = 38,597,376
      pos_emb:        1024 × 768 =    786,432
      12 × TransformerBlock:     ≈ 85,054,464
      final_norm:               ≈      1,536
      out_head:      768 × 50257 = 38,597,376  ← 和 tok_emb 相同（Weight Tying 的依据）
      ─────────────────────────────────────────
      Total（不含 weight tying）: 163,009,536
      Total（含 weight tying）:   124,412,160  ← "124M" 的来源

    【面试考点】
    - Weight Tying：tok_emb 和 out_head 共享同一个权重矩阵
      直觉：把词转成向量，和把向量转回词，用同一个矩阵更高效
      节省参数：38.6M 参数
    - logits[:, -1, :] 取最后一个 token 的输出用于预测下一个词
    - pos_emb 是可训练的（GPT-2），不是固定的正弦函数（原始 Transformer）
    """

    def setUp(self):
        torch.manual_seed(0)
        self.model = GPTModel(GPT_CONFIG_SMALL)
        self.model.eval()
        self.cfg = GPT_CONFIG_SMALL

    def test_demo_01_output_shape(self):
        """GPT 输出 logits [b, T, vocab_size]"""
        idx = torch.randint(0, self.cfg["vocab_size"], (2, 8))
        logits = self.model(idx)
        self.assertEqual(logits.shape, torch.Size([2, 8, self.cfg["vocab_size"]]))

    def test_demo_02_tok_emb_and_out_head_same_shape(self):
        """tok_emb 和 out_head 权重形状相同（Weight Tying 的前提）"""
        tok_shape = self.model.tok_emb.weight.shape
        out_shape = self.model.out_head.weight.shape
        self.assertEqual(tok_shape, out_shape)  # 都是 [vocab_size, emb_dim]

    def test_demo_03_pos_emb_is_trainable(self):
        """位置编码是可训练的 nn.Embedding（不是固定正弦）"""
        self.assertIsInstance(self.model.pos_emb, nn.Embedding)
        self.assertTrue(self.model.pos_emb.weight.requires_grad)

    def test_demo_04_n_transformer_blocks(self):
        """TransformerBlock 堆叠 n_layers 次"""
        n_blocks = len(list(self.model.trf_blocks.children()))
        self.assertEqual(n_blocks, self.cfg["n_layers"])

    def test_demo_05_weight_tying_reduces_params(self):
        """Weight Tying 后参数量减少 vocab_size * emb_dim"""
        total = sum(p.numel() for p in self.model.parameters())
        out_head_params = sum(p.numel() for p in self.model.out_head.parameters())
        tied_total = total - out_head_params
        self.assertLess(tied_total, total)

    def test_demo_06_final_norm_before_output(self):
        """final_norm 在 out_head 之前（Pre-Norm 的一致性）"""
        self.assertIsInstance(self.model.final_norm, LayerNorm)
        self.assertIsInstance(self.model.out_head, nn.Linear)

    def test_ex_01_count_total_params_small(self):
        """
        练习 7.1: 计算 GPT_CONFIG_SMALL 的总参数量。
        手动算：
          tok_emb:  100 * 32 = 3200
          pos_emb:   16 * 32 =  512
          ...（TransformerBlock 内部较复杂）
          用 sum(p.numel() for p in model.parameters()) 验证
        """
        total = sum(p.numel() for p in self.model.parameters())
        # TODO: 只需验证大于 0 且合理
        # raise NotImplementedError("TODO 7.1: 打印并理解参数量")
        print(f"\nGPT_CONFIG_SMALL 总参数量: {total:,}")
        self.assertGreater(total, 0)

    def test_ex_02_logits_last_token(self):
        """
        练习 7.2: 生成时只取最后一个 token 的 logits。
        logits 形状 [b, T, vocab_size]，取 logits[:, -1, :] → [b, vocab_size]
        """
        idx = torch.randint(0, self.cfg["vocab_size"], (2, 6))
        logits = self.model(idx)

        # TODO
        last_token_logits = None
        raise NotImplementedError("TODO 7.2")

        self.assertEqual(last_token_logits.shape, torch.Size([2, self.cfg["vocab_size"]]))

    def test_ex_03_weight_tying_concept(self):
        """
        练习 7.3: 【面试题】实现 Weight Tying：
        让 out_head 的权重等于 tok_emb 的权重，验证参数量减少。
        """
        model = GPTModel(GPT_CONFIG_SMALL)
        total_before = sum(p.numel() for p in model.parameters())

        # Weight Tying
        model.out_head.weight = model.tok_emb.weight

        total_after = sum(p.numel() for p in model.parameters())

        # TODO: 取消注释验证
        # raise NotImplementedError("TODO 7.3: 取消注释验证")
        self.assertLess(total_after, total_before)


# ═════════════════════════════════════════════════════════════════════════════
# 第 8 节  文本生成（Greedy Decoding）
# ═════════════════════════════════════════════════════════════════════════════
class Section8_TextGeneration(unittest.TestCase):
    """
    【概念】Greedy Decoding（贪心解码）：
      每次选择概率最高的 token 作为下一个词。

    generate_text_simple 的流程：
      1. 取最近 context_size 个 token 作为输入
      2. 前向传播得到 logits [b, T, vocab_size]
      3. 只取最后一个位置的 logits [:, -1, :]
      4. argmax 得到最高概率的 token
      5. 追加到序列末尾
      6. 重复 max_new_tokens 次

    【面试考点】
    - 为什么取 logits[:, -1, :]？
      因果注意力保证位置 T-1 的输出融合了所有 [0..T-1] 的信息，
      是对"下一个词"的预测
    - Greedy vs Sampling vs Beam Search：
      Greedy：快，但重复性强
      Sampling（temperature）：多样性更好
      Beam Search：探索多条路径，质量更高但更慢
    - model.eval()：关闭 dropout，保证推理确定性
    - torch.no_grad()：推理时不需要梯度，节省显存
    """

    def setUp(self):
        torch.manual_seed(0)
        self.model = GPTModel(GPT_CONFIG_SMALL)
        self.model.eval()
        self.cfg = GPT_CONFIG_SMALL

    def test_demo_01_generate_extends_sequence(self):
        """生成 N 个新 token 后，序列长度增加 N"""
        idx = torch.randint(0, self.cfg["vocab_size"], (1, 4))
        out = generate_text_simple(self.model, idx, max_new_tokens=3, context_size=self.cfg["context_length"])
        self.assertEqual(out.shape[1], 4 + 3)

    def test_demo_02_greedy_is_deterministic(self):
        """Greedy decoding 是确定性的：相同输入总是相同输出"""
        idx = torch.randint(0, self.cfg["vocab_size"], (1, 4))
        out1 = generate_text_simple(self.model, idx.clone(), max_new_tokens=5, context_size=self.cfg["context_length"])
        out2 = generate_text_simple(self.model, idx.clone(), max_new_tokens=5, context_size=self.cfg["context_length"])
        self.assertTrue(torch.equal(out1, out2))

    def test_demo_03_original_tokens_preserved(self):
        """生成的序列前缀与输入完全一致"""
        idx = torch.tensor([[1, 2, 3, 4]])
        out = generate_text_simple(self.model, idx, max_new_tokens=3, context_size=self.cfg["context_length"])
        self.assertTrue(torch.equal(out[:, :4], idx))

    def test_demo_04_context_cropping(self):
        """输入超过 context_size 时，只取最后 context_size 个 token"""
        # 输入 20 个 token，但 context_size=16
        idx = torch.randint(0, self.cfg["vocab_size"], (1, 20))
        out = generate_text_simple(self.model, idx, max_new_tokens=1, context_size=self.cfg["context_length"])
        self.assertEqual(out.shape[1], 21)

    def test_demo_05_argmax_picks_highest_logit(self):
        """greedy decoding = argmax of logits（不需要 softmax）"""
        logits = torch.tensor([[1.0, 5.0, 2.0, 0.5]])  # 最大值在 index=1
        next_token = torch.argmax(logits, dim=-1)
        self.assertEqual(next_token.item(), 1)

    def test_ex_01_generate_n_tokens(self):
        """
        练习 8.1: 从 [5, 10, 15] 开始，生成 4 个新 token，
                  验证输出序列长度为 7。
        """
        idx = torch.tensor([[5, 10, 15]])

        # TODO
        out = None
        raise NotImplementedError("TODO 8.1")

        self.assertEqual(out.shape[1], 7)

    def test_ex_02_no_grad_during_inference(self):
        """
        练习 8.2: 推理时用 torch.no_grad() 节省显存。
        验证在 no_grad 上下文中，logits 没有梯度。
        """
        idx = torch.randint(0, self.cfg["vocab_size"], (1, 4))

        with torch.no_grad():
            logits = self.model(idx)

        # TODO
        has_grad = None
        raise NotImplementedError("TODO 8.2")
        self.assertFalse(has_grad)


# ═════════════════════════════════════════════════════════════════════════════
# 第 9 节  面试题集中考察
# ═════════════════════════════════════════════════════════════════════════════
class Section9_InterviewQuestions(unittest.TestCase):
    """
    Google MLE 面试高频题目
    """

    def test_iq_01_why_layernorm_not_batchnorm(self):
        """
        【面试题】LayerNorm vs BatchNorm
        LLM 用 LayerNorm 的原因：
        1. 序列长度可变，BatchNorm 统计量不稳定
        2. 推理时 batch_size=1，BatchNorm 退化
        3. LayerNorm 每个样本独立，不依赖 batch

        验证：batch_size=1 时，LayerNorm 正常工作
        """
        ln = LayerNorm(4)
        x = torch.randn(1, 4)  # batch_size=1
        out = ln(x)
        self.assertFalse(torch.isnan(out).any())

        bn = nn.BatchNorm1d(4)
        bn.eval()  # eval 模式才能处理 batch=1
        out_bn = bn(x)
        self.assertFalse(torch.isnan(out_bn).any())

    def test_iq_02_gelu_vs_relu_dead_neuron(self):
        """
        【面试题】GELU 解决 ReLU 的"神经元死亡"问题
        ReLU(x<0) = 0 → 梯度为 0 → 权重永远不更新 → 神经元"死"了
        GELU(x<0) ≠ 0 → 梯度不为 0 → 权重仍然更新
        """
        gelu = GELU()
        relu = nn.ReLU()

        x = torch.tensor([-2.0, -1.0, -0.5])
        self.assertTrue((relu(x) == 0).all())  # ReLU 全部截断为 0
        self.assertFalse((gelu(x) == 0).any())  # GELU 都不为 0

    def test_iq_03_residual_connection_gradient_flow(self):
        """
        【面试题】残差连接的梯度流动
        output = f(x) + x
        ∂output/∂x = ∂f(x)/∂x + 1
        即使 ∂f/∂x 很小，"+1" 保证梯度 ≥ 1
        """
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        f_x = x * 0.0001  # 极小的变换，模拟梯度消失
        out = f_x + x  # 残差连接

        out.sum().backward()
        # 梯度 = 0.0001 + 1 ≈ 1.0001，不会消失
        self.assertTrue(torch.allclose(x.grad, torch.ones(3) * 1.0001, atol=1e-5))

    def test_iq_04_weight_tying_saves_params(self):
        """
        【面试题】Weight Tying 的参数节省
        tok_emb 和 out_head 共享权重：
        节省参数 = vocab_size * emb_dim = 50257 * 768 ≈ 38.6M
        """
        cfg = GPT_CONFIG_SMALL
        saved = cfg["vocab_size"] * cfg["emb_dim"]
        # 节省的参数 = 100 * 32 = 3200
        self.assertEqual(saved, 100 * 32)

    def test_iq_05_logits_to_next_token(self):
        """
        【面试题】为什么用 logits[:, -1, :] 而不是 logits[:, 0, :]？

        因果注意力下，位置 t 的输出只取决于位置 0..t 的输入。
        位置 T-1 的输出融合了整个序列的信息，是对"下一个词"的预测。
        位置 0 的输出只看到自己，信息最少。
        """
        torch.manual_seed(0)
        model = GPTModel(GPT_CONFIG_SMALL)
        model.eval()

        idx = torch.randint(0, 100, (1, 5))
        logits = model(idx)

        # 取最后一个位置的 logits 做预测
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        self.assertEqual(next_token.shape, torch.Size([1]))
        self.assertGreaterEqual(next_token.item(), 0)
        self.assertLess(next_token.item(), 100)

    def test_iq_06_transformer_block_shape_invariant(self):
        """
        【面试题】TransformerBlock 的形状不变性是堆叠 n_layers 层的基础。
        输入输出形状完全相同，才能用 nn.Sequential 串联。
        """
        cfg = GPT_CONFIG_SMALL
        blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        blocks.eval()

        x = torch.randn(2, 8, cfg["emb_dim"])
        out = blocks(x)
        self.assertEqual(out.shape, x.shape)

    def test_iq_07_ffn_4x_expansion_reason(self):
        """
        【面试题】FFN 扩张 4 倍的意义
        中间层 [batch, T, 4*emb_dim] 提供更大的特征空间，
        让模型能学习更复杂的非线性变换，然后再压缩回 emb_dim
        """
        cfg = GPT_CONFIG_SMALL
        ffn = FeedForward(cfg)

        # 验证第一层扩张 4 倍，第二层压缩回来
        first_linear = ffn.layers[0]
        second_linear = ffn.layers[2]

        self.assertEqual(first_linear.out_features, 4 * cfg["emb_dim"])
        self.assertEqual(second_linear.out_features, cfg["emb_dim"])


# ═════════════════════════════════════════════════════════════════════════════
# 运行入口
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("Ch04: GPT Model 互动学习测试")
    print("  Section1: GPT 配置 & 架构")
    print("  Section2: LayerNorm（vs BatchNorm）")
    print("  Section3: GELU（vs ReLU）")
    print("  Section4: FeedForward（4x 扩展）")
    print("  Section5: 残差连接（梯度消失）")
    print("  Section6: TransformerBlock（Pre-Norm）")
    print("  Section7: GPTModel（参数量 & Weight Tying）")
    print("  Section8: 文本生成（Greedy Decoding）")
    print("  Section9: 面试题集中考察")
    print("=" * 65)
    unittest.main(verbosity=2)
