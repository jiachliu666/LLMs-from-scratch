"""
Ch03: Attention Mechanisms — 互动学习文件
==========================================
运行方式:
    python -m pytest learn_ch03.py -v
    python -m pytest learn_ch03.py -v -s        # 显示 print 输出
    python -m pytest learn_ch03.py -v -k demo   # 只跑演示

知识地图（Google MLE 面试级别）:
  Section1: 简单自注意力（无可训练权重）— dot product / softmax / context vector
  Section2: 缩放点积注意力（有 W_q, W_k, W_v）— 为什么要除以 sqrt(d_k)
  Section3: SelfAttention 封装类
  Section4: 因果掩码（Causal Mask）— 为什么用 -inf 而不是 0
  Section5: Dropout in Attention
  Section6: CausalAttention（带 batch 维度）
  Section7: MultiHeadAttention — 两种实现方式及形状变换
"""

import unittest
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 共享数据：6 个词的 3 维 embedding
# ─────────────────────────────────────────────────────────────────────────────
INPUTS = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],  # step     (x^6)
    ]
)  # shape [6, 3]


# ═════════════════════════════════════════════════════════════════════════════
# 第 1 节  简单自注意力（无可训练权重）
# ═════════════════════════════════════════════════════════════════════════════
class Section1_SimpleAttention(unittest.TestCase):
    """
    【概念】最朴素的自注意力：直接用输入向量的点积作为注意力分数。

    步骤:
      1. attn_scores[i,j] = dot(x_i, x_j)  → 全矩阵 = inputs @ inputs.T
      2. attn_weights     = softmax(attn_scores, dim=-1)  → 每行和为 1
      3. context_vec      = attn_weights @ inputs          → 加权求和

    【面试考点】
    - 点积衡量相似度：两个向量越相似，点积越大，注意力权重越高
    - softmax 把分数转成概率分布（和为 1）
    - context vector 是所有输入的加权平均，融合了上下文信息
    """

    def setUp(self):
        self.inputs = INPUTS.clone()
        self.attn_scores = self.inputs @ self.inputs.T  # [6, 6]
        self.attn_weights = torch.softmax(self.attn_scores, dim=-1)

    # ── 演示测试 ──────────────────────────────────────────────────────────────

    def test_demo_01_attn_scores_shape(self):
        """注意力分数矩阵：[T, T]，T=序列长度"""
        self.assertEqual(self.attn_scores.shape, torch.Size([6, 6]))

    def test_demo_02_attn_scores_symmetric(self):
        """无权重的注意力分数矩阵是对称的（inputs @ inputs.T = (inputs @ inputs.T).T）"""
        self.assertTrue(torch.allclose(self.attn_scores, self.attn_scores.T))

    def test_demo_03_attn_weights_sum_to_one(self):
        """softmax 后每行和为 1"""
        row_sums = self.attn_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(6), atol=1e-6))

    def test_demo_04_context_vec_shape(self):
        """context vector 形状与输入相同：[T, d_in]"""
        context_vecs = self.attn_weights @ self.inputs
        self.assertEqual(context_vecs.shape, self.inputs.shape)

    def test_demo_05_context_vec_2_manual(self):
        """手动计算 x^(2) 的 context vector，与矩阵乘法结果一致"""
        query = self.inputs[1]  # x^(2)
        attn_scores_2 = torch.tensor([torch.dot(x_i, query) for x_i in self.inputs])
        attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
        context_vec_2_manual = (attn_weights_2.unsqueeze(-1) * self.inputs).sum(dim=0)

        all_context_vecs = self.attn_weights @ self.inputs
        self.assertTrue(torch.allclose(context_vec_2_manual, all_context_vecs[1], atol=1e-5))

    # ── 练习测试 ──────────────────────────────────────────────────────────────

    def test_ex_01_dot_product_measures_similarity(self):
        """
        练习 1.1: 验证 x^(2) 和 x^(3) 的点积 > x^(2) 和 x^(5) 的点积。
        （journey 和 starts 更相似，one 更不同）

        提示: torch.dot(inputs[1], inputs[2]) vs torch.dot(inputs[1], inputs[4])
        """
        # TODO
        sim_23 = torch.dot(self.inputs[1], self.inputs[2])  # ← torch.dot(self.inputs[1], self.inputs[2])
        sim_25 = torch.dot(self.inputs[1], self.inputs[4])  # ← torch.dot(self.inputs[1], self.inputs[4])
        # raise NotImplementedError("TODO 1.1")
        self.assertGreater(sim_23.item(), sim_25.item())

    def test_ex_02_compute_attn_scores_via_matmul(self):
        """
        练习 1.2: 用矩阵乘法计算完整注意力分数，验证 [1,2] 位置等于 dot(x^2, x^3)
        """
        # TODO
        attn_scores = self.inputs @ self.inputs.T  # ← self.inputs @ self.inputs.T
        # raise NotImplementedError("TODO 1.2")
        expected = torch.dot(self.inputs[1], self.inputs[2])
        self.assertTrue(torch.allclose(attn_scores[1, 2], expected))

    def test_ex_03_context_is_weighted_average(self):
        """
        练习 1.3: context vector = 注意力权重对输入的加权平均。
        手动验证 context_vecs[0] == sum(attn_weights[0, i] * inputs[i] for i in range(6))
        """
        context_vecs = self.attn_weights @ self.inputs

        # TODO: 手动计算第 0 行的 context vector
        manual_cv0 = sum(self.attn_weights[0, i] * self.inputs[i] for i in range(6))  # ← sum of attn_weights[0, i] * inputs[i]
        # raise NotImplementedError("TODO 1.3")
        self.assertTrue(torch.allclose(manual_cv0, context_vecs[0], atol=1e-5))


# ═════════════════════════════════════════════════════════════════════════════
# 第 2 节  缩放点积注意力（有可训练权重 W_q, W_k, W_v）
# ═════════════════════════════════════════════════════════════════════════════
class Section2_ScaledDotProductAttention(unittest.TestCase):
    """
    【概念】真正的 Transformer 自注意力。引入三个可训练矩阵：

      q = x @ W_q   ← query：我想找什么
      k = x @ W_k   ← key：我能提供什么
      v = x @ W_v   ← value：我实际提供的内容

      attn_scores  = q @ k.T / sqrt(d_k)   ← 除以 sqrt(d_k) 防止梯度消失
      attn_weights = softmax(attn_scores)
      context_vec  = attn_weights @ v

    【面试考点】
    - 为什么除以 sqrt(d_k)？
      d_k 越大，点积方差越大，softmax 梯度趋近 0（梯度消失）。
      除以 sqrt(d_k) 把方差拉回 1，保持梯度稳定。

    - Q/K/V 分离的意义：
      让模型学习"查询方向"、"匹配方向"、"输出方向"三者独立，表达能力更强。
    """

    def setUp(self):
        torch.manual_seed(123)
        self.inputs = INPUTS.clone()
        self.d_in = 3
        self.d_out = 2
        self.W_query = nn.Parameter(torch.rand(self.d_in, self.d_out), requires_grad=False)
        self.W_key = nn.Parameter(torch.rand(self.d_in, self.d_out), requires_grad=False)
        self.W_value = nn.Parameter(torch.rand(self.d_in, self.d_out), requires_grad=False)

        self.queries = self.inputs @ self.W_query  # [6, 2]
        self.keys = self.inputs @ self.W_key  # [6, 2]
        self.values = self.inputs @ self.W_value  # [6, 2]

    # ── 演示测试 ──────────────────────────────────────────────────────────────

    def test_demo_01_projection_shape(self):
        """x @ W_q 把 d_in=3 投影到 d_out=2"""
        self.assertEqual(self.queries.shape, torch.Size([6, 2]))
        self.assertEqual(self.keys.shape, torch.Size([6, 2]))
        self.assertEqual(self.values.shape, torch.Size([6, 2]))

    def test_demo_02_attn_scores_scaled(self):
        """注意力分数 = queries @ keys.T / sqrt(d_k)"""
        d_k = self.keys.shape[-1]
        attn_scores = self.queries @ self.keys.T / d_k**0.5
        self.assertEqual(attn_scores.shape, torch.Size([6, 6]))

    def test_demo_03_scaling_reduces_variance(self):
        """
        【面试核心】sqrt(d_k) 缩放的作用：
        未缩放的 attn_scores 方差更大 → softmax 梯度趋近 0
        缩放后方差减小 → 梯度更健康
        """
        d_k = self.keys.shape[-1]
        raw = self.queries @ self.keys.T
        scaled = raw / d_k**0.5
        self.assertGreater(raw.var().item(), scaled.var().item())

    def test_demo_04_context_vec_shape(self):
        """context vector 形状：[T, d_out]"""
        d_k = self.keys.shape[-1]
        attn_scores = self.queries @ self.keys.T / d_k**0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_vec = attn_weights @ self.values
        self.assertEqual(context_vec.shape, torch.Size([6, 2]))

    # ── 练习测试 ──────────────────────────────────────────────────────────────

    def test_ex_01_query_key_value_shapes(self):
        """
        练习 2.1: 给定 d_in=3, d_out=4，
                  W_q, W_k, W_v 的形状是？
                  inputs @ W_q 的形状是？（inputs 是 [6,3]）
        """
        d_out = 4
        W = torch.rand(self.d_in, d_out)
        proj = self.inputs @ W

        # TODO
        expected_W_shape = torch.Size([3, 4])
        expected_proj_shape = torch.Size([6, 4])
        # raise NotImplementedError("TODO 2.1")
        self.assertEqual(W.shape, expected_W_shape)
        self.assertEqual(proj.shape, expected_proj_shape)

    def test_ex_02_compute_full_scaled_attn(self):
        """
        练习 2.2: 完整计算缩放点积注意力（所有 token），
                  验证 attn_weights 每行和为 1，context_vecs shape=[6,2]
        """
        # TODO
        d_k = self.keys.shape[-1]
        attn_scores = self.queries @ self.keys.T / d_k**0.5
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_vecs = attn_weights @ self.values

        self.assertTrue(torch.allclose(attn_weights.sum(dim=-1), torch.ones(6), atol=1e-5))
        self.assertEqual(context_vecs.shape, torch.Size([6, 2]))

    def test_ex_03_why_divide_by_sqrt_dk(self):
        """
        练习 2.3: 【面试题】用随机大矩阵验证缩放的必要性。
        d_k=64 时，未缩放的 softmax 输出接近 one-hot（梯度消失），
        缩放后分布更均匀（熵更高）。
        """
        torch.manual_seed(0)
        d_k = 64
        q = torch.randn(1, d_k)
        k = torch.randn(10, d_k)

        scores_raw = q @ k.T  # 未缩放
        scores_scaled = q @ k.T / d_k**0.5  # 缩放

        w_raw = torch.softmax(scores_raw, dim=-1)
        w_scaled = torch.softmax(scores_scaled, dim=-1)

        # 未缩放的熵（均匀性）更低（更接近 one-hot）
        entropy = lambda w: -(w * (w + 1e-9).log()).sum()

        # TODO: 验证缩放后熵更高
        # raise NotImplementedError("TODO 2.3: 取消注释下面一行")
        self.assertGreater(entropy(w_scaled).item(), entropy(w_raw).item())


# ═════════════════════════════════════════════════════════════════════════════
# 第 3 节  SelfAttention 封装类
# ═════════════════════════════════════════════════════════════════════════════
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ values


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ values


class Section3_SelfAttentionClass(unittest.TestCase):
    """
    【概念】把自注意力封装成 nn.Module。

    v1: nn.Parameter(torch.rand(...))  — 手动权重
    v2: nn.Linear(bias=False)          — 更规范，初始化更稳定

    【面试考点】
    - v2 用 nn.Linear 的好处：Kaiming 初始化，梯度流更健康
    - 两者数学等价（bias=False 时 Linear 就是矩阵乘法）
    """

    def setUp(self):
        self.inputs = INPUTS.clone()
        self.d_in, self.d_out = 3, 2

    def test_demo_01_v1_output_shape(self):
        """SelfAttention_v1 输出形状 [T, d_out]"""
        torch.manual_seed(123)
        sa = SelfAttention_v1(self.d_in, self.d_out)
        out = sa(self.inputs)
        self.assertEqual(out.shape, torch.Size([6, 2]))

    def test_demo_02_v2_output_shape(self):
        """SelfAttention_v2 输出形状 [T, d_out]"""
        torch.manual_seed(123)
        sa = SelfAttention_v2(self.d_in, self.d_out)
        out = sa(self.inputs)
        self.assertEqual(out.shape, torch.Size([6, 2]))

    def test_demo_03_v1_v2_same_weights_same_output(self):
        """v1 和 v2 权重相同时输出相同（验证数学等价）"""
        torch.manual_seed(42)
        sa1 = SelfAttention_v1(self.d_in, self.d_out)
        sa2 = SelfAttention_v2(self.d_in, self.d_out, qkv_bias=False)

        # 把 v1 的权重复制给 v2（注意 Linear.weight 是转置存储的）
        sa2.W_query.weight = nn.Parameter(sa1.W_query.T)
        sa2.W_key.weight = nn.Parameter(sa1.W_key.T)
        sa2.W_value.weight = nn.Parameter(sa1.W_value.T)

        out1 = sa1(self.inputs)
        out2 = sa2(self.inputs)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_ex_01_num_parameters(self):
        """
        练习 3.1: d_in=3, d_out=2 时，SelfAttention_v2 有多少个可训练参数？

        提示: 3 个 Linear(3, 2, bias=False)，每个 3*2=6 个参数
        """
        torch.manual_seed(0)
        sa = SelfAttention_v2(self.d_in, self.d_out)
        total = sum(p.numel() for p in sa.parameters())

        # TODO
        expected = 18
        # raise NotImplementedError("TODO 3.1")
        self.assertEqual(total, expected)

    def test_ex_02_larger_d_out(self):
        """
        练习 3.2: d_out=8 时，输出形状是什么？
        """
        torch.manual_seed(0)
        sa = SelfAttention_v2(self.d_in, d_out=8)
        out = sa(self.inputs)

        # TODO
        expected_shape = torch.Size([6, 8])  # ← torch.Size([6, 8])
        # raise NotImplementedError("TODO 3.2")
        self.assertEqual(out.shape, expected_shape)


# ═════════════════════════════════════════════════════════════════════════════
# 第 4 节  因果掩码（Causal Mask）
# ═════════════════════════════════════════════════════════════════════════════
class Section4_CausalMask(unittest.TestCase):
    """
    【概念】GPT 是 decoder-only 模型，生成 token t 时不能看到 t+1, t+2...。
    因此在注意力矩阵中，把"未来位置"屏蔽掉。

    两种掩码方式（等价）：
      方式 A（低效）: softmax 后把上三角置 0，再重新归一化
      方式 B（推荐）: softmax 前把上三角置 -inf，softmax 自动把它变成 0

    【面试考点】
    - 为什么用 -inf 而不是直接置 0？
      softmax 后置 0 破坏了"和为 1"的约束，需要额外重归一化。
      -inf 进 softmax 直接输出 0，且剩余位置自动重归一化。

    - torch.tril vs torch.triu：
      tril = lower triangular (下三角含对角线) → 保留哪些位置
      triu = upper triangular (上三角不含对角线, diagonal=1) → 屏蔽哪些位置
    """

    def setUp(self):
        self.T = 6
        self.attn_scores = torch.randn(self.T, self.T)

    def test_demo_01_tril_mask_shape(self):
        """torch.tril 生成下三角矩阵"""
        mask = torch.tril(torch.ones(self.T, self.T))
        # 对角线及以下全是 1
        self.assertEqual(mask[0, 0].item(), 1.0)
        self.assertEqual(mask[0, 1].item(), 0.0)
        self.assertEqual(mask[5, 5].item(), 1.0)

    def test_demo_02_triu_upper_triangular(self):
        """torch.triu(diagonal=1) 生成上三角（不含对角线）"""
        mask = torch.triu(torch.ones(self.T, self.T), diagonal=1)
        self.assertEqual(mask[0, 0].item(), 0.0)  # 对角线是 0
        self.assertEqual(mask[0, 1].item(), 1.0)  # 上三角是 1

    def test_demo_03_mask_with_neg_inf_before_softmax(self):
        """
        方式 B（推荐）：-inf 在 softmax 前掩码，结果上三角自动变 0，行和为 1
        """
        T = self.T
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        masked = self.attn_scores.masked_fill(mask, -torch.inf)
        weights = torch.softmax(masked, dim=-1)

        # 上三角应为 0
        self.assertTrue((weights[0, 1:] == 0).all())
        # 行和为 1
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(T), atol=1e-5))

    def test_demo_04_mask_after_softmax_breaks_row_sum(self):
        """
        方式 A（低效）：softmax 后直接置 0，行和不再为 1
        """
        weights = torch.softmax(self.attn_scores, dim=-1)
        tril = torch.tril(torch.ones(self.T, self.T))
        masked = weights * tril

        # 除第一行（只有 1 个元素）外，其余行和 < 1
        row_sums = masked.sum(dim=-1)
        self.assertFalse(torch.allclose(row_sums, torch.ones(self.T), atol=1e-5))

    def test_demo_05_first_token_attends_only_itself(self):
        """因果掩码后，第一个 token 只能关注自己（attn_weights[0] = [1, 0, 0, ...]）"""
        mask = torch.triu(torch.ones(self.T, self.T), diagonal=1).bool()
        masked = self.attn_scores.masked_fill(mask, -torch.inf)
        weights = torch.softmax(masked, dim=-1)
        self.assertAlmostEqual(weights[0, 0].item(), 1.0, places=5)
        self.assertTrue((weights[0, 1:] == 0).all())

    def test_ex_01_implement_causal_mask(self):
        """
        练习 4.1: 给定 attn_scores [4, 4]，手动实现因果掩码（-inf 方式），
                  验证第 2 行（index=1）的权重在 index 2, 3 位置为 0。
        """
        attn_scores = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [0.5, 1.5, 2.5, 3.5],
                [0.2, 0.3, 0.8, 1.2],
                [0.1, 0.4, 0.6, 0.9],
            ]
        )
        mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()
        masked = attn_scores.masked_fill(mask, -torch.inf)

        # TODO: 创建 mask，填充 -inf，再 softmax
        weights = torch.softmax(masked, dim=-1)
        # raise NotImplementedError("TODO 4.1")

        self.assertAlmostEqual(weights[1, 2].item(), 0.0, places=5)
        self.assertAlmostEqual(weights[1, 3].item(), 0.0, places=5)
        self.assertGreater(weights[1, 0].item(), 0.0)

    def test_ex_02_why_neg_inf_not_zero(self):
        """
        练习 4.2: 【面试题】用数值验证：softmax(-inf) = 0，softmax(0) != 0。
        如果直接把分数置 0（不是 -inf），softmax 后该位置不为 0。
        """
        scores = torch.tensor([1.0, 0.0, -torch.inf])  # 第 2 位用 -inf

        # TODO
        weights = torch.softmax(scores, dim=0)  # ← torch.softmax(scores, dim=0)
        # raise NotImplementedError("TODO 4.2")

        self.assertAlmostEqual(weights[2].item(), 0.0, places=5)
        self.assertGreater(weights[1].item(), 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 第 5 节  Dropout in Attention
# ═════════════════════════════════════════════════════════════════════════════
class Section5_AttentionDropout(unittest.TestCase):
    """
    【概念】Dropout 在注意力权重上随机置 0，防止模型过拟合于固定的注意力模式。

    关键细节：
    - 被 dropout 的位置置 0
    - 未被 dropout 的位置乘以 1/(1-p) 来保持期望值不变
    - dropout 只在训练时激活，推理时关闭（model.eval()）

    【面试考点】
    - 为什么对注意力权重做 dropout 而不对分数做？
      分数通过 softmax 会重新归一化，dropout 在权重上更直接有效。
    """

    def test_demo_01_dropout_scales_surviving_values(self):
        """dropout=0.5 时，存活的值被乘以 2（=1/(1-0.5)）"""
        torch.manual_seed(123)
        dropout = nn.Dropout(0.5)
        x = torch.ones(4, 4)
        out = dropout(x)
        # 存活的值全是 2.0，被 dropout 的是 0.0
        surviving = out[out != 0]
        self.assertTrue((surviving == 2.0).all())

    def test_demo_02_dropout_disabled_in_eval(self):
        """model.eval() 模式下 dropout 不生效"""
        dropout = nn.Dropout(0.5)
        dropout.eval()
        x = torch.ones(4, 4)
        out = dropout(x)
        self.assertTrue(torch.equal(out, x))

    def test_demo_03_dropout_preserves_expected_value(self):
        """大量样本下，dropout 前后均值相同（期望值不变）"""
        torch.manual_seed(0)
        dropout = nn.Dropout(0.5)
        x = torch.ones(1000, 1000)
        out = dropout(x)
        self.assertAlmostEqual(out.mean().item(), 1.0, delta=0.05)

    def test_ex_01_apply_dropout_to_attn_weights(self):
        """
        练习 5.1: 对 [4,4] 的注意力权重（已 softmax）施加 dropout=0.5，
                  验证输出中存在 0（有些权重被 drop 掉）
        """
        torch.manual_seed(42)
        attn_weights = torch.softmax(torch.randn(4, 4), dim=-1)
        dp = nn.Dropout(0.5)
        # TODO
        out = dp(attn_weights)
        # raise NotImplementedError("TODO 5.1")
        self.assertTrue((out == 0).any())


# ═════════════════════════════════════════════════════════════════════════════
# 第 6 节  CausalAttention（支持 batch）
# ═════════════════════════════════════════════════════════════════════════════
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # self.W_key = [d_in, d_out]
        # x = [b, nt, d_in]
        keys = self.W_key(x)
        # x * W_key = [b, nt, d_out]
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        # attn_scores = [b, nt, d_out] * [b, d_out, nt ] = [b, nt, nt]
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # [b, nt, nt] @ [b, nt, d_out] = [b, nt, d_out]
        return attn_weights @ values


class Section6_CausalAttentionBatch(unittest.TestCase):
    """
    【概念】真实 LLM 中的因果注意力：支持 batch 维度 [b, T, d_in]。

    关键形状变换:
      输入:  [b, T, d_in]
      Q/K/V: [b, T, d_out]
      scores: [b, T, T]  （keys.transpose(1,2) 转置最后两维）
      output: [b, T, d_out]

    【面试考点】
    - keys.transpose(1,2) vs keys.T
      有 batch 时不能用 .T（会转置所有维度），要用 transpose(1,2) 只转置 T 和 d_out
    - register_buffer vs nn.Parameter
      buffer 不参与梯度计算，随模型 .to(device) 移动，适合 mask 这类固定张量
    """

    def setUp(self):
        self.inputs = INPUTS.clone()
        self.batch = torch.stack([self.inputs, self.inputs], dim=0)  # [2, 6, 3]
        self.d_in, self.d_out = 3, 2
        self.context_length = 6

    def test_demo_01_batch_shape(self):
        """batch = stack 两个相同输入，shape [2, 6, 3]"""
        self.assertEqual(self.batch.shape, torch.Size([2, 6, 3]))

    def test_demo_02_causal_attention_output_shape(self):
        """CausalAttention 输出 [b, T, d_out]"""
        torch.manual_seed(123)
        ca = CausalAttention(self.d_in, self.d_out, self.context_length, dropout=0.0)
        out = ca(self.batch)
        self.assertEqual(out.shape, torch.Size([2, 6, 2]))

    def test_demo_03_same_input_same_output_across_batch(self):
        """两个相同输入产生相同输出（batch 内独立）"""
        torch.manual_seed(123)
        ca = CausalAttention(self.d_in, self.d_out, self.context_length, dropout=0.0)
        out = ca(self.batch)
        self.assertTrue(torch.allclose(out[0], out[1]))

    def test_demo_04_causal_property(self):
        """
        因果性验证：改变第 3 个 token，只影响第 3 个及之后的输出，
        不影响第 0、1、2 个 token 的 context vector。
        """
        torch.manual_seed(123)
        ca = CausalAttention(self.d_in, self.d_out, self.context_length, dropout=0.0)

        x1 = self.batch.clone()
        x2 = self.batch.clone()
        x2[:, 3, :] = torch.randn(3)  # 修改第 3 个 token

        out1 = ca(x1)
        out2 = ca(x2)

        # token 0,1,2 不受影响
        self.assertTrue(torch.allclose(out1[:, :3, :], out2[:, :3, :], atol=1e-5))
        # token 3,4,5 受影响
        self.assertFalse(torch.allclose(out1[:, 3:, :], out2[:, 3:, :]))

    def test_ex_01_transpose_for_batch(self):
        """
        练习 6.1: 【面试题】有 batch 时为什么不能用 keys.T？

        验证：对 shape [2, 6, 2] 的张量：
          .T              → shape [2, 6, 2]（把所有维度反转：变成 [2, 6, 2]，错误）
          .transpose(1,2) → shape [2, 2, 6]（只转置最后两维，正确）
        """
        keys = torch.randn(2, 6, 2)

        # TODO
        wrong = None  # ← keys.T
        correct = None  # ← keys.transpose(1, 2)
        raise NotImplementedError("TODO 6.1")

        self.assertEqual(wrong.shape, torch.Size([2, 6, 2]))
        self.assertEqual(correct.shape, torch.Size([2, 2, 6]))

    def test_ex_02_output_shape_different_d_out(self):
        """
        练习 6.2: d_out=8 时，CausalAttention 输出形状是？
        """
        torch.manual_seed(0)
        ca = CausalAttention(self.d_in, d_out=8, context_length=self.context_length, dropout=0.0)
        out = ca(self.batch)

        # TODO
        expected_shape = None  # ← torch.Size([2, 6, 8])
        raise NotImplementedError("TODO 6.2")
        self.assertEqual(out.shape, expected_shape)


# ═════════════════════════════════════════════════════════════════════════════
# 第 7 节  多头注意力（Multi-Head Attention）
# ═════════════════════════════════════════════════════════════════════════════
class MultiHeadAttentionWrapper(nn.Module):
    """简单实现：多个 CausalAttention 并行 + concat"""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """高效实现：单套 W_q/W_k/W_v，内部 view 拆分 heads"""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
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
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 拆分 heads: [b, T, d_out] → [b, T, num_heads, head_dim] → [b, num_heads, T, head_dim]
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)  # [b, num_heads, T, T]
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # [b, num_heads, T, head_dim] → [b, T, num_heads, head_dim] → [b, T, d_out]
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous()
        context_vec = context_vec.view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class Section7_MultiHeadAttention(unittest.TestCase):
    """
    【概念】多头注意力：并行运行多个注意力头，捕获不同子空间的依赖关系。

    两种实现：
      Wrapper: n 个独立 CausalAttention 输出 concat → 输出 [b, T, d_out * n_heads]
      Efficient: 单套权重内部 view 拆分 → 输出 [b, T, d_out]（更常用）

    形状变换（Efficient 实现）:
      输入:      [b, T, d_in]
      W_q/k/v:   [b, T, d_out]
      view:      [b, T, n_heads, head_dim]
      transpose: [b, n_heads, T, head_dim]
      scores:    [b, n_heads, T, T]
      output:    [b, T, d_out]  （经过 out_proj）

    【面试考点】
    - 多头的本质：让不同的 head 学习不同的"关注模式"
    - head_dim = d_out // n_heads（每个 head 的维度更小）
    - out_proj 把多个 head 的结果线性混合
    - Wrapper 和 Efficient 数学等价，但 Efficient 可以 batch 矩阵乘，更快
    """

    def setUp(self):
        self.inputs = INPUTS.clone()
        self.batch = torch.stack([self.inputs, self.inputs], dim=0)  # [2, 6, 3]
        self.d_in = 3
        self.context_length = 6

    def test_demo_01_wrapper_output_shape(self):
        """Wrapper: d_out=2, num_heads=2 → output [b, T, d_out*num_heads] = [2,6,4]"""
        torch.manual_seed(123)
        mha = MultiHeadAttentionWrapper(self.d_in, d_out=2, context_length=6, dropout=0.0, num_heads=2)
        out = mha(self.batch)
        self.assertEqual(out.shape, torch.Size([2, 6, 4]))

    def test_demo_02_efficient_output_shape(self):
        """Efficient: d_out=4, num_heads=2 → output [b, T, d_out] = [2,6,4]"""
        torch.manual_seed(123)
        mha = MultiHeadAttention(self.d_in, d_out=4, context_length=6, dropout=0.0, num_heads=2)
        out = mha(self.batch)
        self.assertEqual(out.shape, torch.Size([2, 6, 4]))

    def test_demo_03_head_dim_calculation(self):
        """head_dim = d_out // num_heads"""
        mha = MultiHeadAttention(self.d_in, d_out=8, context_length=6, dropout=0.0, num_heads=4)
        self.assertEqual(mha.head_dim, 2)

    def test_demo_04_view_split_heads(self):
        """
        view + transpose 把 [b, T, d_out] 拆成 [b, num_heads, T, head_dim]
        """
        b, T, d_out, num_heads = 2, 6, 4, 2
        head_dim = d_out // num_heads
        x = torch.randn(b, T, d_out)
        split = x.view(b, T, num_heads, head_dim).transpose(1, 2)
        self.assertEqual(split.shape, torch.Size([b, num_heads, T, head_dim]))

    def test_demo_05_batch_matmul_per_head(self):
        """
        [b, n_heads, T, head_dim] @ [b, n_heads, head_dim, T]
        → scores [b, n_heads, T, T]
        PyTorch 对最后两维做矩阵乘，对前两维做 batch
        """
        b, n_heads, T, head_dim = 1, 2, 3, 4
        q = torch.randn(b, n_heads, T, head_dim)
        k = torch.randn(b, n_heads, T, head_dim)
        scores = q @ k.transpose(2, 3)
        self.assertEqual(scores.shape, torch.Size([b, n_heads, T, T]))

    def test_ex_01_d_out_must_divisible_by_num_heads(self):
        """
        练习 7.1: d_out=5, num_heads=2 时应该抛出 AssertionError
        """
        # TODO
        # raise NotImplementedError("TODO 7.1: 取消注释下面代码")
        with self.assertRaises(AssertionError):
            MultiHeadAttention(self.d_in, d_out=5, context_length=6, dropout=0.0, num_heads=2)

    def test_ex_02_wrapper_vs_efficient_output_dim(self):
        """
        练习 7.2: 【面试题】Wrapper(d_out=2, num_heads=2) 和
                  Efficient(d_out=4, num_heads=2) 的输出维度相同吗？为什么？

        验证两者输出 shape 的最后一维都是 4。
        """
        torch.manual_seed(0)
        wrapper = MultiHeadAttentionWrapper(self.d_in, d_out=2, context_length=6, dropout=0.0, num_heads=2)
        efficient = MultiHeadAttention(self.d_in, d_out=4, context_length=6, dropout=0.0, num_heads=2)
        out_w = wrapper(self.batch)
        out_e = efficient(self.batch)

        # TODO: 验证两者最后一维都是 4
        # raise NotImplementedError("TODO 7.2: 取消注释下面两行")
        self.assertEqual(out_w.shape[-1], 4)
        self.assertEqual(out_e.shape[-1], 4)

    def test_ex_03_num_heads_4_output_shape(self):
        """
        练习 7.3: d_in=3, d_out=8, num_heads=4, batch=[2,6,3]，
                  MultiHeadAttention 输出 shape 是？
        """
        torch.manual_seed(0)
        mha = MultiHeadAttention(self.d_in, d_out=8, context_length=6, dropout=0.0, num_heads=4)
        out = mha(self.batch)

        # TODO
        expected_shape = None  # ← torch.Size([2, 6, 8])
        raise NotImplementedError("TODO 7.3")
        self.assertEqual(out.shape, expected_shape)

    def test_ex_04_gpt2_style_config(self):
        """
        练习 7.4: 模拟 GPT-2 small 的多头注意力配置：
                  d_model=768, num_heads=12, context_length=1024
                  输入 [1, 10, 768]，输出 shape 是？
        """
        torch.manual_seed(0)
        d_model = 768
        num_heads = 12
        context_length = 1024
        mha = MultiHeadAttention(d_model, d_model, context_length, dropout=0.0, num_heads=num_heads)
        x = torch.randn(1, 10, d_model)
        out = mha(x)

        # TODO
        expected_shape = None  # ← torch.Size([1, 10, 768])
        raise NotImplementedError("TODO 7.4")
        self.assertEqual(out.shape, expected_shape)


# ═════════════════════════════════════════════════════════════════════════════
# 综合练习：面试题集中考察
# ═════════════════════════════════════════════════════════════════════════════
class Section8_InterviewQuestions(unittest.TestCase):
    """
    【Google MLE 面试常见题】

    Q1: 为什么要除以 sqrt(d_k)？
    Q2: 为什么掩码用 -inf 而不是 0？
    Q3: 多头注意力的优势是什么？
    Q4: Causal mask 保证了什么？
    Q5: Self-attention 的时间复杂度是？
    """

    def test_iq_01_attn_complexity(self):
        """
        【面试题】Self-attention 对序列长度 T 的时间复杂度是 O(T^2 * d)。
        验证：注意力矩阵 scores 形状是 [T, T]，T 增大时计算量平方增长。
        """
        for T in [4, 8, 16]:
            inputs = torch.randn(T, 3)
            scores = inputs @ inputs.T
            self.assertEqual(scores.shape, torch.Size([T, T]))

    def test_iq_02_context_vector_is_not_just_self(self):
        """
        【面试题】context vector 不是 token 自身，而是融合了所有 token 的信息。
        验证：context_vec[i] != inputs[i]（注意力引入了其他 token 的信息）
        """
        inputs = INPUTS.clone()
        attn_weights = torch.softmax(inputs @ inputs.T, dim=-1)
        context_vecs = attn_weights @ inputs
        # context vector 和原始 embedding 不同
        self.assertFalse(torch.allclose(context_vecs, inputs))

    def test_iq_03_causal_mask_makes_first_token_attend_only_self(self):
        """
        【面试题】因果掩码下第一个 token 的 context vector == 第一个 token 的 value 向量。
        因为它只能关注自己，attn_weights[0] = [1, 0, 0, ...]
        """
        torch.manual_seed(0)
        d_in, d_out, T = 3, 4, 6
        ca = CausalAttention(d_in, d_out, context_length=T, dropout=0.0)
        ca.eval()

        batch = INPUTS.unsqueeze(0)  # [1, 6, 3]
        out = ca(batch)

        # 手动计算 value 的第一行
        with torch.no_grad():
            v0 = ca.W_value(INPUTS)[0]  # value of first token

        self.assertTrue(torch.allclose(out[0, 0], v0, atol=1e-5))

    def test_iq_04_multi_head_concatenation(self):
        """
        【面试题】Wrapper 实现的多头注意力是 concat，不是相加。
        2 heads, d_out=3 → 输出维度是 6，不是 3。
        """
        torch.manual_seed(0)
        batch = INPUTS.unsqueeze(0)
        mha = MultiHeadAttentionWrapper(d_in=3, d_out=3, context_length=6, dropout=0.0, num_heads=2)
        out = mha(batch)
        self.assertEqual(out.shape[-1], 6)  # concat: 3+3=6

    def test_iq_05_register_buffer_vs_parameter(self):
        """
        【面试题】register_buffer 的 mask 不是可训练参数，但会随模型移动设备。
        验证：mask 不在 parameters() 中，但在 buffers() 中。
        """
        ca = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.0)
        param_names = [n for n, _ in ca.named_parameters()]
        buffer_names = [n for n, _ in ca.named_buffers()]
        self.assertNotIn("mask", param_names)
        self.assertIn("mask", buffer_names)


# ═════════════════════════════════════════════════════════════════════════════
# 运行入口
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("Ch03: Attention Mechanisms 互动学习测试")
    print("  Section1: 简单自注意力（dot product + softmax）")
    print("  Section2: 缩放点积注意力（W_q/W_k/W_v + sqrt(d_k)）")
    print("  Section3: SelfAttention 封装类")
    print("  Section4: 因果掩码（-inf 技巧）")
    print("  Section5: Dropout in Attention")
    print("  Section6: CausalAttention（batch）")
    print("  Section7: MultiHeadAttention（两种实现）")
    print("  Section8: 面试题集中考察")
    print("=" * 65)
    unittest.main(verbosity=2)
