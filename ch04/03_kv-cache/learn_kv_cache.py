"""
Ch04 Bonus: KV Cache — 互动学习文件
=====================================
运行方式:
    python -m pytest learn_kv_cache.py -v
    python -m pytest learn_kv_cache.py -v -s   # 显示 print 输出
    python -m pytest learn_kv_cache.py -v -k demo

知识地图（面试级别）:
  Section1: 为什么需要 KV Cache？（复杂度分析）
  Section2: cache_k / cache_v 的形状变化
  Section3: 位置编码在 KV Cache 下的变化
  Section4: 有 cache vs 无 cache 输出一致性验证
  Section5: reset_kv_cache 的必要性
  Section6: 面试题集中考察
"""

import time
import unittest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from gpt_with_kv_cache import (
    MultiHeadAttention,
    TransformerBlock,
    GPTModel,
    generate_text_simple,
    generate_text_simple_cached,
)

# ─────────────────────────────────────────────────────────────────────────────
# 共享配置（小模型，测试快）
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "vocab_size": 100,
    "context_length": 32,
    "emb_dim": 16,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


def make_model():
    torch.manual_seed(0)
    m = GPTModel(CFG)
    m.eval()
    return m


# ═════════════════════════════════════════════════════════════════════════════
# 第 1 节  为什么需要 KV Cache？
# ═════════════════════════════════════════════════════════════════════════════
class Section1_WhyKVCache(unittest.TestCase):
    """
    【概念】不用 KV Cache 时，生成第 t 个 token 要对长度为 t 的序列做 attention：
      - 每步复杂度 O(t × d)
      - 生成 N 个 token 总复杂度 O(N² × d)  ← 二次方！

    用 KV Cache 时：
      - K、V 只算一次，存起来复用
      - 每步只需计算新 token 的 Q，与缓存的 K/V 做 attention
      - 每步复杂度 O(t × d)，但 t 的计算已省略
      - 总复杂度降到 O(N × d)  ← 线性！

    【面试考点】
    - KV Cache 只能用于推理（inference），不能用于训练
      训练时每条样本的 K/V 都不同，无法复用
    - KV Cache 带来的代价：内存随序列长度线性增长
    - 典型加速：5x（小模型 CPU），更长序列加速比更大
    """

    def test_demo_01_without_cache_recomputes_all_tokens(self):
        """
        无 cache：每步都把完整序列喂给模型（长度不断增长）
        生成 5 个 token，第 5 步输入长度 = 4（prompt） + 4（已生成） = 8
        """
        model = make_model()
        idx = torch.randint(0, 100, (1, 4))
        input_lengths = []

        original_forward = model.forward

        def tracked_forward(in_idx, use_cache=False):
            input_lengths.append(in_idx.shape[1])
            return original_forward(in_idx, use_cache=use_cache)

        model.forward = tracked_forward

        # 手动跑无 cache 的生成循环
        for _ in range(4):
            logits = model.forward(idx)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)

        # 每步输入长度：4, 5, 6, 7（递增）
        self.assertEqual(input_lengths, [4, 5, 6, 7])

    def test_demo_02_with_cache_only_feeds_new_token(self):
        """
        有 cache：prompt 只喂一次，之后每步只喂 1 个新 token
        """
        model = make_model()
        idx = torch.randint(0, 100, (1, 4))
        input_lengths = []

        original_forward = model.forward

        def tracked_forward(in_idx, use_cache=False):
            input_lengths.append(in_idx.shape[1])
            return original_forward(in_idx, use_cache=use_cache)

        model.forward = tracked_forward

        # 模拟 cached 生成
        model.reset_kv_cache()
        logits = model.forward(idx, use_cache=True)  # 第一步：整个 prompt

        for _ in range(4):
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
            logits = model.forward(next_token, use_cache=True)  # 每步只喂 1 个

        # 输入长度：4, 1, 1, 1, 1
        self.assertEqual(input_lengths[0], 4)   # prompt 完整喂入
        self.assertEqual(input_lengths[1], 1)   # 之后每步只喂 1 个
        self.assertEqual(input_lengths[2], 1)
        self.assertEqual(input_lengths[3], 1)

    def test_demo_03_cache_cannot_be_used_in_training(self):
        """
        KV Cache 只能用于推理：训练时 use_cache=False（默认值）
        验证训练模式下模型可以正常计算梯度
        """
        model = make_model()
        model.train()

        idx = torch.randint(0, 100, (2, 6))
        logits = model(idx, use_cache=False)
        loss = logits.sum()
        loss.backward()  # 不应该报错

        # 至少有一个参数有梯度
        has_grad = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_grad)

    def test_ex_01_why_cache_only_for_inference(self):
        """
        练习 1.1: 【面试题】为什么 KV Cache 不能用于训练？

        训练时每个样本的输入不同，K/V 无法跨样本复用。
        而且训练时需要对所有 token 计算梯度，cache 会破坏计算图。

        验证：训练模式 + use_cache=True 时，模型仍能运行（但不应该在真实训练中这样用）。
        """
        model = make_model()
        model.train()
        model.reset_kv_cache()

        idx = torch.randint(0, 100, (1, 4))

        # TODO
        logits = None
        raise NotImplementedError("TODO 1.1: 用 use_cache=True 跑前向传播")

        self.assertEqual(logits.shape, torch.Size([1, 4, 100]))


# ═════════════════════════════════════════════════════════════════════════════
# 第 2 节  cache_k / cache_v 的形状变化
# ═════════════════════════════════════════════════════════════════════════════
class Section2_CacheShape(unittest.TestCase):
    """
    【概念】每次 forward 后，cache_k 和 cache_v 会把新 token 的 K/V 追加进去：

      第 1 步（prompt=4 tokens）：cache_k shape = [b, 4, n_heads, head_dim]
      第 2 步（new token=1）：    cache_k shape = [b, 5, n_heads, head_dim]
      第 3 步：                   cache_k shape = [b, 6, n_heads, head_dim]

    【面试考点】
    - cache 存在 MultiHeadAttention 里（每层独立）
    - cache 形状沿 dim=1（token 维度）增长
    - reset 后 cache_k = None，重新开始
    """

    def setUp(self):
        self.model = make_model()
        self.att = self.model.trf_blocks[0].att  # 第一个 block 的 attention

    def test_demo_01_cache_starts_as_none(self):
        """初始状态 cache_k, cache_v 都是 None"""
        self.assertIsNone(self.att.cache_k)
        self.assertIsNone(self.att.cache_v)

    def test_demo_02_cache_fills_after_first_forward(self):
        """第一次 forward 后，cache_k shape = [b, prompt_len, n_heads, head_dim]"""
        idx = torch.randint(0, 100, (1, 4))
        self.model.reset_kv_cache()
        self.model(idx, use_cache=True)

        self.assertIsNotNone(self.att.cache_k)
        # shape: [batch, prompt_len, n_heads, head_dim]
        self.assertEqual(self.att.cache_k.shape[1], 4)

    def test_demo_03_cache_grows_with_each_token(self):
        """每生成一个新 token，cache 在 dim=1 增加 1"""
        idx = torch.randint(0, 100, (1, 4))
        self.model.reset_kv_cache()
        logits = self.model(idx, use_cache=True)

        cache_len_before = self.att.cache_k.shape[1]  # 4

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        self.model(next_token, use_cache=True)

        cache_len_after = self.att.cache_k.shape[1]   # 5
        self.assertEqual(cache_len_after, cache_len_before + 1)

    def test_demo_04_cache_reset_clears_to_none(self):
        """reset 后 cache 归零，ptr 归零"""
        idx = torch.randint(0, 100, (1, 4))
        self.model.reset_kv_cache()
        self.model(idx, use_cache=True)

        self.assertIsNotNone(self.att.cache_k)

        self.model.reset_kv_cache()
        self.assertIsNone(self.att.cache_k)
        self.assertEqual(self.att.ptr_current_pos, 0)

    def test_ex_01_cache_shape_after_n_steps(self):
        """
        练习 2.1: prompt=3 tokens，再生成 5 个 token 后，
                  cache_k 在 dim=1 的大小应该是多少？
        """
        idx = torch.randint(0, 100, (1, 3))
        self.model.reset_kv_cache()
        logits = self.model(idx, use_cache=True)

        for _ in range(5):
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            logits = self.model(next_token, use_cache=True)

        # TODO
        expected_cache_len = None
        raise NotImplementedError("TODO 2.1: prompt + 生成的 token 数 = ?")

        self.assertEqual(self.att.cache_k.shape[1], expected_cache_len)

    def test_ex_02_all_layers_have_independent_cache(self):
        """
        练习 2.2: 每个 TransformerBlock 的 attention 有独立的 cache。
        验证 2 层模型中，两层的 cache_k 都不为 None。
        """
        idx = torch.randint(0, 100, (1, 4))
        self.model.reset_kv_cache()
        self.model(idx, use_cache=True)

        # TODO
        layer0_cache = None  # ← self.model.trf_blocks[0].att.cache_k
        layer1_cache = None  # ← self.model.trf_blocks[1].att.cache_k
        raise NotImplementedError("TODO 2.2")

        self.assertIsNotNone(layer0_cache)
        self.assertIsNotNone(layer1_cache)


# ═════════════════════════════════════════════════════════════════════════════
# 第 3 节  位置编码在 KV Cache 下的变化
# ═════════════════════════════════════════════════════════════════════════════
class Section3_PositionalEmbedding(unittest.TestCase):
    """
    【概念】无 cache 时，pos_ids 永远从 0 开始：
      pos_ids = torch.arange(0, seq_len)

    有 cache 时，pos_ids 从当前已处理的位置继续：
      pos_ids = torch.arange(current_pos, current_pos + seq_len)

    这样才能保证新 token 的位置编码和它在完整序列中的位置一致。

    【面试考点】
    - current_pos 记录已处理的 token 数量
    - 如果新 token 总是用 pos=0 的编码，模型会误以为它是序列开头
    - reset 时 current_pos 归 0
    """

    def test_demo_01_without_cache_pos_starts_from_zero(self):
        """无 cache 时，每次 forward pos 都从 0 开始"""
        model = make_model()
        model.reset_kv_cache()

        # 第一次 forward
        model(torch.randint(0, 100, (1, 4)), use_cache=False)
        pos_after_first = model.current_pos

        # 第二次 forward
        model(torch.randint(0, 100, (1, 4)), use_cache=False)
        pos_after_second = model.current_pos

        # use_cache=False 时 current_pos 不更新
        self.assertEqual(pos_after_first, 0)
        self.assertEqual(pos_after_second, 0)

    def test_demo_02_with_cache_pos_accumulates(self):
        """有 cache 时，current_pos 累积增长"""
        model = make_model()
        model.reset_kv_cache()

        model(torch.randint(0, 100, (1, 4)), use_cache=True)
        self.assertEqual(model.current_pos, 4)

        model(torch.randint(0, 100, (1, 1)), use_cache=True)
        self.assertEqual(model.current_pos, 5)

        model(torch.randint(0, 100, (1, 1)), use_cache=True)
        self.assertEqual(model.current_pos, 6)

    def test_demo_03_reset_clears_current_pos(self):
        """reset 后 current_pos 归 0"""
        model = make_model()
        model.reset_kv_cache()
        model(torch.randint(0, 100, (1, 4)), use_cache=True)
        self.assertEqual(model.current_pos, 4)

        model.reset_kv_cache()
        self.assertEqual(model.current_pos, 0)

    def test_ex_01_pos_tracking(self):
        """
        练习 3.1: prompt=6，再生成 3 个 token 后，current_pos 应该是多少？
        """
        model = make_model()
        model.reset_kv_cache()

        idx = torch.randint(0, 100, (1, 6))
        logits = model(idx, use_cache=True)

        for _ in range(3):
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            logits = model(next_token, use_cache=True)

        # TODO
        expected_pos = None
        raise NotImplementedError("TODO 3.1")
        self.assertEqual(model.current_pos, expected_pos)


# ═════════════════════════════════════════════════════════════════════════════
# 第 4 节  有 cache vs 无 cache 输出一致性
# ═════════════════════════════════════════════════════════════════════════════
class Section4_CacheConsistency(unittest.TestCase):
    """
    【概念】KV Cache 是纯粹的优化，不改变模型输出。
    同样的 prompt，有 cache 和无 cache 生成的 token 序列必须完全一致。

    这是验证 KV Cache 实现正确性的核心测试。

    【面试考点】
    - KV Cache 正确实现的标志：输出与无 cache 完全一致
    - 最容易出错的地方：位置编码偏移（current_pos）、mask 的切片范围
    """

    def test_demo_01_cached_and_uncached_same_output(self):
        """有 cache 和无 cache 生成的 token 序列完全一致"""
        torch.manual_seed(42)
        model = make_model()
        idx = torch.randint(0, 100, (1, 4))

        out_no_cache = generate_text_simple(
            model, idx.clone(), max_new_tokens=6,
            context_size=CFG["context_length"]
        )

        out_cached = generate_text_simple_cached(
            model, idx.clone(), max_new_tokens=6,
            context_size=CFG["context_length"], use_cache=True
        )

        self.assertTrue(torch.equal(out_no_cache, out_cached))

    def test_demo_02_cached_output_extends_input(self):
        """生成 N 个 token 后，序列长度增加 N"""
        model = make_model()
        idx = torch.randint(0, 100, (1, 4))

        out = generate_text_simple_cached(
            model, idx, max_new_tokens=5,
            context_size=CFG["context_length"], use_cache=True
        )
        self.assertEqual(out.shape[1], 9)

    def test_demo_03_input_prefix_preserved(self):
        """生成序列的前缀与输入完全一致"""
        model = make_model()
        idx = torch.tensor([[10, 20, 30, 40]])

        out = generate_text_simple_cached(
            model, idx, max_new_tokens=3,
            context_size=CFG["context_length"], use_cache=True
        )
        self.assertTrue(torch.equal(out[:, :4], idx))

    def test_ex_01_verify_consistency(self):
        """
        练习 4.1: 用不同的 prompt 验证 cached 和 uncached 输出一致。
        prompt = [1, 2, 3, 4, 5]，生成 8 个 token。
        """
        model = make_model()
        idx = torch.tensor([[1, 2, 3, 4, 5]])

        # TODO
        out_no_cache = None
        out_cached = None
        raise NotImplementedError("TODO 4.1")

        self.assertTrue(torch.equal(out_no_cache, out_cached))


# ═════════════════════════════════════════════════════════════════════════════
# 第 5 节  reset_kv_cache 的必要性
# ═════════════════════════════════════════════════════════════════════════════
class Section5_CacheReset(unittest.TestCase):
    """
    【概念】两次独立的生成之间必须 reset cache，否则：
      - 第二次生成会把第一次的 K/V 拼在后面
      - 位置编码也不会重置
      - 输出完全错误

    【面试考点】
    - 不 reset 的后果：cache 污染，输出错乱
    - reset_kv_cache 同时清除所有层的 cache 和 current_pos
    """

    def test_demo_01_without_reset_cache_pollutes(self):
        """不 reset 时，直接调 forward 会把旧 cache 拼在后面"""
        model = make_model()
        att = model.trf_blocks[0].att

        # 第一次：手动用 4 个 token 填满 cache
        model.reset_kv_cache()
        model(torch.randint(0, 100, (1, 4)), use_cache=True)
        cache_len_after_first = att.cache_k.shape[1]  # 4
        self.assertEqual(cache_len_after_first, 4)

        # 第二次：不 reset，直接再喂 4 个 token
        # cache 会累积，变成 8，而不是重置为 4
        model(torch.randint(0, 100, (1, 4)), use_cache=True)
        cache_len_after_second = att.cache_k.shape[1]  # 8

        self.assertGreater(cache_len_after_second, cache_len_after_first)

    def test_demo_02_with_reset_cache_is_clean(self):
        """reset 后第二次生成的 cache 长度与第一次相同"""
        model = make_model()
        att = model.trf_blocks[0].att
        idx = torch.randint(0, 100, (1, 4))

        model.reset_kv_cache()
        generate_text_simple_cached(model, idx.clone(), max_new_tokens=3,
                                    context_size=CFG["context_length"])
        cache_len_after_first = att.cache_k.shape[1]

        # 这次 reset 再生成
        model.reset_kv_cache()
        generate_text_simple_cached(model, idx.clone(), max_new_tokens=3,
                                    context_size=CFG["context_length"])
        cache_len_after_second = att.cache_k.shape[1]

        self.assertEqual(cache_len_after_first, cache_len_after_second)

    def test_demo_03_reset_clears_all_layers(self):
        """reset_kv_cache 清除所有 TransformerBlock 的 cache"""
        model = make_model()
        idx = torch.randint(0, 100, (1, 4))

        model.reset_kv_cache()
        model(idx, use_cache=True)

        # 所有层都有 cache
        for blk in model.trf_blocks:
            self.assertIsNotNone(blk.att.cache_k)

        model.reset_kv_cache()

        # reset 后所有层 cache 清空
        for blk in model.trf_blocks:
            self.assertIsNone(blk.att.cache_k)

    def test_ex_01_two_independent_generations(self):
        """
        练习 5.1: 用两次独立生成验证 reset 的正确性。
        两次相同 prompt + reset，输出应该完全一致。
        """
        model = make_model()
        idx = torch.tensor([[5, 10, 15]])

        # TODO
        out1 = None  # 第一次：reset → generate
        out2 = None  # 第二次：reset → generate（相同 prompt）
        raise NotImplementedError("TODO 5.1")

        self.assertTrue(torch.equal(out1, out2))


# ═════════════════════════════════════════════════════════════════════════════
# 第 6 节  面试题集中考察
# ═════════════════════════════════════════════════════════════════════════════
class Section6_InterviewQuestions(unittest.TestCase):
    """
    KV Cache 相关面试高频题
    """

    def test_iq_01_complexity_comparison(self):
        """
        【面试题】无 cache vs 有 cache 的时间复杂度

        无 cache：生成 N 个 token，第 t 步处理 t 个 token
          总计算量 ∝ 1 + 2 + ... + N = N(N+1)/2 = O(N²)

        有 cache：每步只处理 1 个新 token
          总计算量 ∝ N = O(N)

        验证：N=100 时，无 cache 的"计算量比" = 5050/100 = 50.5
        """
        N = 100
        work_no_cache = sum(range(1, N + 1))   # 1+2+...+N
        work_cached = N                         # 每步 1

        ratio = work_no_cache / work_cached
        self.assertAlmostEqual(ratio, 50.5, places=1)

    def test_iq_02_memory_tradeoff(self):
        """
        【面试题】KV Cache 的内存开销

        每层每个 token 要存 K 和 V，内存用量：
          2 × n_layers × seq_len × n_heads × head_dim × bytes_per_float

        验证小模型（CFG）生成 16 个 token 时 cache 的 element 数量
        """
        n_layers = CFG["n_layers"]        # 2
        seq_len = 16
        n_heads = CFG["n_heads"]          # 2
        head_dim = CFG["emb_dim"] // CFG["n_heads"]  # 8

        # K 和 V 各一份，每层独立
        total_elements = 2 * n_layers * seq_len * n_heads * head_dim
        # 2 * 2 * 16 * 2 * 8 = 1024
        self.assertEqual(total_elements, 1024)

    def test_iq_03_mask_is_different_with_cache(self):
        """
        【面试题】有 cache 时 attention mask 的切片方式不同

        无 cache：mask[:num_tokens_Q, :num_tokens_K]  Q=K=当前序列长度
        有 cache：mask[ptr:ptr+1, :num_tokens_K]      Q=1（新token），K=全部缓存

        因为新 token 只需要一行 mask（它看不到未来，但能看到所有历史 K）
        """
        context_length = 8
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()

        # 无 cache：第 5 步，Q 和 K 都是 5
        mask_no_cache = mask[:5, :5]
        self.assertEqual(mask_no_cache.shape, torch.Size([5, 5]))

        # 有 cache：第 5 步，Q=1（新token），K=5（缓存的全部）
        ptr = 4  # 已处理 4 个 token，现在处理第 5 个（index=4）
        mask_cached = mask[ptr:ptr+1, :5]
        self.assertEqual(mask_cached.shape, torch.Size([1, 5]))
        # 第 5 个 token 不应该屏蔽自己之前的任何 token
        self.assertFalse(mask_cached.any())

    def test_iq_04_cache_k_v_but_not_q(self):
        """
        【面试题】为什么只 cache K 和 V，不 cache Q？

        Q（query）：新 token 主动"提问"，每步都需要新算
        K（key）：旧 token 被"询问"的特征，不变，可以缓存
        V（value）：旧 token 的内容，不变，可以缓存

        验证：cache 中只有 cache_k 和 cache_v，没有 cache_q
        """
        model = make_model()
        att = model.trf_blocks[0].att

        self.assertTrue(hasattr(att, 'cache_k'))
        self.assertTrue(hasattr(att, 'cache_v'))
        self.assertFalse(hasattr(att, 'cache_q'))

    def test_iq_05_cached_output_equals_uncached(self):
        """
        【面试题】KV Cache 是纯优化，不改变结果（最重要的正确性验证）
        """
        torch.manual_seed(99)
        model = make_model()
        idx = torch.randint(0, 100, (1, 5))

        out_no_cache = generate_text_simple(
            model, idx.clone(), max_new_tokens=4,
            context_size=CFG["context_length"]
        )
        out_cached = generate_text_simple_cached(
            model, idx.clone(), max_new_tokens=4,
            context_size=CFG["context_length"], use_cache=True
        )
        self.assertTrue(torch.equal(out_no_cache, out_cached))


# ═════════════════════════════════════════════════════════════════════════════
# 运行入口
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("Ch04 Bonus: KV Cache 互动学习测试")
    print("  Section1: 为什么需要 KV Cache？（O(N²) → O(N)）")
    print("  Section2: cache_k / cache_v 形状变化")
    print("  Section3: 位置编码在 cache 下的变化")
    print("  Section4: cached vs uncached 输出一致性")
    print("  Section5: reset_kv_cache 的必要性")
    print("  Section6: 面试题集中考察")
    print("=" * 65)
    unittest.main(verbosity=2)
