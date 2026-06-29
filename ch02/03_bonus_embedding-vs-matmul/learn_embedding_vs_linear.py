"""
Ch02 Bonus: Embedding vs Linear Layer — 互动学习文件
=====================================================
运行方式:
    python -m pytest learn_embedding_vs_linear.py -v
    python -m pytest learn_embedding_vs_linear.py -v -s   # 显示 print 输出

核心知识点:
    Embedding 层 = one-hot 编码 @ Linear 层（数学等价，但 Embedding 更高效）

每节包含:
  - 演示测试 (test_demo_*): 直接运行，展示正确行为
  - 练习测试 (test_ex_*)  : 需要你填写 TODO 部分

遇到 NotImplementedError 时，找到对应 TODO 补全代码即可。
"""

import unittest
import torch
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
# 第 1 节  Embedding 层是查找表
# ═════════════════════════════════════════════════════════════════════════════
class Section1_EmbeddingIsLookup(unittest.TestCase):
    """
    【概念】nn.Embedding(vocab_size, embed_dim) 本质是一个权重矩阵：
      - 形状: [vocab_size, embed_dim]
      - 输入 token_id → 直接返回该行，无任何计算
      - embedding(torch.tensor([2])) == embedding.weight[2]
    """

    def setUp(self):
        torch.manual_seed(123)
        self.idx = torch.tensor([2, 3, 1])
        self.num_idx = 4  # max(idx) + 1
        self.out_dim = 5
        self.embedding = torch.nn.Embedding(self.num_idx, self.out_dim)

    def test_demo_01_weight_shape(self):
        """Embedding 权重矩阵形状是 [vocab_size, embed_dim]"""
        self.assertEqual(self.embedding.weight.shape, torch.Size([4, 5]))

    def test_demo_02_single_lookup(self):
        """查询单个 ID，结果等于权重矩阵对应行"""
        result = self.embedding(torch.tensor([1]))[0]
        expected = self.embedding.weight[1]
        self.assertTrue(torch.allclose(result, expected))

    def test_demo_03_batch_lookup_shape(self):
        """批量查询 3 个 ID，输出形状是 [3, embed_dim]"""
        result = self.embedding(self.idx)
        self.assertEqual(result.shape, torch.Size([3, 5]))

    def test_demo_04_batch_lookup_values(self):
        """批量查询结果等于逐行拼接"""
        result = self.embedding(self.idx)
        expected = torch.stack(
            [
                self.embedding.weight[2],
                self.embedding.weight[3],
                self.embedding.weight[1],
            ]
        )
        self.assertTrue(torch.allclose(result, expected))

    def test_ex_01_output_shape_2d_input(self):
        """
        练习 1.1: 输入是 2D 张量 [2, 3]（batch），输出形状是什么？

        提示: 输入 shape + [embed_dim] = 输出 shape
        """
        idx_2d = torch.tensor([[2, 3, 1], [0, 1, 2]])
        result = self.embedding(idx_2d)

        # TODO: 填入正确的期望形状
        expected_shape = torch.Size([2, 3, 5])  # ← 改这里
        # raise NotImplementedError("TODO 1.1: 填入正确的 expected_shape")
        self.assertEqual(result.shape, expected_shape)

    def test_ex_02_lookup_equals_weight_row(self):
        """
        练习 1.2: 验证 embedding(tensor([3])) 的结果（第0行）
                  等于 embedding.weight[3]
        """
        # TODO: 填写两种查询方式
        via_forward = None  # ← embedding(torch.tensor([3]))[0]
        via_weight = None  # ← embedding.weight[3]
        raise NotImplementedError("TODO 1.2: 填写 via_forward 和 via_weight")

        self.assertTrue(torch.allclose(via_forward, via_weight))


# ═════════════════════════════════════════════════════════════════════════════
# 第 2 节  One-Hot 编码
# ═════════════════════════════════════════════════════════════════════════════
class Section2_OneHotEncoding(unittest.TestCase):
    """
    【概念】One-Hot 编码把整数 ID 转成稀疏向量：
      - ID=2, vocab_size=4 → [0, 0, 1, 0]
      - 只有第 ID 位是 1，其余全是 0
      - F.one_hot(idx) 自动推断 num_classes = max(idx)+1
      - F.one_hot(idx, num_classes=N) 可手动指定维度
    """

    def setUp(self):
        self.idx = torch.tensor([2, 3, 1])

    def test_demo_01_single_onehot(self):
        """ID=2 的 one-hot 向量，第2位为1"""
        onehot = F.one_hot(torch.tensor([2]), num_classes=4)
        self.assertEqual(onehot.tolist(), [[0, 0, 1, 0]])

    def test_demo_02_batch_onehot_shape(self):
        """3 个 ID 的 one-hot 矩阵形状是 [3, num_classes]"""
        onehot = F.one_hot(self.idx)
        self.assertEqual(onehot.shape, torch.Size([3, 4]))

    def test_demo_03_onehot_sum_is_one(self):
        """每行 one-hot 向量的和恒为 1"""
        onehot = F.one_hot(self.idx).float()
        row_sums = onehot.sum(dim=1)
        self.assertTrue(torch.all(row_sums == 1.0))

    def test_ex_01_onehot_values(self):
        """
        练习 2.1: idx=[2,3,1]，one-hot 矩阵每行的值是什么？

        提示: F.one_hot(self.idx)
        """
        onehot = F.one_hot(self.idx)

        # TODO: 填写期望的矩阵值
        expected = [
            [0, 0, 1, 0],  # ID=2
            [0, 0, 0, 1],  # ID=3
            [0, 1, 0, 0],  # ID=1
        ]
        # raise NotImplementedError("TODO 2.1: 确认 expected 是否正确，然后删掉这行")

        self.assertEqual(onehot.tolist(), expected)

    def test_ex_02_onehot_hot_position(self):
        """
        练习 2.2: 验证 one-hot 向量中，值为 1 的位置就是原始 ID
        """
        onehot = F.one_hot(self.idx)

        # TODO: 找到每行中值为 1 的位置索引，应等于原始 self.idx
        hot_positions = onehot.argmax(dim=1)  # ← onehot.argmax(dim=1)
        # raise NotImplementedError("TODO 2.2: 填写 hot_positions")

        self.assertTrue(torch.equal(hot_positions, self.idx))


# ═════════════════════════════════════════════════════════════════════════════
# 第 3 节  Embedding == One-Hot @ Linear（核心等价关系）
# ═════════════════════════════════════════════════════════════════════════════
class Section3_EmbeddingEqualsLinear(unittest.TestCase):
    """
    【概念】Embedding(idx) 等价于 Linear(one_hot(idx))：

      embedding(idx)
        = embedding.weight[idx]          # 查表

      linear(one_hot(idx).float())
        = one_hot(idx).float() @ linear.weight.T   # 矩阵乘法

      当 linear.weight = embedding.weight.T 时，两者结果完全相同。

    【为什么用 Embedding 而不用 Linear？】
      one-hot 向量极度稀疏（大量 0），做矩阵乘法浪费算力。
      Embedding 直接按索引取行，跳过所有乘以 0 的操作，效率更高。
    """

    def setUp(self):
        torch.manual_seed(123)
        self.idx = torch.tensor([2, 3, 1])
        num_idx = 4
        out_dim = 5

        self.embedding = torch.nn.Embedding(num_idx, out_dim)

        # Linear 层权重设为 embedding.weight.T，使两者等价
        self.linear = torch.nn.Linear(num_idx, out_dim, bias=False)
        self.linear.weight = torch.nn.Parameter(self.embedding.weight.T)

        self.onehot = F.one_hot(self.idx).float()

    def test_demo_01_linear_weight_shape(self):
        """Linear 权重形状是 [out_dim, num_idx]，恰好是 Embedding 权重的转置"""
        self.assertEqual(self.linear.weight.shape, torch.Size([5, 4]))
        self.assertEqual(self.embedding.weight.shape, torch.Size([4, 5]))

    def test_demo_02_embedding_equals_linear(self):
        """Embedding 和 Linear(one-hot) 结果完全相同"""
        emb_result = self.embedding(self.idx)
        linear_result = self.linear(self.onehot)
        self.assertTrue(torch.allclose(emb_result, linear_result))

    def test_demo_03_manual_matmul_equals_embedding(self):
        """手动矩阵乘法也等价：one_hot @ embedding.weight"""
        emb_result = self.embedding(self.idx)
        matmul_result = self.onehot @ self.embedding.weight
        self.assertTrue(torch.allclose(emb_result, matmul_result))

    def test_ex_01_why_onehot_matmul_is_row_select(self):
        """
        练习 3.1: 解释为什么 one_hot @ W 等价于取 W 的某一行。

        one_hot([2]) = [0, 0, 1, 0]
        [0,0,1,0] @ W = 0*W[0] + 0*W[1] + 1*W[2] + 0*W[3] = W[2]

        验证：one_hot(tensor([2])).float() @ embedding.weight == embedding.weight[2]
        """
        onehot_2 = F.one_hot(torch.tensor([2]), num_classes=4).float()

        # TODO: 计算 onehot_2 @ self.embedding.weight，赋值给 result
        result = None  # ← onehot_2 @ self.embedding.weight
        raise NotImplementedError("TODO 3.1: 填写 result")

        expected = self.embedding.weight[2]
        self.assertTrue(torch.allclose(result[0], expected))

    def test_ex_02_set_linear_weight_and_compare(self):
        """
        练习 3.2: 从头创建 Linear 层，手动把权重设为 embedding.weight.T，
                  验证结果与 Embedding 相同。
        """
        torch.manual_seed(123)
        embedding = torch.nn.Embedding(4, 5)

        # TODO: 创建 linear 层并设置权重
        linear = None  # ← torch.nn.Linear(4, 5, bias=False)
        # linear.weight = torch.nn.Parameter(embedding.weight.T)
        raise NotImplementedError("TODO 3.2: 创建 linear 并设置权重")

        idx = torch.tensor([2, 3, 1])
        onehot = F.one_hot(idx).float()

        self.assertTrue(torch.allclose(embedding(idx), linear(onehot)))

    def test_ex_03_efficiency_intuition(self):
        """
        练习 3.3: 体会稀疏性。
        vocab_size=50257 时，one-hot 向量有多少个 0？
        验证：非零元素只有 1 个。
        """
        vocab_size = 50257
        token_id = torch.tensor([42])

        # TODO: 生成 one-hot 向量，赋值给 onehot
        onehot = None  # ← F.one_hot(token_id, num_classes=vocab_size).float()
        raise NotImplementedError("TODO 3.3: 填写 onehot")

        num_nonzero = (onehot != 0).sum().item()
        num_zero = (onehot == 0).sum().item()

        self.assertEqual(num_nonzero, 1)
        self.assertEqual(num_zero, vocab_size - 1)
        print(f"\nvocab_size={vocab_size} 时，one-hot 向量有 {num_zero} 个 0，只有 {num_nonzero} 个 1")
        print("→ 矩阵乘法做了 50257 次乘法，其中 50256 次都在乘以 0，纯属浪费")
        print("→ Embedding 直接取行，跳过所有无效计算")


# ═════════════════════════════════════════════════════════════════════════════
# 第 4 节  综合练习
# ═════════════════════════════════════════════════════════════════════════════
class Section4_综合练习(unittest.TestCase):
    """综合验证 Embedding 和 Linear 的等价性"""

    def test_ex_01_full_equivalence_check(self):
        """
        综合练习: 给定 idx=[0,1,2,3]，用以下两种方式计算 embedding，验证结果相同：
          方式 A: nn.Embedding(4, 8)(idx)
          方式 B: nn.Linear(4, 8, bias=False)(one_hot(idx).float())
                  （Linear 权重设为 embedding.weight.T）

        步骤:
          1. torch.manual_seed(42) 初始化
          2. 创建 Embedding(4, 8)
          3. 创建 Linear(4, 8, bias=False)，权重设为 embedding.weight.T
          4. 计算两种结果并断言 allclose
        """
        # TODO: 完整实现
        raise NotImplementedError("TODO 综合练习: 实现完整等价性验证")

    def test_ex_02_gradient_flows_through_embedding(self):
        """
        综合练习 2: Embedding 权重支持梯度（requires_grad=True），
        可以被反向传播更新。验证 embedding.weight.requires_grad == True。
        """
        embedding = torch.nn.Embedding(10, 4)

        # TODO: 验证 embedding.weight.requires_grad
        result = None  # ← embedding.weight.requires_grad
        raise NotImplementedError("TODO: 填写 result")

        self.assertTrue(result)


# ═════════════════════════════════════════════════════════════════════════════
# 运行入口
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Ch02 Bonus: Embedding vs Linear 互动学习测试")
    print("  演示测试 (test_demo_*): 展示正确行为")
    print("  练习测试 (test_ex_*)  : 需要你填写 TODO 完成")
    print("  遇到 NotImplementedError = 找到 TODO 补全代码！")
    print("=" * 60)
    unittest.main(verbosity=2)
