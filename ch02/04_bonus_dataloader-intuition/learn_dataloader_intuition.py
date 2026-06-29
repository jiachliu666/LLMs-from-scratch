"""
Ch02 Bonus: DataLoader 直觉理解 — 互动学习文件
===============================================
运行方式:
    python -m pytest learn_dataloader_intuition.py -v
    python -m pytest learn_dataloader_intuition.py -v -s   # 显示 print 输出

核心知识点:
    用数字序列 [0, 1, 2, ..., 1000] 代替文字 token，
    让滑动窗口的行为一眼可见。

    - input:  token_ids[i : i+max_length]
    - target: token_ids[i+1 : i+max_length+1]  （向右偏移一位）
    - stride 控制窗口每次移动多少步
"""

import unittest
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# 工具：用数字文本构造 DataLoader（不需要 tokenizer）
# ─────────────────────────────────────────────────────────────────────────────
class NumberDataset(Dataset):
    """把 '0 1 2 3 ... N' 这样的数字文本当作 token 序列"""

    def __init__(self, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = [int(i) for i in txt.strip().split()]
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def make_number_text(n=1000):
    """生成 '0 1 2 ... n' 的字符串"""
    return " ".join(str(i) for i in range(n + 1))


def make_dataloader(n=1000, batch_size=1, max_length=4, stride=1, shuffle=False):
    txt = make_number_text(n)
    dataset = NumberDataset(txt, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# ═════════════════════════════════════════════════════════════════════════════
# 第 1 节  单样本：input 和 target 的关系
# ═════════════════════════════════════════════════════════════════════════════
class Section1_InputTargetRelationship(unittest.TestCase):
    """
    【概念】用数字序列理解 input/target 偏移关系：

      token_ids = [0, 1, 2, 3, 4, 5, ...]

      第一个样本（i=0, max_length=4）:
        input:  [0, 1, 2, 3]
        target: [1, 2, 3, 4]   ← 整体向右偏移一位

      含义：给模型看 [0,1,2,3]，让它预测 [1,2,3,4]
           即：看到 0 预测 1，看到 0,1 预测 2，...
    """

    def setUp(self):
        self.dl = make_dataloader(batch_size=1, max_length=4, stride=1)
        self.it = iter(self.dl)

    def test_demo_01_first_batch_input(self):
        """第一个 batch 的 input 是 [0,1,2,3]"""
        inputs, _ = next(self.it)
        self.assertEqual(inputs[0].tolist(), [0, 1, 2, 3])

    def test_demo_02_first_batch_target(self):
        """第一个 batch 的 target 是 [1,2,3,4]（input 右移一位）"""
        _, targets = next(self.it)
        self.assertEqual(targets[0].tolist(), [1, 2, 3, 4])

    def test_demo_03_target_is_input_shifted(self):
        """target 始终等于 input 向右偏移一位"""
        inputs, targets = next(self.it)
        # input[1:] == target[:-1]
        self.assertTrue(torch.equal(inputs[0, 1:], targets[0, :-1]))

    def test_demo_04_second_batch(self):
        """stride=1 时，第二个 batch 从位置 1 开始"""
        next(self.it)
        inputs, targets = next(self.it)
        self.assertEqual(inputs[0].tolist(), [1, 2, 3, 4])
        self.assertEqual(targets[0].tolist(), [2, 3, 4, 5])

    def test_ex_01_third_batch(self):
        """
        练习 1.1: stride=1, max_length=4 时，第三个 batch 的 input 和 target 是什么？
        """
        next(self.it)
        next(self.it)
        inputs, targets = next(self.it)

        # TODO: 填入正确的期望值
        expected_input = [2, 3, 4, 5]  # ← [2, 3, 4, 5]
        expected_target = [3, 4, 5, 6]  # ← [3, 4, 5, 6]
        # raise NotImplementedError("TODO 1.1: 填写 expected_input 和 expected_target")

        self.assertEqual(inputs[0].tolist(), expected_input)
        self.assertEqual(targets[0].tolist(), expected_target)

    def test_ex_02_last_batch(self):
        """
        练习 1.2: token_ids = [0..1000]，max_length=4，stride=1，
                  最后一个 batch 的 input 是什么？

        提示: 最后一个窗口从 index 996 开始（1001-4=997，range 不含末尾）
        """
        dl = make_dataloader(batch_size=1, max_length=4, stride=1)
        last_inputs = last_targets = None
        for last_inputs, last_targets in dl:
            pass

        # TODO: 填入正确的期望值
        expected_input = [996, 997, 998, 999]  # ← [996, 997, 998, 999]
        expected_target = [997, 998, 999, 1000]  # ← [997, 998, 999, 1000]
        # raise NotImplementedError("TODO 1.2: 填写 expected_input 和 expected_target")

        self.assertEqual(last_inputs[0].tolist(), expected_input)
        self.assertEqual(last_targets[0].tolist(), expected_target)


# ═════════════════════════════════════════════════════════════════════════════
# 第 2 节  stride 的作用
# ═════════════════════════════════════════════════════════════════════════════
class Section2_StrideEffect(unittest.TestCase):
    """
    【概念】stride 决定相邻样本的起始位置差：

      stride=1:  [0,1,2,3], [1,2,3,4], [2,3,4,5], ...   重叠 3 个
      stride=2:  [0,1,2,3], [2,3,4,5], [4,5,6,7], ...   重叠 2 个
      stride=4:  [0,1,2,3], [4,5,6,7], [8,9,10,11], ...  无重叠

      重叠数 = max_length - stride
    """

    def test_demo_01_stride1_overlap(self):
        """stride=1，相邻样本重叠 3 个 token：first[1:] == second[:-1]"""
        dl = make_dataloader(max_length=4, stride=1)
        it = iter(dl)
        first, _ = next(it)
        second, _ = next(it)
        self.assertTrue(torch.equal(first[0, 1:], second[0, :-1]))

    def test_demo_02_stride4_no_overlap(self):
        """stride=max_length=4，相邻样本无重叠"""
        dl = make_dataloader(max_length=4, stride=4)
        it = iter(dl)
        first, _ = next(it)
        second, _ = next(it)
        # first=[0,1,2,3], second=[4,5,6,7]，无公共元素
        self.assertEqual(first[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(second[0].tolist(), [4, 5, 6, 7])

    def test_demo_03_num_samples_formula(self):
        """样本数 = (len(token_ids) - max_length) // stride"""
        max_length = 4
        stride = 4
        txt = make_number_text(1000)  # 1001 个 token
        dataset = NumberDataset(txt, max_length, stride)
        expected = len(range(0, 1001 - max_length, stride))
        self.assertEqual(len(dataset), expected)

    def test_ex_01_overlap_count(self):
        """
        练习 2.1: stride=2, max_length=4 时，相邻两个样本重叠几个 token？

        验证: first[0, 2:] == second[0, :-2]
        """
        dl = make_dataloader(max_length=4, stride=2)
        it = iter(dl)
        first, _ = next(it)
        second, _ = next(it)

        # TODO: 填写重叠的切片验证
        # first 的后半段 == second 的前半段
        overlap_in_first = first[0, 2:]  # ← first[0, 2:]
        overlap_in_second = second[0, :-2]  # ← second[0, :-2]
        # raise NotImplementedError("TODO 2.1: 填写 overlap_in_first 和 overlap_in_second")

        self.assertTrue(torch.equal(overlap_in_first, overlap_in_second))

    def test_ex_02_stride2_second_batch_start(self):
        """
        练习 2.2: stride=2, max_length=4 时，
                  第一个 batch input=[0,1,2,3]，
                  第二个 batch input 从哪里开始？
        """
        dl = make_dataloader(max_length=4, stride=2)
        it = iter(dl)
        next(it)
        second, _ = next(it)

        # TODO: 填写期望值
        expected = [2, 3, 4, 5]  # ← [2, 3, 4, 5]
        # raise NotImplementedError("TODO 2.2: 填写 expected")

        self.assertEqual(second[0].tolist(), expected)

    def test_ex_03_num_samples_stride1(self):
        """
        练习 2.3: token_ids 长度=1001, max_length=4, stride=1，
                  数据集有多少个样本？

        提示: (len - max_length) // stride
        """
        txt = make_number_text(1000)
        dataset = NumberDataset(txt, max_length=4, stride=1)

        # TODO: 填写期望样本数
        expected = (1001 - 4) // 1  # ← (1001 - 4) // 1 = 997
        # raise NotImplementedError("TODO 2.3: 填写 expected")

        self.assertEqual(len(dataset), expected)


# ═════════════════════════════════════════════════════════════════════════════
# 第 3 节  batch_size 的作用
# ═════════════════════════════════════════════════════════════════════════════
class Section3_BatchSize(unittest.TestCase):
    """
    【概念】batch_size 决定每次 next() 取出多少个样本：

      batch_size=1:  inputs.shape = [1, max_length]
      batch_size=2:  inputs.shape = [2, max_length]
      batch_size=4:  inputs.shape = [4, max_length]

      batch 内部样本按 stride 连续排列，batch 之间不重叠。
    """

    def test_demo_01_batch2_shape(self):
        """batch_size=2, max_length=4 → inputs shape [2, 4]"""
        dl = make_dataloader(batch_size=2, max_length=4, stride=4)
        inputs, targets = next(iter(dl))
        self.assertEqual(inputs.shape, torch.Size([2, 4]))
        self.assertEqual(targets.shape, torch.Size([2, 4]))

    def test_demo_02_batch2_values(self):
        """batch_size=2, stride=4：第一个 batch 包含样本0和样本1"""
        dl = make_dataloader(batch_size=2, max_length=4, stride=4)
        inputs, _ = next(iter(dl))
        self.assertEqual(inputs[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(inputs[1].tolist(), [4, 5, 6, 7])

    def test_demo_03_batch_does_not_cause_overlap(self):
        """batch_size 不影响样本间是否重叠，重叠只由 stride 决定"""
        dl1 = make_dataloader(batch_size=1, max_length=4, stride=4)
        dl2 = make_dataloader(batch_size=2, max_length=4, stride=4)

        # batch_size=1 取两次，等于 batch_size=2 取一次
        it1 = iter(dl1)
        a, _ = next(it1)
        b, _ = next(it1)

        it2 = iter(dl2)
        ab, _ = next(it2)

        self.assertTrue(torch.equal(a[0], ab[0]))
        self.assertTrue(torch.equal(b[0], ab[1]))

    def test_ex_01_batch4_shape(self):
        """
        练习 3.1: batch_size=4, max_length=8, stride=8，
                  第一个 batch 的 inputs.shape 是什么？
        """
        dl = make_dataloader(batch_size=4, max_length=8, stride=8)
        inputs, _ = next(iter(dl))

        # TODO: 填写期望形状
        expected_shape = None  # ← torch.Size([4, 8])
        raise NotImplementedError("TODO 3.1: 填写 expected_shape")

        self.assertEqual(inputs.shape, expected_shape)

    def test_ex_02_batch2_second_batch_values(self):
        """
        练习 3.2: batch_size=2, max_length=4, stride=4，
                  第二个 batch 的两行 input 分别是什么？
        """
        dl = make_dataloader(batch_size=2, max_length=4, stride=4)
        it = iter(dl)
        next(it)
        inputs, _ = next(it)

        # TODO: 填写期望值
        expected_row0 = None  # ← [8, 9, 10, 11]
        expected_row1 = None  # ← [12, 13, 14, 15]
        raise NotImplementedError("TODO 3.2: 填写 expected_row0 和 expected_row1")

        self.assertEqual(inputs[0].tolist(), expected_row0)
        self.assertEqual(inputs[1].tolist(), expected_row1)


# ═════════════════════════════════════════════════════════════════════════════
# 第 4 节  shuffle 的作用
# ═════════════════════════════════════════════════════════════════════════════
class Section4_Shuffle(unittest.TestCase):
    """
    【概念】shuffle=True 时，每个 epoch DataLoader 会打乱样本顺序。

      - shuffle=False: batch 按顺序从样本0开始
      - shuffle=True:  batch 的起始样本是随机的（需固定 manual_seed 才可复现）

      训练时通常开 shuffle，防止模型记住顺序；
      验证/测试时关闭 shuffle，保证结果可复现。
    """

    def test_demo_01_no_shuffle_starts_from_zero(self):
        """shuffle=False 时，第一个 batch 从 token 0 开始"""
        dl = make_dataloader(batch_size=1, max_length=4, stride=4, shuffle=False)
        inputs, _ = next(iter(dl))
        self.assertEqual(inputs[0].tolist(), [0, 1, 2, 3])

    def test_demo_02_shuffle_changes_order(self):
        """shuffle=True 时，第一个 batch 不一定从 token 0 开始"""
        torch.manual_seed(123)
        dl = make_dataloader(batch_size=1, max_length=4, stride=4, shuffle=True)
        inputs, _ = next(iter(dl))
        # 打乱后第一个 batch 大概率不是 [0,1,2,3]
        self.assertNotEqual(inputs[0].tolist(), [0, 1, 2, 3])

    def test_demo_03_shuffle_same_seed_reproducible(self):
        """相同 seed 下，shuffle=True 的结果可复现"""
        torch.manual_seed(42)
        dl1 = make_dataloader(batch_size=2, max_length=4, stride=4, shuffle=True)
        result1, _ = next(iter(dl1))

        torch.manual_seed(42)
        dl2 = make_dataloader(batch_size=2, max_length=4, stride=4, shuffle=True)
        result2, _ = next(iter(dl2))

        self.assertTrue(torch.equal(result1, result2))

    def test_ex_01_shuffle_covers_all_samples(self):
        """
        练习 4.1: shuffle=True 时，遍历完整个 DataLoader 后，
                  所有样本都被访问过（总 token 数不变）。

        验证: shuffle=False 和 shuffle=True 遍历到的 input 集合相同
              （忽略顺序）
        """

        def collect_all_inputs(shuffle):
            dl = make_dataloader(batch_size=1, max_length=4, stride=4, shuffle=shuffle)
            all_rows = []
            for inputs, _ in dl:
                all_rows.append(tuple(inputs[0].tolist()))
            return set(all_rows)

        # TODO: 验证两个集合相同
        # set_no_shuffle = collect_all_inputs(False)
        # set_shuffle    = collect_all_inputs(True)
        raise NotImplementedError("TODO 4.1: 取消注释并断言两个集合相等")

        # self.assertEqual(set_no_shuffle, set_shuffle)


# ═════════════════════════════════════════════════════════════════════════════
# 综合练习
# ═════════════════════════════════════════════════════════════════════════════
class Section5_综合练习(unittest.TestCase):
    def test_ex_full_pipeline(self):
        """
        综合练习: 用数字序列 [0..99]，max_length=4, stride=2, batch_size=2，
                  验证：
                  1. 第一个 batch inputs shape = [2, 4]
                  2. inputs[0] = [0,1,2,3]，inputs[1] = [2,3,4,5]
                  3. 对应 targets[0] = [1,2,3,4]，targets[1] = [3,4,5,6]
                  4. 数据集总样本数 = (100 - 4) // 2 = 48
        """
        txt = make_number_text(99)  # 0..99，共 100 个 token

        # TODO: 创建 dataset 和 dataloader，完成验证
        raise NotImplementedError("TODO 综合练习: 完整实现")


# ═════════════════════════════════════════════════════════════════════════════
# 运行入口
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Ch02 Bonus: DataLoader 直觉理解 互动学习测试")
    print("  演示测试 (test_demo_*): 展示正确行为")
    print("  练习测试 (test_ex_*)  : 需要你填写 TODO 完成")
    print("=" * 60)
    unittest.main(verbosity=2)
