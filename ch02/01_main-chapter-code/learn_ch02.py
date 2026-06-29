"""
Ch02: Working with Text Data — 互动学习文件
============================================
运行方式:
    python learn_ch02.py          # 只跑已实现的测试
    python -m pytest learn_ch02.py -v  # 详细输出

每节包含:
  - 概念讲解 (docstring)
  - 演示测试 (test_demo_*): 直接运行，展示正确行为
  - 练习测试 (test_ex_*)  : 需要你填写 TODO 部分

遇到 NotImplementedError 时，找到对应 TODO 补全代码即可。
"""

import re
import unittest
import os
import torch
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# 工具：加载文本
# ─────────────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_TEXT_PATH = os.path.join(_DIR, "the-verdict.txt")


def load_verdict() -> str:
    with open(_TEXT_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ═════════════════════════════════════════════════════════════════════════════
# 第 1 节  文本分词 (Tokenization)
# ═════════════════════════════════════════════════════════════════════════════
class Section1_Tokenization(unittest.TestCase):
    """
    【概念】Tokenization = 把原始文本切分成最小单元 (token)。
    LLM 无法直接处理字符串，需要把文字变成数字 ID。
    流程: 原始文本 → tokens (字符串列表) → token IDs (整数列表)

    本节使用正则表达式实现一个简单分词器。
    关键正则: ([,.:;?_!"()\\'']|--|\\s)
      - 匹配标点、破折号、空白，并把它们当分隔符保留在结果里
    """

    def test_demo_01_basic_split(self):
        """最简单的按空格分词"""
        text = "Hello, world. This, is a test."
        result = re.split(r"(\s)", text)
        # 空格也被保留了
        self.assertIn("Hello,", result)
        self.assertIn(" ", result)

    def test_demo_02_split_with_punctuation(self):
        """按空格 + 标点分词，并过滤空字符串"""
        text = "Hello, world. Is this-- a test?"
        result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        result = [item.strip() for item in result if item.strip()]
        self.assertEqual(result, ["Hello", ",", "world", ".", "Is", "this", "--", "a", "test", "?"])

    def test_demo_03_tokenize_full_text(self):
        """对完整文本分词，统计 token 数量"""
        raw_text = load_verdict()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # 原文本约 4690 个 token
        self.assertGreater(len(preprocessed), 4000)

    # ──────────────────────── 练习 1.1 ────────────────────────
    def test_ex_01_count_unique_tokens(self):
        """
        练习 1.1: 对 the-verdict.txt 分词后，计算有多少个「唯一」token。
        期望答案约为 1130。

        提示: 使用 set() 去重，再 len()。
        """
        raw_text = load_verdict()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        # TODO: 计算唯一 token 数量，赋值给 unique_count
        unique_count = None  # ← 改这里
        raise NotImplementedError("TODO 1.1: 计算唯一 token 数量")

        self.assertAlmostEqual(unique_count, 1130, delta=50)

    # ──────────────────────── 练习 1.2 ────────────────────────
    def test_ex_02_custom_tokenize(self):
        """
        练习 1.2: 写一个函数 simple_tokenize(text) -> list[str]
        要求：用正则把文本切成 token 列表（去掉空字符串和纯空白）。

        测试用例: "Hello, world!" → ['Hello', ',', 'world', '!']
        """

        def simple_tokenize(text: str) -> list:
            # TODO: 实现分词
            raise NotImplementedError("TODO 1.2: 实现 simple_tokenize")

        result = simple_tokenize("Hello, world!")
        self.assertEqual(result, ["Hello", ",", "world", "!"])

        result2 = simple_tokenize("one two--three")
        self.assertIn("--", result2)


# ═════════════════════════════════════════════════════════════════════════════
# 第 2 节  词表与 Token ID
# ═════════════════════════════════════════════════════════════════════════════
def build_vocab(text: str) -> dict:
    """从文本构建词表 {token: id}，按字母排序分配 ID。"""
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))
    return {token: integer for integer, token in enumerate(all_words)}


class SimpleTokenizerV1:
    """
    【V1】最简单的 tokenizer：encode/decode，不处理未知词。
    encode: 文本 → ID 列表
    decode: ID 列表 → 文本
    """

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r'\s+([,.?!"()\'])', r"\1", text)


class Section2_VocabAndIDs(unittest.TestCase):
    """
    【概念】词表 (Vocabulary) = {token字符串: 整数ID} 的映射。
    - 每个唯一 token 对应一个唯一整数
    - encode: 字符串 → 整数序列（供模型处理）
    - decode: 整数序列 → 字符串（供人类阅读）
    """

    def setUp(self):
        raw_text = load_verdict()
        self.vocab = build_vocab(raw_text)
        self.tokenizer = SimpleTokenizerV1(self.vocab)

    def test_demo_01_vocab_size(self):
        """词表大小约为 1130"""
        self.assertGreater(len(self.vocab), 1000)

    def test_demo_02_encode_and_decode(self):
        """encode 后再 decode 应该还原原文（已在词表内的词）"""
        text = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)
        self.assertIn("Gisburn", decoded)
        self.assertIn("painted", decoded)

    def test_demo_03_unknown_word_raises(self):
        """V1 遇到词表外的词会抛出 KeyError"""
        with self.assertRaises(KeyError):
            self.tokenizer.encode("Hello, do you like tea?")  # "Hello" 不在词表

    # ──────────────────────── 练习 2.1 ────────────────────────
    def test_ex_01_manual_encode(self):
        """
        练习 2.1: 不使用 tokenizer，手动把 ['Mrs', '.', 'Gisburn'] 转为 ID 列表。
        提示: 直接查 self.vocab 字典。
        """
        tokens = ["Mrs", ".", "Gisburn"]

        # TODO: 把 tokens 列表转换成对应的整数 ID 列表，赋值给 ids
        ids = [self.vocab[token] for token in tokens]

        self.assertEqual(len(ids), 3)
        self.assertIsInstance(ids[0], int)

    # ──────────────────────── 练习 2.2 ────────────────────────
    def test_ex_02_manual_decode(self):
        """
        练习 2.2: 给定 ID 列表 [850, 988, 602]，把它解码回字符串列表。
        提示: 需要反向映射 {id: token}。
        """
        ids = [850, 988, 602]
        int_to_str = {i: s for s, i in self.vocab.items()}

        # TODO: 把 ids 转换回 token 字符串列表，赋值给 tokens
        tokens = [int_to_str[_id] for _id in ids]

        self.assertEqual(len(tokens), 3)
        self.assertIsInstance(tokens[0], str)


# ═════════════════════════════════════════════════════════════════════════════
# 第 3 节  特殊 Token
# ═════════════════════════════════════════════════════════════════════════════
class SimpleTokenizerV2:
    """
    【V2】加入特殊 token：
    - <|endoftext|>: 标记文本结束（或拼接两段不相关文本的分隔符）
    - <|unk|>: 代替词表外的词

    GPT-2 只用 <|endoftext|>（通过 BPE 避免了未知词问题）。
    """

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)


class Section3_SpecialTokens(unittest.TestCase):
    """
    【概念】特殊 token 的作用：
    - <|endoftext|>: 分隔两段无关文本，如 "文章1 <|endoftext|> 文章2"
    - <|unk|>      : 表示词表外的词（simple tokenizer 用）
    - [BOS]/[EOS]/[PAD]: 其他模型常用，GPT-2 不使用
    """

    def setUp(self):
        raw_text = load_verdict()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        all_tokens = sorted(set(preprocessed))
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])
        vocab = {token: i for i, token in enumerate(all_tokens)}
        self.vocab = vocab
        self.tokenizer = SimpleTokenizerV2(vocab)

    def test_demo_01_unknown_word_becomes_unk(self):
        """词表外的词被替换为 <|unk|>"""
        ids = self.tokenizer.encode("Hello world")
        decoded = self.tokenizer.decode(ids)
        self.assertIn("<|unk|>", decoded)

    def test_demo_02_endoftext_separator(self):
        """两段文本用 <|endoftext|> 拼接"""
        text = "In the sunlit terraces <|endoftext|> of the palace."
        ids = self.tokenizer.encode(text)
        eos_id = self.vocab["<|endoftext|>"]
        self.assertIn(eos_id, ids)

    def test_demo_03_special_tokens_at_end_of_vocab(self):
        """特殊 token 位于词表末尾"""
        keys = list(self.vocab.keys())
        self.assertEqual(keys[-2], "<|endoftext|>")
        self.assertEqual(keys[-1], "<|unk|>")

    # ──────────────────────── 练习 3.1 ────────────────────────
    def test_ex_01_endoftext_id(self):
        """
        练习 3.1: 找出 <|endoftext|> 的 ID，并验证它是词表中最大 ID 之一。
        提示: self.vocab["<|endoftext|>"]
        """
        # TODO: 获取 <|endoftext|> 的整数 ID，赋值给 eos_id
        eos_id = self.vocab["<|endoftext|>"]

        self.assertIsInstance(eos_id, int)
        self.assertGreater(eos_id, 1000)  # 应该是词表末尾附近

    # ──────────────────────── 练习 3.2 ────────────────────────
    def test_ex_02_concatenate_texts(self):
        """
        练习 3.2: 把下面两段文本用 <|endoftext|> 连接后，
        用 tokenizer 编码，验证结果中含有 <|endoftext|> 的 ID。

        text1 = "in the sunlit terraces"  (这些词在词表里)
        text2 = "of the palace"
        """
        text1 = "in the sunlit terraces"
        text2 = "of the palace"

        # TODO: 把 text1 和 text2 用 " <|endoftext|> " 拼接，再编码
        texts = text1 + " <|endoftext|> " + text2
        ids = self.tokenizer.encode(texts)

        eos_id = self.vocab["<|endoftext|>"]
        self.assertIn(eos_id, ids)


# ═════════════════════════════════════════════════════════════════════════════
# 第 4 节  BPE Tokenizer (tiktoken)
# ═════════════════════════════════════════════════════════════════════════════
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@unittest.skipUnless(TIKTOKEN_AVAILABLE, "需要安装 tiktoken: pip install tiktoken")
class Section4_BPE(unittest.TestCase):
    """
    【概念】Byte Pair Encoding (BPE) — GPT-2 使用的分词算法。

    核心思想: 把「频繁相邻的字节对」合并成新 token，反复迭代。
    优势:
    1. 无未知词: 任何词都可拆成子词/字节
    2. 词表大小可控 (GPT-2 词表: 50257)
    3. 在常见词和罕见词之间取得平衡

    示例: "unfamiliarword" → ["unfam", "iliar", "word"]（根据训练的合并规则）
    """

    def setUp(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        raw_text = load_verdict()
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        all_tokens = sorted(set(preprocessed))
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])
        vocab = {token: i for i, token in enumerate(all_tokens)}
        self.simple = SimpleTokenizerV2(vocab)

    def test_demo_01_vocab_size(self):
        """GPT-2 BPE 词表大小为 50257"""
        self.assertEqual(self.tokenizer.n_vocab, 50257)

    def test_demo_02_encode_decode_roundtrip(self):
        """encode 后 decode 能还原原文"""
        text = "Hello, do you like tea?"
        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(decoded, text)

    def test_demo_03_endoftext_token(self):
        """<|endoftext|> 的 ID 是 50256（需要 allowed_special 参数）"""
        ids = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
        self.assertEqual(ids, [50256])

    def test_demo_04_unknown_word_breakdown(self):
        """
        BPE 会把未知词拆成子词，而不是返回 <unk>。
        'someunknownPlace' 会被拆成多个 token。
        """
        ids = self.tokenizer.encode("someunknownPlace")
        self.assertGreater(len(ids), 1)  # 被拆开了

    # ──────────────────────── 练习 4.1 (书中 Exercise 2.1) ────────────────────
    def test_ex_01_tokenize_akwirw(self):
        """
        练习 4.1 (书中 Exercise 2.1):
        对字符串 "Akwirw ier" 用 GPT-2 BPE tokenizer 编码。
        1. 打印每个 ID 及其对应的子词
        2. 验证所有 token 解码后能还原原始字符串

        期望 token 数量: 6 个
        期望 IDs: [33901, 86, 343, 86, 220, 959]
        """
        text = "Akwirw ier"

        # TODO: 用 self.tokenizer.encode(text) 获取 IDs
        ids = self.tokenizer.encode(text)

        self.assertEqual(len(ids), 6)
        self.assertEqual(ids, [33901, 86, 343, 86, 220, 959])

        # 每个 ID 单独解码
        for i in ids:
            sub = self.tokenizer.decode([i])
            self.assertIsInstance(sub, str)

        # 全部解码回原文
        self.assertEqual(self.tokenizer.decode(ids), text)

    # ──────────────────────── 练习 4.2 ────────────────────────
    def test_ex_02_compare_with_simple_tokenizer(self):
        """
        练习 4.2: 对同一段文本，比较 BPE tokenizer 和 simple_tokenize 的结果。
        文本: "Hello, world. Is this a test?"

        思考: 哪个产生更少 token？哪个能处理未知词？
        """
        text = "Hello, world. Is this a test?"

        # TODO: 用 BPE encode，赋值给 bpe_ids
        bpe_ids = self.tokenizer.encode(text)

        # TODO: 用正则 simple_tokenize，赋值给 simple_tokens
        simple_tokens = self.simple.encode(text)

        # BPE 和 simple 的 token 数量通常不同
        self.assertIsInstance(bpe_ids, list)
        self.assertIsInstance(simple_tokens, list)
        print(f"\n  BPE tokens ({len(bpe_ids)}): {[self.tokenizer.decode([i]) for i in bpe_ids]}")
        print(f"  Simple tokens ({len(simple_tokens)}): {simple_tokens}")


# ═════════════════════════════════════════════════════════════════════════════
# 第 5 节  滑动窗口数据集 (Sliding Window Dataset)
# ═════════════════════════════════════════════════════════════════════════════
class GPTDatasetV1(Dataset):
    """
    【概念】LLM 训练数据的构造方式：
    - 输入 (x): token_ids[i : i+max_length]
    - 目标 (y): token_ids[i+1 : i+max_length+1]  ← 向右偏移一位
    - 用「滑动窗口」遍历整个文本，步长由 stride 控制

    stride < max_length → 样本间有重叠（数据量更多，但可能过拟合）
    stride = max_length → 样本互不重叠（更常用的训练方式）
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i : i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1 : i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


@unittest.skipUnless(TIKTOKEN_AVAILABLE, "需要安装 tiktoken")
class Section5_SlidingWindow(unittest.TestCase):
    """
    滑动窗口数据加载器——为「预测下一个 token」任务构造训练样本。
    """

    def setUp(self):
        self.raw_text = load_verdict()

    def test_demo_01_input_target_offset(self):
        """target 是 input 向右偏移一位"""
        tokenizer = tiktoken.get_encoding("gpt2")
        enc = tokenizer.encode(self.raw_text)
        enc = enc[50:]  # 取一段
        context_size = 4
        x = enc[:context_size]
        y = enc[1 : context_size + 1]
        # y 是 x 向右移一位
        self.assertEqual(x[1:], y[:-1])

    def test_demo_02_dataloader_shape(self):
        """batch_size=8, max_length=4 → 形状 [8, 4]"""
        dl = create_dataloader_v1(self.raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
        inputs, targets = next(iter(dl))
        self.assertEqual(inputs.shape, torch.Size([8, 4]))
        self.assertEqual(targets.shape, torch.Size([8, 4]))

    def test_demo_03_stride_equals_length_no_overlap(self):
        """stride=max_length 时，相邻 batch 的第一个样本不重叠"""
        dl = create_dataloader_v1(self.raw_text, batch_size=1, max_length=4, stride=4, shuffle=False)
        it = iter(dl)
        first, _ = next(it)
        second, _ = next(it)
        # 两个样本没有共同 token（stride=max_length）
        self.assertFalse(torch.equal(first, second))

    def test_demo_04_stride_1_overlap(self):
        """stride=1 时，相邻样本差一个 token"""
        dl = create_dataloader_v1(self.raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
        it = iter(dl)
        first, _ = next(it)  # [a, b, c, d]
        second, _ = next(it)  # [b, c, d, e]
        # 重叠部分: first[1:] == second[:-1]
        self.assertTrue(torch.equal(first[0, 1:], second[0, :-1]))

    # ──────────────────────── 练习 5.1 (书中 Exercise 2.2) ────────────────────
    def test_ex_01_dataloader_max_length_2(self):
        """
        练习 5.1 (书中 Exercise 2.2):
        创建一个 batch_size=4, max_length=2, stride=2 的 dataloader，
        取第一个 batch 的 input，验证形状为 [4, 2]。
        """
        # TODO: 调用 create_dataloader_v1 创建 dataloader，赋值给 dl
        dl = None  # ← 改这里
        raise NotImplementedError("TODO 5.1: 创建 max_length=2 的 dataloader")

        inputs, targets = next(iter(dl))
        self.assertEqual(inputs.shape, torch.Size([4, 2]))

    # ──────────────────────── 练习 5.2 ────────────────────────
    def test_ex_02_dataset_length(self):
        """
        练习 5.2: 用 max_length=4, stride=2 创建数据集，
        计算数据集中样本数量。

        提示: GPT-2 对 the-verdict.txt 编码后约 5145 个 token。
              样本数 ≈ (5145 - max_length) // stride
        """
        tokenizer = tiktoken.get_encoding("gpt2")

        # TODO: 创建 GPTDatasetV1，赋值给 dataset
        dataset = None  # ← 改这里
        raise NotImplementedError("TODO 5.2: 创建 dataset 并计算长度")

        # 验证样本数量合理（不要求精确，误差 ±10 即可）
        expected = (5145 - 4) // 2
        self.assertAlmostEqual(len(dataset), expected, delta=10)

    # ──────────────────────── 练习 5.3 ────────────────────────
    def test_ex_03_next_token_prediction(self):
        """
        练习 5.3: 手动实现「预测下一个 token」的打印。
        对编码后的文本，从位置 0 开始，依次展示:
          context=[token_0] → next=token_1
          context=[token_0, token_1] → next=token_2
          ...（共 4 步）

        提示: 参考第 5 节 Section 2.6 的代码。
        """
        tokenizer = tiktoken.get_encoding("gpt2")
        enc = tokenizer.encode(self.raw_text)
        enc_sample = enc[50:]
        context_size = 4

        # TODO: 循环 4 次，打印 context 和 target，不需要断言，打印即可
        raise NotImplementedError("TODO 5.3: 打印 next-token prediction 示例")


# ═════════════════════════════════════════════════════════════════════════════
# 第 6 节  词嵌入 (Token Embeddings)
# ═════════════════════════════════════════════════════════════════════════════
class Section6_TokenEmbeddings(unittest.TestCase):
    """
    【概念】Embedding 层的本质 = 可训练的查找表 (lookup table)。

    torch.nn.Embedding(vocab_size, embed_dim):
    - 权重矩阵形状: [vocab_size, embed_dim]
    - 输入 token_id → 返回对应行的向量
    - 等价于: one-hot 编码 @ 权重矩阵（但更高效）
    - 训练时通过反向传播更新权重，让语义相似的词向量相近
    """

    def test_demo_01_embedding_lookup(self):
        """embedding 层对 ID=3 的查找，等于权重矩阵第 3 行"""
        torch.manual_seed(123)
        embedding = torch.nn.Embedding(6, 3)
        result = embedding(torch.tensor([3]))
        expected = embedding.weight[3]
        self.assertTrue(torch.allclose(result[0], expected))

    def test_demo_02_batch_embedding(self):
        """对多个 ID 同时查找，输出形状 [N, embed_dim]"""
        torch.manual_seed(123)
        embedding = torch.nn.Embedding(6, 3)
        ids = torch.tensor([2, 3, 5, 1])
        result = embedding(ids)
        self.assertEqual(result.shape, torch.Size([4, 3]))

    def test_demo_03_gpt2_scale(self):
        """GPT-2 规模: 50257 词 × 256 维"""
        embedding = torch.nn.Embedding(50257, 256)
        self.assertEqual(embedding.weight.shape, torch.Size([50257, 256]))

    # ──────────────────────── 练习 6.1 ────────────────────────
    def test_ex_01_create_embedding(self):
        """
        练习 6.1: 创建一个 vocab_size=10, embed_dim=4 的 Embedding 层。
        验证: 权重矩阵形状为 [10, 4]，且对 ID=7 的查找结果形状为 [1, 4]。
        """
        torch.manual_seed(42)

        # TODO: 创建 Embedding 层，赋值给 emb
        emb = None  # ← 改这里
        raise NotImplementedError("TODO 6.1: 创建 Embedding 层")

        self.assertEqual(emb.weight.shape, torch.Size([10, 4]))
        out = emb(torch.tensor([7]))
        self.assertEqual(out.shape, torch.Size([1, 4]))

    # ──────────────────────── 练习 6.2 ────────────────────────
    def test_ex_02_embedding_is_lookup(self):
        """
        练习 6.2: 验证 Embedding 层等价于查表操作。
        对同一个 ID，embedding(id) 的结果应等于 embedding.weight[id]。
        """
        torch.manual_seed(0)
        emb = torch.nn.Embedding(100, 8)
        target_id = 42

        # TODO: 用两种方式获取 ID=42 的向量，赋值给 via_forward 和 via_weight
        via_forward = None  # ← emb(torch.tensor([42])) 的结果（第 0 行）
        via_weight = None  # ← emb.weight[42]
        raise NotImplementedError("TODO 6.2: 验证 embedding 等价于查表")

        self.assertTrue(torch.allclose(via_forward, via_weight))


# ═════════════════════════════════════════════════════════════════════════════
# 第 7 节  位置编码 (Positional Embeddings)
# ═════════════════════════════════════════════════════════════════════════════
class Section7_PositionalEmbeddings(unittest.TestCase):
    """
    【概念】为什么需要位置编码？
    Embedding 层对同一个 token 总是返回相同的向量，
    不管它在句子的第几个位置。但「位置」对理解语义很重要：
    "The cat sat" vs "sat The cat"

    GPT-2 使用「绝对位置编码」(Absolute Positional Embedding):
    - 另一个 Embedding(context_length, embed_dim)
    - 对位置 [0, 1, 2, ..., context_length-1] 查表
    - 最终输入 = token_embedding + pos_embedding
    """

    def test_demo_01_pos_embedding_shape(self):
        """位置编码矩阵形状: [context_length, embed_dim]"""
        context_length = 4
        embed_dim = 256
        pos_emb = torch.nn.Embedding(context_length, embed_dim)
        positions = torch.arange(context_length)
        result = pos_emb(positions)
        self.assertEqual(result.shape, torch.Size([4, 256]))

    def test_demo_02_add_token_and_pos(self):
        """
        token_embedding [batch, seq, dim] + pos_embedding [seq, dim]
        → 广播相加，形状仍为 [batch, seq, dim]
        """
        torch.manual_seed(0)
        batch_size, seq_len, embed_dim = 8, 4, 256
        token_emb = torch.nn.Embedding(50257, embed_dim)
        pos_emb = torch.nn.Embedding(seq_len, embed_dim)

        dummy_ids = torch.randint(0, 50257, (batch_size, seq_len))
        token_out = token_emb(dummy_ids)  # [8, 4, 256]
        pos_out = pos_emb(torch.arange(seq_len))  # [4, 256]

        input_emb = token_out + pos_out  # 广播
        self.assertEqual(input_emb.shape, torch.Size([8, 4, 256]))

    # ──────────────────────── 练习 7.1 ────────────────────────
    def test_ex_01_combine_embeddings(self):
        """
        练习 7.1: 把 token embedding 和 positional embedding 相加，
        得到最终的输入向量。

        给定:
          vocab_size=50257, context_length=8, embed_dim=64, batch_size=4

        步骤:
          1. 创建 token_embedding_layer (50257, 64)
          2. 创建 pos_embedding_layer (8, 64)
          3. 随机生成 token_ids，形状 [4, 8]
          4. 计算 input_embeddings = token_emb + pos_emb
          5. 验证 input_embeddings.shape == [4, 8, 64]
        """
        torch.manual_seed(7)
        vocab_size = 50257
        context_length = 8
        embed_dim = 64
        batch_size = 4

        # TODO: 完成以下步骤
        token_embedding_layer = None  # ← 步骤 1
        pos_embedding_layer = None  # ← 步骤 2
        token_ids = None  # ← 步骤 3: torch.randint(...)
        input_embeddings = None  # ← 步骤 4
        raise NotImplementedError("TODO 7.1: 组合 token 和位置嵌入")

        self.assertEqual(input_embeddings.shape, torch.Size([4, 8, 64]))

    # ──────────────────────── 练习 7.2 ────────────────────────
    def test_ex_02_same_token_different_positions(self):
        """
        练习 7.2: 验证「同一个 token，在不同位置，最终向量不同」。

        用同一个 token_id (如 42) 分别放在位置 0 和位置 3，
        两者的 input_embedding 应该不同（因为位置编码不同）。
        """
        torch.manual_seed(1)
        emb_dim = 16
        context_len = 4
        token_emb = torch.nn.Embedding(100, emb_dim)
        pos_emb = torch.nn.Embedding(context_len, emb_dim)

        token_id = 42

        # TODO: 计算 token_id 在 position=0 时的 input_embedding → vec_at_0
        # TODO: 计算 token_id 在 position=3 时的 input_embedding → vec_at_3
        vec_at_0 = None  # ← token_emb(42) + pos_emb(0)
        vec_at_3 = None  # ← token_emb(42) + pos_emb(3)
        raise NotImplementedError("TODO 7.2: 验证位置改变向量")

        # 同一 token，不同位置，向量应该不同
        self.assertFalse(torch.allclose(vec_at_0, vec_at_3))


# ═════════════════════════════════════════════════════════════════════════════
# 第 8 节  综合练习：完整数据管道
# ═════════════════════════════════════════════════════════════════════════════
@unittest.skipUnless(TIKTOKEN_AVAILABLE, "需要安装 tiktoken")
class Section8_EndToEnd(unittest.TestCase):
    """
    【综合】把所有步骤串起来：
    原始文本 → BPE 编码 → 滑动窗口 DataLoader → Embedding → 输入向量

    这就是 GPT-2 训练时数据处理的完整流程！
    """

    def test_demo_01_full_pipeline(self):
        """完整的数据预处理管道"""
        raw_text = load_verdict()

        # 1. 创建 DataLoader
        dl = create_dataloader_v1(raw_text, batch_size=4, max_length=8, stride=8, shuffle=False)
        inputs, targets = next(iter(dl))
        self.assertEqual(inputs.shape, torch.Size([4, 8]))

        # 2. Token Embedding
        token_emb = torch.nn.Embedding(50257, 256)
        token_vecs = token_emb(inputs)
        self.assertEqual(token_vecs.shape, torch.Size([4, 8, 256]))

        # 3. Positional Embedding
        pos_emb = torch.nn.Embedding(8, 256)
        pos_vecs = pos_emb(torch.arange(8))
        self.assertEqual(pos_vecs.shape, torch.Size([8, 256]))

        # 4. 最终输入向量
        input_vecs = token_vecs + pos_vecs
        self.assertEqual(input_vecs.shape, torch.Size([4, 8, 256]))

    # ──────────────────────── 综合练习 ────────────────────────
    def test_ex_full_pipeline_from_scratch(self):
        """
        综合练习: 从零实现完整管道，不使用上面的辅助函数。

        要求:
        1. 读取 the-verdict.txt
        2. 用 tiktoken GPT-2 tokenizer 编码
        3. 手动实现滑动窗口（max_length=16, stride=16），取前 4 个样本
        4. 把 inputs 通过 token_embedding(vocab_size=50257, dim=128)
        5. 加上 positional_embedding(max_length=16, dim=128)
        6. 断言最终形状为 [4, 16, 128]
        """
        import tiktoken as tk

        # TODO: 步骤 1: 读文本
        raw_text = None  # ← load_verdict()

        # TODO: 步骤 2: BPE 编码
        tokenizer = None  # ← tk.get_encoding("gpt2")
        token_ids = None  # ← tokenizer.encode(raw_text, ...)

        # TODO: 步骤 3: 滑动窗口，取前 4 个 input 和 target
        max_length = 16
        stride = 16
        inputs_list = []
        # for i in range(...):
        #     inputs_list.append(...)
        # inputs = torch.stack(inputs_list[:4])

        inputs = None  # ← torch.stack(...)

        # TODO: 步骤 4 & 5: 创建并使用 embedding 层
        token_emb_layer = None
        pos_emb_layer = None
        input_embeddings = None

        raise NotImplementedError("TODO: 综合练习，完整实现数据管道")

        self.assertEqual(input_embeddings.shape, torch.Size([4, 16, 128]))


# ═════════════════════════════════════════════════════════════════════════════
# 运行入口
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Ch02 互动学习测试")
    print("  演示测试 (test_demo_*): 展示正确行为")
    print("  练习测试 (test_ex_*) : 需要你填写 TODO 完成")
    print("  遇到 NotImplementedError = 找到 TODO 补全代码！")
    print("=" * 60)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 按节顺序加载
    for cls in [
        Section1_Tokenization,
        Section2_VocabAndIDs,
        Section3_SpecialTokens,
        Section4_BPE,
        Section5_SlidingWindow,
        Section6_TokenEmbeddings,
        Section7_PositionalEmbeddings,
        Section8_EndToEnd,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
