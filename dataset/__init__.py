from transformers import AutoTokenizer
from abc import ABC
import torch


class Sampler(ABC):
    """采样器基类：用于将文本格式的训练数据，转换成Token序列"""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        """
        采样器基类 - 构造函数

        Args:
            tokenizer (AutoTokenizer): 分词器
            max_length (int): 最大Token长度，默认512

        Returns:
            None: 无返回值
        """

        self.tokenizer = tokenizer
        self.max_length = max_length


class PretrainSampler(Sampler):
    """预训练采样器：用于将数据集转换成Token序列，预训练数据格式为jsonl，格式如下：

    {"text": "<|im_start|>请提取出文章中关于人工智能在医疗上的作用。人工智能还将会在医疗、农业、安全防范等领域发挥重要的作用。<|im_end|>"}
    {"text": "<|im_start|>请从这首诗选出三句话，重新组成另一首新的诗歌。秋天美景尽收眼底，枫叶红艳色。碧空如水流，心情格外好。<|im_end|>"}
    ...

    Example:

    ```python
    from datasets import load_dataset

    train_ds = load_dataset("json", data_files="dataset/pretrain.jsonl", split="all")
    train_ds = train_ds.map(PretrainSampler(tokenizer, max_length=512), batched=False)
    ```
    """

    def __call__(self, sample: dict):
        assert "text" in sample, f"训练数据格式不正确（没有text字段）: {sample}"

        # 使用分词器分词，将文本编码成ID，如果编码后长度大于max_length则截断，如果小于max_length则添加padding
        encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = input_ids != self.tokenizer.pad_token_id  # padding不参与训练

        # 转成Tensor对象，模型的输入是X，输出是Y，X和Y的序列相差1位（模型总是根据现有的Token序列来预测下一个Token）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return {"X": X, "Y": Y, "loss_mask": loss_mask}


class SFTSampler(Sampler):
    """SFT采样器：用于将数据集转换成Token序列，SFT数据格式为jsonl，格式如下：

    {"conversations": [{"role": "user", "value": "请提取出文章中关于人工智能在医疗上的作用。"}, {"role": "assistant", "value": "人工智能还将会在医疗、农业、安全防范等领域发挥重要的作用。"}]}
    {"conversations": [{"role": "user", "value": "请从这首诗选出三句话，重新组成另一首新的诗歌。"}, {"role": "assistant", "value": "秋天美景尽收眼底，枫叶红艳色。碧空如水流，心情格外好。"}]}
    ...
    """

    def __init__(self, tokenizer, max_length=1024):
        super().__init__(tokenizer, max_length)
        self.bos_id = tokenizer(
            "<|im_start|>assistant", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer("<|im_end|>", add_special_tokens=False).input_ids

    def _generate_loss_mask(self, input_ids):
        """生成动态损失掩码，只计算AI回答部分的损失"""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(
                    start + 1, min(end + len(self.eos_id) + 1, self.max_length)
                ):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __call__(self, sample: dict):
        assert "conversations" in sample, f"训练数据格式不正确（没有conversations字段）: {sample}"

        # 构建对话提示
        prompt = self.tokenizer.apply_chat_template(
            sample["conversations"], tokenize=False, add_generation_prompt=False
        )
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(
            loss_mask[1:], dtype=torch.long
        )  # 对齐预测位置 - 只有AI的部分才计算损失

        return {"X": X, "Y": Y, "loss_mask": loss_mask}
