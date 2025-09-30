from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer
from transformers import AutoTokenizer
import argparse
import random
import json
import os


def read_texts_from_jsonl(data_files):
    """从jsonl文件中读取训练数据"""
    for file in data_files:
        with open(file, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                line = line.strip()
                if not line:
                    continue
                    
                data = json.loads(line)
                if "text" in data:
                    yield data["text"]


def train_tokenizer(data_files: list[str]):
    """训练分词器
    
    Args:
        data_files (list[str]): 训练文件列表
    """    
    print(f"Training tokenizer on data from: ")
    for file in data_files:
        print(f" - {file}")
    print("This may take a while...\n")

    # 初始化tokenizer - 采用BPE算法
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token
    special_tokens = ["<|padding|>", "<|im_start|>", "<|im_end|>", "<|think_start|>", "<|think_end|>"]

    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=8192,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # 读取文本数据
    texts = read_texts_from_jsonl(data_files)

    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    assert tokenizer.token_to_id("<|padding|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2
    assert tokenizer.token_to_id("<|think_start|>") == 3
    assert tokenizer.token_to_id("<|think_end|>") == 4

    # 保存tokenizer
    tokenizer_dir = "model"
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))

    print("Tokenizer training completed and saved.\n")


def eval_tokenizer():
    """评估分词器"""
    # 加载预训练的Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("model")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": "你来自哪里？"},
        {"role": "assistant", "content": "我来自地球"},
    ]
    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"Prompt: {new_prompt}")

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print("Tokenizer实际词表长度: ", actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print("Encoder长度: ", len(model_inputs["input_ids"]))

    input_ids = model_inputs["input_ids"]
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print("Decoder和原始文本是否一致: ", response == new_prompt)


def main(data_files):
    """入口函数"""
    train_tokenizer(data_files)
    eval_tokenizer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Tokenizer", description="Train a tokenizer")
    parser.add_argument("-i", "--data-file", type=str, default=["dataset/pretrain.jsonl"], nargs="+")
    parser.add_argument("-s", "--seed", type=int, default=2025)
    args = parser.parse_args()

    if args.seed > 0:
        random.seed(args.seed) # 固定Seed
        
    main(args.data_file)
