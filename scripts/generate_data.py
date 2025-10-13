from openai import OpenAI
import re


client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")


def do_completion(prompt: str | list, system_prompt: str = None, model: str = "qwen3:32b", verbose: bool = True) -> str:
    if isinstance(prompt, str):
        if not system_prompt:
            system_prompt = """你是一名专业的大语言模型预训练数据生成助手，负责产出高质量、多样化、真实可信的文本数据，用于大语言模型的预训练阶段。
在生成内容时，你必须遵守以下原则：
    1. 真实性：基于已知或合理推测的真实信息，不捏造事实，不虚构来源。
    2. 多样性：覆盖广泛主题（科学、人文、生活、对话、叙事、推理等）与多种表达风格（正式、口语化、叙事、说明等）。
    3. 自然流畅：文本应符合人类的语言习惯，语法正确，结构清晰。
    4. 安全合规：不包含违法、敏感、有害内容；尊重隐私。
    5. 可用性：内容格式明确、无多余标记，适合直接用于预训练阶段的数据集。
你的回答应简洁清晰，但在必要时可提供完整、丰富的细节，以帮助模型建立通用语言理解能力。"""

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    else:
        messages = prompt

    if verbose:
        print(f">>> {prompt}")

    response: str = ""
    for chunk in client.chat.completions.create(messages=messages, model=model, stream=True):
        if content := getattr(chunk.choices[0].delta, "content", None):
            response += content
            if verbose:
                print(content, end="", flush=True)

    if verbose:
        print("\n")

    result = re.match(r"<think>(.*)</think>(.*)", response, re.DOTALL)
    if result:
        return result.group(1).strip(), result.group(2).strip()
    return "", response


def generate_sft():
    # _, answer = do_completion(f"什么是“现代桥梁”？")
    _, answer = do_completion(f"什么是“《许生传》”？")


if __name__ == "__main__":
    generate_sft()