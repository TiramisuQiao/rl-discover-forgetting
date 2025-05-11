from transformers import AutoModelForCausalLM, AutoTokenizer

from rich.console import Console
from rich.text import Text

PRE_MODEL = "Qwen/Qwen2.5-0.5B" 
POST_MODEL = "/home/tlmsq/rlrover/checkpoint-1868"
TEXT =  """
This question refers to the following information.
Read the the following quotation to answer questions.
The various modes of worship which prevailed in the Roman world were all considered by the people as equally true; by the philosopher as equally false; and by the magistrate as equally useful.
Edward Gibbon, The Decline and Fall of the Roman Empire, 1776–1788
Gibbon's interpretation of the state of religious worship in ancient Rome could be summarized as", "A": "In ancient Rome, religious worship was decentralized and tended to vary with one's social position.", "B": "In ancient Rome, religious worship was the source of much social tension and turmoil.", "C": "In ancient Rome, religious worship was homogeneous and highly centralized.", "D": "In ancient Rome, religious worship was revolutionized by the introduction of Christianity."
"""


LAYER = -1             
DEVICE = "cuda"     

# TARGET A
def get_attentions(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_attentions=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # attentions: tuple(num_layers, batch, heads, seq_len, seq_len)
    attns = outputs.attentions
    return attns, tokens


def compute_token_importance(attn_tensor):
    # attn_tensor: heads x seq_len x seq_len
    # 平均 heads -> seq_len x seq_len, 再平均 from-axis -> seq_len
    mean_heads = attn_tensor.mean(dim=0)               # seq_len x seq_len
    importance = mean_heads.mean(dim=0)                # seq_len
    imp = importance / importance.max() if importance.max() > 0 else importance
    return imp.tolist()


def print_attention_map(console, title, tokens, scores):
    console.rule(f"{title}")
    text = Text()
    for token, score in zip(tokens, scores):
        intensity = int(score * 255)
        hex_color = f"#{intensity:02x}00{(255-intensity):02x}"
        text.append(token + " ", style=f"on {hex_color}")
    console.print(text)


def compare_and_display(pre_model_name, post_model_name, text, layer, device="cpu"):
    console = Console()
    tokenizer = AutoTokenizer.from_pretrained(pre_model_name)
    pre_model = AutoModelForCausalLM.from_pretrained(pre_model_name).to(device)
    post_model = AutoModelForCausalLM.from_pretrained(post_model_name).to(device)

    pre_attns, tokens = get_attentions(pre_model, tokenizer, text, device)
    post_attns, _ = get_attentions(post_model, tokenizer, text, device)

    num_layers = len(pre_attns)
    layer_idx = layer if layer >= 0 else num_layers - 1

    # 提取指定层的 attention heads
    pre_tensor = pre_attns[layer_idx][0]   # heads x seq_len x seq_len
    post_tensor = post_attns[layer_idx][0]

    # 计算 token 重要性分数
    pre_scores = compute_token_importance(pre_tensor)
    post_scores = compute_token_importance(post_tensor)

    # 打印带颜色的 attention map
    print_attention_map(console, "Pre-RL Attention Map", tokens, pre_scores)
    print_attention_map(console, "Post-RL Attention Map", tokens, post_scores)

# 直接运行，无需外部输入
compare_and_display(
    PRE_MODEL,
    POST_MODEL,
    TEXT,
    layer=LAYER,
    device=DEVICE
)
