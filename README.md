# rl-discover-forgetting

A collection of scripts and experiments to study “false forgetting” in LLMs under RL fine-tuning (GRPO) and recovery techniques.


## 📂 Repository Structure

```

rl-discover-forgetting/
├── paper/                   # Reference PDFs and BibTeX
├── gsm8k\_grpo/              # GSM8K GRPO training & evaluation
│   └── eval\_gsm8k.py
├── concept\_vector/          # Probing: locate\_concept\_dims.py
├── attention\_map/           # Attention‐map comparison scripts
├── test\_avoid/              # Replay‐based mitigation of GRPO forgetting
├── main.py                  # GRPO training & recovery entry point
├── sft\_mmlu.py              # SFT training on MMLU
└── README.md                # This file

```

---

## 🚀 Getting Started


- Python 3.10+
- CUDA toolkit (for GPU acceleration)
- `vLLM` & `transformers` with BF16 support

  ```bash
  uv sync
````


---

## 🎯 Training

### 1. GRPO on GSM8K

```bash
python main.py \
  --task grpo_train \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --bf16 \
  --use_vllm \
  --checkpoint_dir outputs/Qwen-2.5B-GRPO \
  --save_strategy epoch
```

### 2. SFT on MMLU

```bash
python sft_mmlu.py
```

### 3. GRPO Recovery (after SFT)

```bash
python main.py \
  --task=grpo_train-recover \
  --model_name_or_path=path/for/aftersft \
  --bf16 \
  --use_vllm \
  --checkpoint_dir=outputs/Qwen-0.5B-GRPO-recover
```

---

## 📊 Evaluation

### GSM8K

```bash
cd gsm8k_grpo
python eval_gsm8k.py \
  --model_checkpoint ../outputs/Qwen-2.5B-GRPO
```

### MMLU

#### Baseline

```bash
evalscope eval \
  --model Qwen/Qwen2.5-0.5B \
  --datasets mmlu \
  --limit 5
```

#### After GRPO

```bash
evalscope eval \
  --model outputs/Qwen-0.5B-GRPO-Recover \
  --datasets mmlu \
  --limit 5
```

---

## 🔍 Probing & Analysis

* **Locate concept‐specific dimensions**

  ```bash
  python concept_vector/locate_concept_dims.py
  ```
* **Compare attention maps (pre vs. post GRPO)**

  ```bash
  python attention_map/compare_attention.py
  ```

---

## 🛡 Mitigating “False Forgetting”

Replay‐based suppression of GRPO‐induced forgetting:

```bash
python test_avoid/run.py
```

---
## 📑 paper/
All relevant literature and BibTeX entries are stored under `paper/`.
---
