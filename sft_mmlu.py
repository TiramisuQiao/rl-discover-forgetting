import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from swanlab.integration.huggingface import SwanLabCallback
from datasets import load_dataset

MODEL_PATH = "/root/autodl-tmp/qwen_grpo_gsm8k/outputs/Qwen-0.5B-GRPO/checkpoint-1868"
CHECKPOINT_DIR = "outputs/sft_mmlu_pro"
CACHE_DIR = None 

LEARNING_RATE = 5e-5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
LR_SCHEDULER_TYPE = "cosine"
LOGGING_STEPS = 50
BF16 = True
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
MAX_SEQ_LENGTH = 512
EPOCHS = 3
SAVE_STEPS = 500
SAVE_STRATEGY = "steps"
MAX_GRAD_NORM = 1.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_args = SFTConfig(
        output_dir=CHECKPOINT_DIR,
        learning_rate=LEARNING_RATE,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        logging_steps=LOGGING_STEPS,
        bf16=BF16,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_seq_length=MAX_SEQ_LENGTH,
        num_train_epochs=EPOCHS,
        save_steps=SAVE_STEPS,
        save_strategy=SAVE_STRATEGY,
        max_grad_norm=MAX_GRAD_NORM,
        log_on_each_node=False,
        report_to="none",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if BF16 else None,
        cache_dir=CACHE_DIR
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        cache_dir=CACHE_DIR
    )
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset("ankner/mmlu-pro-sft")
    train_ds = ds["train"]
    eval_ds = ds["test"]

    def preprocess(batch):
        texts = []
        for inp, resp in zip(batch["input"], batch["response"]):
            prompt = inp.strip()
            full = prompt + "\n" + resp.strip()
            texts.append(full)
        toks = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
        )
        toks["labels"] = toks["input_ids"].copy()
        return toks

    train_tok = train_ds.map(
        preprocess,
        batched=True,
        remove_columns=train_ds.column_names
    )
    eval_tok = eval_ds.map(
        preprocess,
        batched=True,
        remove_columns=eval_ds.column_names
    )

    swanlab_cb = SwanLabCallback(
        project="huggingface",
        experiment_name="Qwen2.5-mmlu-sft"
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        callbacks=[swanlab_cb]
    )

    trainer.train()
    trainer.save_model()
    print(f"✅ SFT 完成，模型保存在：{CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()