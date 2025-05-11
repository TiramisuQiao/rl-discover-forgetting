from trl import GRPOConfig, GRPOTrainer
from copy import deepcopy
import torch
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoTokenizer

class ReplayBuffer:
    def __init__(self):
        self.data_by_task = {}

    def update(self, task_id, dataset):
        self.data_by_task[task_id] = dataset

    def get_all_data(self):
        all_data = []
        all_data.extend(iter(self.data_by_task.values()))
        return concatenate_datasets(all_data) if all_data else None

def gsm8k_reward_fn(completions, prompts=None, **kwargs):
    """Reward 1.0 if numeric answer matches, else 0.0"""
    rewards = []
    targets = prompts.get('answer', []) if prompts else []
    for pred, targ in zip(completions, targets):
        try:
            pred_num = float(pred.strip())
            targ_num = float(targ)
            rewards.append(1.0 if abs(pred_num - targ_num) < 1e-3 else 0.0)
        except:
            rewards.append(0.0)
    return rewards


def mmlu_reward_fn(completions, prompts=None, **kwargs):
    """Reward 1.0 if predicted choice letter matches answer, else 0.0"""
    rewards = []
    targets = prompts.get('answer', []) if prompts else []
    for pred, ans in zip(completions, targets):
        pred_char = pred.strip()[0] if pred.strip() else ''
        rewards.append(1.0 if pred_char.upper() == ans.upper() else 0.0)
    return rewards

class ContinualGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model_name_or_path,
        tokenizer,
        args,
        train_dataset,
        eval_dataset=None,
        reward_fns=None,
        task_types=None,
        callbacks=None,
        **kwargs
    ):
        # Initialize base GRPOTrainer with processing_class
        super().__init__(
            model=model_name_or_path,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            reward_funcs=self.compute_rewards,
            processing_class=tokenizer,
            callbacks=callbacks,
            **kwargs
        )
        self.task_id = 0
        self.reward_fns = reward_fns or {}
        self.task_types = task_types or {}
        self.old_policy_snapshots = {}
        self.replay_buffer = ReplayBuffer()

    def update_task(self, new_task_id, new_dataset):
        if self.task_id != new_task_id:
            # snapshot current policy
            self.old_policy_snapshots[self.task_id] = deepcopy(self.model).eval().cpu()
            self.replay_buffer.update(self.task_id, new_dataset)
            self.task_id = new_task_id

    def compute_rewards(self, completions, prompts=None, **kwargs):
        task_type = self.task_types.get(self.task_id)
        reward_fn = self.reward_fns.get(task_type)
        if reward_fn is None:
            return [0.0] * len(completions)
        return reward_fn(completions, prompts=prompts, **kwargs)

    def add_replay_data(self, dataset):
        replay_ds = self.replay_buffer.get_all_data()
        if replay_ds:
            return concatenate_datasets([dataset, replay_ds]).shuffle(seed=42)
        return dataset


def load_gsm8k_dataset():
    ds = load_dataset("openai/gsm8k", "main")["train"].shuffle(seed=42).select(range(100))
    def fmt(ex):
        return {
            'prompt': f"Question: {ex['question']}\nAnswer:",
            'answer': ex['answer']
        }
    return ds.map(fmt)


def load_mmlu_dataset():
    ds = load_dataset("cais/mmlu")["test"].filter(lambda x: x['subject'] == 'european_history').shuffle(seed=42).select(range(5))
    def fmt(ex):
        choice_map = ['A', 'B', 'C', 'D']
        choices = ex['choices']
        choice_text = "\n".join([f"{c}. {t}" for c, t in zip(choice_map, choices)])
        return {
            'prompt': f"{ex['question']}\n{choice_text}\nAnswer:",
            'answer': ex['answer']
        }
    return ds.map(fmt)


def run_continual_grpo():
    gsm8k_ds = load_gsm8k_dataset()
    mmlu_ds = load_mmlu_dataset()
    task_datasets = {0: gsm8k_ds, 1: mmlu_ds}
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = GRPOConfig(
        output_dir="./grpo_outputs",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        max_steps=1000,
        logging_steps=10,
        report_to=["swanlab"],
        run_name="continual-grpo-qwen",
        save_strategy="no"
    )
    reward_fns = {"gsm8k": gsm8k_reward_fn, "mmlu": mmlu_reward_fn}
    task_types = {0: "gsm8k", 1: "mmlu"}
    trainer = ContinualGRPOTrainer(
        model_name_or_path=model_name,
        tokenizer=tokenizer,
        args=config,
        train_dataset=gsm8k_ds,
        eval_dataset=None,
        reward_fns=reward_fns,
        task_types=task_types,
        callbacks=[SwanLabCallback()]
    )

    # train on GSM8K
    # trainer.train()

    # switch to MMLU
    trainer.update_task(1, mmlu_ds)
    trainer.train_dataset = trainer.add_replay_data(mmlu_ds)
    trainer.train()

if __name__ == '__main__':
    run_continual_grpo()
