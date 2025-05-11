from typing import Optional
from datasets import load_dataset,IterableDataset
from modelscope.msdatasets import MsDataset

SYSTEM_PROMPT = """you're a helpful assistant."""

XML_COT_FORMAT = """
{think}

boxed{{{answer}}}

"""

def extract_answer(text: str) -> Optional[str]:
    return None if "####" not in text else text.split("####")[1].strip()


def extract_cot(text: str) -> str:
    if "####" not in text:
        return ""
    cot = text.split("####")
    #print(XML_COT_FORMAT.format(think=cot[0].strip(), answer=cot[1].strip()))
    return XML_COT_FORMAT.format(think=cot[0].strip(), answer=cot[1].strip())


def get_gsm8k_dataset(split="train", sft=False, cache_dir=None, first_half=False, second_half=False) -> IterableDataset:
    # Define the file paths for the local datasets
    local_data_paths = {
        'train': './Gsm8k/train-00000-of-00001.parquet',
        #'train': './Gsm8k/train_r1_distill_final_1500.parquet',
        'test': './Gsm8k/test-00000-of-00001.parquet'
    }
    
    # Load the dataset from the local file
    data = load_dataset('parquet', data_files=local_data_paths[split], cache_dir=cache_dir)["train"]

    if first_half:
        data = data.shard(2, 0)
    elif second_half:
        data = data.shard(2, 1)

    if not sft:
        data = data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': extract_answer(x['answer'])
        })
    else:
        data = data.map(lambda x: {
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']},
                {'role': 'assistant', 'content': extract_cot(x['answer'])},
            ]
        })
    return data

def get_gsm8k_recover(
    split: str = "train",
    sft: bool = False,
    cache_dir: Optional[str] = None,
    num_samples: int = 100,
    seed: int = 42
) -> IterableDataset:
    data = get_gsm8k_dataset(
        split=split,
        sft=sft,
        cache_dir=cache_dir
    )
    data = data.shuffle(seed=seed)
    data = data.select(range(min(num_samples, len(data))))

    return data
    
