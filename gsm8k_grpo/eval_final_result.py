import json
import re
from typing import Optional, List, Dict, Any
from pathlib import Path

BOXED_RE  = re.compile(r"""\\boxed\{\s*([-+]?\d+)\s*\}""")  
HASH_RE   = re.compile(r"""####\s*([-+]?\d+)\s*$""", re.MULTILINE)
def extract_int(pattern: re.Pattern, text: str) -> Optional[int]:
    matches = pattern.findall(text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None

def score_records(records: List[Dict[str, Any]]) -> float:
    correct = 0
    for rec in records:
        pred = extract_int(BOXED_RE, rec.get("answer", ""))
        gold = extract_int(HASH_RE, rec.get("right_answer", ""))
        rec["predicted"]  = pred
        rec["gold"]       = gold
        rec["is_correct"] = (pred is not None
                             and gold is not None
                             and pred == gold)
        if rec["is_correct"]:
            correct += 1
    return correct / len(records) if records else 0.0

def main() -> None:
    in_path  = Path("/home/tlmsq/rlrover/result-1800.json")
    out_path = Path("/home/tlmsq/rlrover/result-1800-out.json")

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    accuracy = score_records(data)
    result = {
        "records": data,
        "accuracy": round(accuracy, 4)   
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✔ 评分完成，正确率 = {accuracy:.2%}")
    print(f"✔ 结果已写入: {out_path.resolve()}")

if __name__ == "__main__":
    main()
    # Raw: 36.5
    # One-Epoch GRPO: 46.78