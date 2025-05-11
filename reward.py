import re

def extract_number_from_boxed_string(s):
    s = s.replace('\\!', '')
    number = re.search(r'boxed[^\d]*(\d[\d,]*)', s)
    return number[1].replace(',', '') if number else None

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_number_from_boxed_string(r) for r in responses]

    q = prompts[0][-1]['content']
    print('-' * 20, f"Question:\n{q}", f"\nResponse:\n{responses[0]}",f"\nAnswer:\n{answer[0].replace(" ", "").replace(',', '')}", 
        f"\nExtracted:\n{extracted_responses[0]}")

    return [2.0 if r == a.replace(" ", "").replace(',', '') else 0.0 for r, a in zip(extracted_responses, answer)]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r'boxed\{[^}]*\}'
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r) for r in responses]
    return [1.0 if match else 0.0 for match in matches]

def contains_boxed_structure(s):
    pattern = r'boxed\{[^}]*\}'
    return 1 if re.search(pattern, s) else 0

REWARD_FUNCS = {
    'correctness_reward_func': correctness_reward_func,
    'strict_format_reward_func': strict_format_reward_func
}
