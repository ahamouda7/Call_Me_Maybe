import json
import numpy as np
from numpy import ndarray


def load_vocabulary(vocab_path):
    try:
        with open(vocab_path, "r", encoding="utf-8") as file:
            token_id = json.load(file)
        id_token = {v: k for k, v in token_id.items()}
        return token_id, id_token
    except Exception as e:
        print(e)
        raise


def get_valid_next_tokens(
    current_prefix: str, functions_names: list[str],
    token_id: dict[str, int]
) -> set[int]:
    valid_strings = [
        func_name
        for func_name in functions_names
        if func_name.startswith(current_prefix)
    ]
    valid_tokens = set()
    for token, id in token_id.items():
        new_prefix = current_prefix + token

        for s in valid_strings:
            if s.startswith(new_prefix):
                valid_tokens.add(id)
                break
    return valid_tokens


def mask_logits(
    logits: ndarray,
    valid_token_ids: set[int]
) -> ndarray:
    logits = np.array(logits)

    logits_shadow = np.full_like(logits, -np.inf)
    valid_ids = list(valid_token_ids)
    logits_shadow[valid_ids] = logits[valid_ids]
    return logits_shadow
