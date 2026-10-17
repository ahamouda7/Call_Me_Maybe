import re
import numpy as np
from typing import Any
from shema import ResultValidation
from c_encoding import get_valid_next_tokens, mask_logits


def build_prompt(user_prompt: str, functions: list) -> str:
    result = "You are a function calling assistant.\n"
    result += "Available functions:\n"

    for func in functions:
        name = func.name
        desc = func.description
        params = []
        for k, v in func.parameters.items():
            params.append(f"{k} ({v.type})")
            result += f"{name} - {desc} - Parameters: {', '.join(params)}\n"

    result += f"\nUser request: {user_prompt}\n"
    result += '\nResponse:\n'

    return result


# def is_valid_prompt(prompt: str) -> bool:
#     if len(prompt.strip()) < 3:
#         return False
#
#     words = prompt.strip().split()
#     real_words = [
#         w for w in words if w.isalpha() and len(w) > 12 and not w.isdigit()]
#     if len(real_words) > 0:
#         return False
#     return True


def generate_stops_and_numbers(
    token_id: dict[str, int]
) -> dict[str, set[int]]:
    stop_token = set()
    stop = [",", "}", "Ġ", " ", " ,", "\n"]
    stop_token = {token_id[c] for c in stop if c in token_id}
    numbers_tokens = set()
    for k, v in token_id.items():
        if (all(c in "0123456789.-" for c in k)
                and k.count('.') <= 1 and k.count('-') <= 1):
            numbers_tokens.add(v)
    result = {
        "stop_token": stop_token,
        "all_tokens": numbers_tokens
    }
    return result


def generate_string_value(
    input_ids: list[int],
    model: Any,
    id_token: dict[int, str]
) -> tuple[str, list[int]]:
    gen_id: list[int] = []
    result: str = ""
    while True:
        full_id = input_ids + gen_id
        logits = model.get_logits_from_input_ids(full_id)
        max_token: int = int(np.argmax(logits))
        if '"' in id_token[int(max_token)]:
            break
        gen_id.append(max_token)
        result += id_token[max_token]

    return result.replace("Ġ", " ").replace("\\'", "'"), gen_id


def generate_number_value(
    input_ids: list[int],
    model: Any,
    token_id: dict[str, int],
    id_token: dict[int, str]
) -> tuple[float, list[int]]:
    all_valid = generate_stops_and_numbers(token_id)
    curr_number = ""
    gen_ids: list[int] = []
    while True:
        full_ids = input_ids + gen_ids
        logits = model.get_logits_from_input_ids(full_ids)
        masked = mask_logits(
            logits, all_valid["all_tokens"] | all_valid["stop_token"])
        max_token = int(np.argmax(masked))
        token_str = id_token[max_token]
        if max_token in all_valid["stop_token"]:
            break
        gen_ids.append(max_token)
        curr_number += token_str
    curr_number = curr_number.strip()
    try:
        val = float(curr_number) if curr_number else 0.0
    except ValueError:
        val = 0.0

    return val, gen_ids,


def generate_function_call(
    prompt,
    functions,
    model,
    token_id,
    id_token
):
    # if not is_valid_prompt(prompt.prompt):
    #     return ResultValidation(
    #         prompt=prompt.prompt, name="Invalid!",
    #         parameters={"error": "prompt not recognized"})

    prompt_text = build_prompt(prompt, functions)
    input_ids = model.encode(prompt_text)[0].tolist()

    # Model's Response Prefix
    input_ids.append(token_id['{'])
    input_ids.append(token_id['"'])
    input_ids.append(token_id['name'])
    input_ids.append(token_id['":'])
    input_ids.append(token_id['Ġ'])
    input_ids.append(token_id['"'])

    # Constrained Decoding:
    curr_str = ""
    gen_ids: list[int] = []
    functions_names = [fn.name for fn in functions]
    for _ in range(30):
        full_input_ids = input_ids + gen_ids
        logits = model.get_logits_from_input_ids(full_input_ids)
        logits_numpy = np.array(logits)
        valid_tokens_ids = get_valid_next_tokens(
            curr_str, functions_names, token_id
        )
        masked_logits = mask_logits(logits_numpy, valid_tokens_ids)
        next_token_id = int(np.argmax(masked_logits))
        next_token = id_token[next_token_id]
        gen_ids.append(next_token_id)

        curr_str += next_token
        if curr_str in functions_names:
            selected_function = next(
                fn for fn in functions if fn.name == curr_str)
            input_ids.extend(gen_ids)
            input_ids.append(token_id['"'])
            input_ids.append(token_id[','])
            input_ids.append(token_id['Ġ'])
            input_ids.append(token_id['"'])
            input_ids.append(token_id['parameters'])
            input_ids.append(token_id['":'])
            input_ids.append(token_id['Ġ'])
            input_ids.append(token_id['{'])
            params = list(selected_function.parameters.items())
            parameters: dict[str, float | str | int] = {}
            for i, (name, value) in enumerate(params):
                input_ids.append(token_id['"'])
                for chr_id in model.encode(name)[0].tolist():
                    input_ids.append(chr_id)
                input_ids.append(token_id['"'])
                input_ids.append(token_id[':'])
                input_ids.append(token_id['Ġ'])
                if value.type == "number" or value.type == "integer":
                    has_negative = bool(re.search(r"-\d+", prompt.prompt))
                    if has_negative:
                        numb = re.findall(
                            r"-?\d+(?:\.\d+)?", prompt.prompt)
                        if value.type == "number":
                            parameters[name] = float(numb[i])
                        else:
                            parameters[name] = int(numb[i])
                    else:
                        val, ids = generate_number_value(
                            input_ids, model, token_id, id_token)
                        if value.type == "number":
                            parameters[name] = val
                        else:
                            parameters[name] = int(val)
                        input_ids += ids

                    if i < len(params) - 1:
                        input_ids.append(token_id[','])
                        input_ids.append(token_id['Ġ'])
                    else:
                        input_ids.append(token_id['}'])
                        input_ids.append(token_id['}'])
                elif value.type == "string":
                    input_ids.append(token_id['"'])
                    (str_val, ids) = generate_string_value(
                        input_ids, model, id_token)
                    parameters[name] = str_val
                    input_ids += ids

            return ResultValidation(
                prompt=prompt.prompt,
                name=curr_str,
                parameters=parameters
            )

    return ResultValidation(
        prompt=prompt.prompt,
        name="invalid",
        parameters={"error": "no matching function found"}
    )
