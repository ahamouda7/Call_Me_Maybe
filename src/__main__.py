import os
import json
import argparse
from generator import generate_function_call
from c_encoding import load_vocabulary
from llm_sdk.llm_sdk import Small_LLM_Model  # type: ignore
from loader import validate_functions, validate_prompts


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json"
    )

    args = parser.parse_args()

    try:
        model = Small_LLM_Model()

        vocab_path = model.get_path_to_vocab_file()
        token_id, id_token = load_vocabulary(vocab_path)

        validate_prom = validate_prompts(args.input)
        validate_func = validate_functions(args.functions_definition)
        if validate_prom is None:
            print("Cannot continue: input file is invalid.")
            return
        if validate_func is None:
            print("Cannot continue: functions definition file is invalid.")
            return

        results = []
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        for prompt in validate_prom:
            data = generate_function_call(
                prompt, validate_func, model, token_id, id_token
            )
            results.append(data.model_dump())
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
