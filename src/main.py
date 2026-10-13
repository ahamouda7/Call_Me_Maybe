import argparse
from c_encoding import load_vocabulary
from llm_sdk.llm_sdk import Small_LLM_Model
from loader import read_json, validate_prompt, validate_functions


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

    args = parser.parse_args()

    try:
        model = Small_LLM_Model()
        vocab_path = model.get_path_to_vocab_file()
        token, id = load_vocabulary(vocab_path)
        print(vocab_path)

        data = read_json(args.input)
        if data is not None:
            prompts = validate_prompt(data)
            print(prompts)

        data = read_json(args.functions_definition)
        if data is not None:
            functions = validate_functions(data)
            print(functions)
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()