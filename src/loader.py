
"""Read and validate call_me_maybe input files."""

import sys
import json
import argparse
from typing import Dict, Literal

from pydantic import BaseModel, ConfigDict, ValidationError


JsonType = Literal[
    "string", "number", "integer", "boolean",
    "array", "object", "null"
]


class Prompts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str


class Parameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: JsonType


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Parameter


def print_error(e, location):
    error = e.errors()[0]
    msg = error["msg"]

    sys.exit(f"[ERROR] in {location}: {msg}")


def validate_prompt(data):
    prompts = []

    if not isinstance(data, list):
        sys.exit("[ERROR]: Prompts must be a list")

    for i, p in enumerate(data, 1):
        try:
            prompts.append(Prompts(**p))
        except ValidationError as e:
            print_error(e, f"prompt {i}")

    return prompts


def validate_functions(data):
    functions = []

    if not isinstance(data, list):
        print("[ERROR]: Functions must be a list")
        return []

    try:
        names = set()

        for f in data:
            func = FunctionDefinition(**f)

            if not func.name.strip():
                print("[ERROR]: Function name cannot be empty")
                return []

            if func.name in names:
                print(f"[ERROR]: Duplicate function '{func.name}'")
                return []

            names.add(func.name)
            functions.append(func)

    except ValidationError as e:
        print(f"[ERROR]: {e.errors()[0]['msg']}")
        return []

    return functions


def read_json(path):
    try:
        with open(path, "r") as file:
            return json.load(file)

    except FileNotFoundError:
        print(f"[ERROR]: File '{path}' not found")

    except json.JSONDecodeError as e:
        print(f"[ERROR]: Invalid JSON in '{path}': {e}")

    except OSError as e:  # FileNotFoundError + PermissionError + IsADirectoryError
        print(f"[ERROR]: Cannot read '{path}': {e}") 

    return None


if __name__ == "__main__":
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

    data = read_json(args.input)
    if data is not None:
        prompts = validate_prompt(data)
        print(prompts)

    data = read_json(args.functions_definition)
    if data is not None:
        functions = validate_functions(data)
        print(functions)
