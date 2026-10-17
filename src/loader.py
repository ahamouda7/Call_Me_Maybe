
"""Read and validate call_me_maybe input files."""

import sys
import json
from typing import Dict, Literal

from pydantic import BaseModel, Field, ConfigDict, ValidationError


JsonType = Literal["string", "number", "integer", "boolean"]


class Prompts(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str = Field(min_length=1)


class Parameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: JsonType


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, Parameter]
    returns: Parameter


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError:
        print(f"[ERROR]: File '{path}' not found")

    except json.JSONDecodeError as e:
        print(f"[ERROR]: Invalid JSON in '{path}': {e}")

    except OSError as e:
        # FileNotFoundError + PermissionError + IsADirectoryError
        print(f"[ERROR]: Cannot read '{path}': {e}")

    return None


def print_error(e, location):
    error = e.errors()[0]
    msg = error["msg"]

    sys.exit(f"[ERROR] in {location}: {msg}")


def validate_prompts(prompts_path):
    prompts = []
    data = read_json(prompts_path)

    if not isinstance(data, list):
        sys.exit("[ERROR]: Prompts must be a list")

    for i, p in enumerate(data, 1):
        try:
            prompts.append(Prompts(**p))
        except ValidationError as e:
            print_error(e, f"prompt {i}")

    return prompts


def validate_functions(functions_defintion_path):
    functions = []
    data = read_json(functions_defintion_path)

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
