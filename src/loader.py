import json
from typing import List
from src.models import FunctionDef, Prompt


def load_functions(path: str) -> List[FunctionDef]:
    """Load and validate functions_definition.json."""
    try:
        with open(path) as f:
            data = json.load(f)
        return [FunctionDef(**item) for item in data]
    except FileNotFoundError:
        print(f"Error: file not found → {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON → {e}")
        return []


def load_prompts(path: str) -> List[Prompt]:
    """Load and validate function_calling_tests.json."""
    try:
        with open(path) as f:
            data = json.load(f)
        return [Prompt(**item) for item in data]
    except FileNotFoundError:
        print(f"Error: file not found → {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON → {e}")
        return []
