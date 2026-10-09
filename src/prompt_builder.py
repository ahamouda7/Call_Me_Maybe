from typing import List
from src.models import FunctionDef


def build_prompt(
    functions: List[FunctionDef],
    user_request: str
) -> str:
    """Build the full prompt string fed to the LLM."""

    # part 1 — system instructions (constant)
    system = (
        "You are a function-calling assistant.\n"
        "Given a user request and available functions, "
        "select the correct function and extract the arguments.\n\n"
    )

    # part 2 — function catalogue (built from functions_definition.json)
    catalogue = "Available functions:\n\n"
    for fn in functions:
        params = ", ".join(
            f"{name} ({p.type})"
            for name, p in fn.parameters.items()
        )
        catalogue += f"- {fn.name}: {fn.description}\n"
        catalogue += f"  parameters: {params}\n\n"

    # part 3 — user request
    request = f'User request: "{user_request}"\n'

    return system + catalogue + request
