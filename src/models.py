from pydantic import BaseModel
from typing import Dict


class ParameterDef(BaseModel):
    """One parameter of a function — just its type."""
    type: str  # "number", "string", "boolean"


class FunctionDef(BaseModel):
    """One entry from functions_definition.json."""
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ParameterDef


class Prompt(BaseModel):
    """One entry from function_calling_tests.json."""
    prompt: str
