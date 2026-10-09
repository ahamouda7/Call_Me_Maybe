
import json
import keyword
import logging
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError


logging.basicConfig(
    filename="src/log.log",
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    filemode="w"
)


class Type(str, Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    INTEGER = "integer"


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, strict=True)

    @field_validator("prompt")
    @classmethod
    def check_prompt(cls, value):
        if not value.strip():
            raise ValueError("Empty prompt")
        return value


class Parameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Type


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter

    @field_validator("name")
    @classmethod
    def check_name(cls, value):
        if not value.isidentifier() or keyword.iskeyword(value):
            raise ValueError("Invalid function name")
        return value

    @field_validator("description")
    @classmethod
    def check_description(cls, value):
        if not value.strip():
            raise ValueError("Empty description")
        return value

    @field_validator("parameters")
    @classmethod
    def check_parameters(cls, value):
        for name in value:
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(f"Invalid parameter name: {name}")
        return value


class Parse_data:
    def __init__(self, input_data, functions_data):
        self.errors_input = []
        self.errors_functions = []

        self.parsed_input_data = self.validate(
            input_data, Prompt, self.errors_input
        )
        self.parsed_function_data = self.validate(
            functions_data, FunctionDefinition, self.errors_functions
        )

    def validate(self, data, model, errors):
        valid = []
        names = set()

        if not isinstance(data, list) or not data:
            errors.append("Expected a non-empty list")
            return valid

        for i, element in enumerate(data, start=1):
            try:
                item = model.model_validate(element)

                if model == FunctionDefinition:
                    if item.name in names:
                        raise ValueError("Duplicate function name")
                    names.add(item.name)

                valid.append(item)

            except (ValidationError, ValueError) as e:
                message = (
                    e.errors()[0]["msg"]
                    if isinstance(e, ValidationError)
                    else str(e)
                )
                label = (
                    "Function" if model == FunctionDefinition else "Prompt"
                )
                error = f"{label} {i}: {message}"
                errors.append(error)
                logging.error(error)

        return valid

    def is_valid(self):
        return not self.errors_input and not self.errors_functions


if __name__ == "__main__":
    try:
        with open("data/input/function_calling_tests.json") as f:
            input_data = json.load(f)

        with open("data/input/functions_definition.json") as f:
            functions_data = json.load(f)

        parser = Parse_data(input_data, functions_data)

        print("Valid:", parser.is_valid())
        print("Input errors:", parser.errors_input)
        print("Function errors:", parser.errors_functions)

    except (OSError, json.JSONDecodeError) as e:
        print("Error:", e)
        logging.error(str(e))
