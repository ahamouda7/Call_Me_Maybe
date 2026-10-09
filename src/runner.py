from llm_sdk import Small_LLM_Model
from src.loader import load_functions, load_prompts
from src.prompt_builder import build_prompt


def run(
    functions_path: str,
    input_path: str,
    output_path: str
) -> None:
    """Main pipeline."""

    # stage 0 — load and validate inputs
    functions = load_functions(functions_path)
    prompts   = load_prompts(input_path)

    if not functions or not prompts:
        print("Error: could not load input files.")
        return

    # load the model once — expensive, do it outside the loop
    model = Small_LLM_Model()

    results = []

    for p in prompts:

        # stage 1+2 — build prompt string
        prompt_text = build_prompt(functions, p.prompt)

        # stage 3 — tokenize (BPE + vocab lookup)
        input_ids = model.encode(prompt_text)

        # stage 4 — embedding happens inside here automatically
        # stage 5 — constrained decoding ← coming next
        logits = model.get_logits_from_input_ids(input_ids)
