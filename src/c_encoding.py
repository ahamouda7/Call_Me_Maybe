import json


def load_vocabulary(vocab_path):
    try:
        with open(vocab_path, "r") as file:
            tokens = json.load(file)
        print(tokens)
    except Exception as e:
        print(e)
