#!/usr/bin/env python3
import sys
import json
import os
import tarfile

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.tar.gz")
_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    # Default configuration-based "model" with sensible thresholds.
    model = {
        "small_max_chars": 80,
        "medium_max_chars": 300,
    }

    if os.path.exists(MODEL_PATH):
        try:
            data = None
            if MODEL_PATH.endswith(".json"):
                with open(MODEL_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif MODEL_PATH.endswith((".tar.gz", ".tgz")):
                with tarfile.open(MODEL_PATH, "r:gz") as tar:
                    member = tar.next()
                    while member and not member.name.endswith(".json"):
                        member = tar.next()
                    if member is not None:
                        extracted = tar.extractfile(member)
                        if extracted is not None:
                            data = json.load(extracted)

            if isinstance(data, dict):
                small_val = data.get("small_max_chars")
                medium_val = data.get("medium_max_chars")
                if isinstance(small_val, int):
                    model["small_max_chars"] = small_val
                if isinstance(medium_val, int):
                    model["medium_max_chars"] = medium_val
        except Exception:
            # On any problem reading the model, fall back to defaults.
            pass

    _model = model
    return _model

def classify(text: str) -> str:
    """
    Return one of: 'small-llm', 'medium-llm', 'large-llm'.
    Replace this with your real model inference.
    """

    # model = load_model()

    # small_max = model.get("small_max_chars", 80)
    # medium_max = model.get("medium_max_chars", 300)

    # length = len(text)
    # if length <= small_max:
    #     return "small-llm"
    # elif length <= medium_max:
    #     return "medium-llm"
    # else:
    #     return "large-llm"

    length = len(text)
    if length < 80:
        return "small-llm"
    elif length < 300:
        return "medium-llm"
    else:
        return "large-llm"

def main():
    # Read full prompt from stdin
    prompt = sys.stdin.read().strip()
    if not prompt:
        print(json.dumps({"error": "empty prompt"}))
        return

    size_class = classify(prompt)

    # You can also decide the exact model ID here if you want.
    # For now we only return the class; Node will map it to a Bedrock model.
    result = {
        "class": size_class
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()
