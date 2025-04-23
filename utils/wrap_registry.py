# wrap_registry.py

from typing import List, Callable

# === Individual Wrap Functions ===

def plain_wrap(prompt: str, _: List) -> str:
    return prompt

def list_only_wrap(prompt: str, _: List) -> str:
    return f"{prompt}\nOnly output a list, no other information.\nList: ["

def python_interpreter_wrap(prompt: str, inputs: List) -> str:
    return (
        f"Pretend you are a Python interpreter.\n"
        f"TASK: {prompt}\n"
        f"INPUT: {inputs}\n"
        f"OUTPUT:"
    )

def system_message_wrap(prompt: str, _: List) -> str:
    return f"<|system|> You are a helpful assistant.\n<|user|> {prompt}\n<|assistant|>"

# === Named Functions for Direct Use ===

PLAIN_WRAP = plain_wrap
LIST_ONLY_WRAP = list_only_wrap
PYTHON_INTERPRETER_WRAP = python_interpreter_wrap
SYSTEM_MESSAGE_WRAP = system_message_wrap

# === Grouped and Registered ===

ALL_WRAP_FNS: List[Callable[[str, List], str]] = [
    PLAIN_WRAP,
    LIST_ONLY_WRAP,
    PYTHON_INTERPRETER_WRAP,
    SYSTEM_MESSAGE_WRAP
]

WRAP_REGISTRY = {
    "plain": PLAIN_WRAP,
    "list": LIST_ONLY_WRAP,
    "interpreter": PYTHON_INTERPRETER_WRAP,
    "system": SYSTEM_MESSAGE_WRAP,
}
