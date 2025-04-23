from utils.prompt_interface import Prompt

# === Individual Prompt Definitions ===

PRINT_PROMPT = Prompt(
    name="print",
    prompt_fn=lambda inp: f"Print out this list of numbers: {inp[0]}.",
    transform_fn=lambda inp: inp[0],
    random_input_fn=None,
)

APPEND_PROMPT = Prompt(
    name="append",
    prompt_fn=lambda inp: f"Append {inp[1]} to the end of this list {inp[0]}",
    transform_fn=lambda inp: inp[0] + [inp[1]],
    random_input_fn=None,
)

ADD_ALL_PROMPT = Prompt(
    name="add_all",
    prompt_fn=lambda inp: f"Add {inp[1]} to every element in this list: {inp[0]}",
    transform_fn=lambda inp: [x + inp[1] for x in inp[0]],
    random_input_fn=None,
)

INSERT_MIDDLE_PROMPT = Prompt(
    name="insert_middle",
    prompt_fn=lambda inp: f"Insert {inp[1]} between the third and fourth element in this list: {inp[0]}",
    transform_fn=lambda inp: inp[0][:3] + [inp[1]] + inp[0][3:],
    random_input_fn=None,
)

SWAP_INDICES_PROMPT = Prompt(
    name="swap_indices",
    prompt_fn=lambda inp: (
        f"Given a {inp[3]} indexed list, {inp[0]}, "
        f"what would the list be if you swapped the elements at position {inp[1]} and {inp[2]}?"
    ),
    transform_fn=lambda inp: (
        lambda x: (x := x.copy(), x.__setitem__(inp[1], x[inp[2]]),
                   x.__setitem__(inp[2], inp[0][inp[1]]), x)[-1]
    )(inp[0]),
    random_input_fn=None,
)


LIST_PROMPTS = [
    PRINT_PROMPT,
    APPEND_PROMPT,
    ADD_ALL_PROMPT,
    INSERT_MIDDLE_PROMPT,
    SWAP_INDICES_PROMPT,
]

PROMPT_REGISTRY = {
    "list": {p.name: p for p in LIST_PROMPTS}
}

ALL_PROMPTS = {
    f"list.{p.name}": p for p in LIST_PROMPTS
}
