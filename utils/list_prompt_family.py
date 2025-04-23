from utils.prompt_interface import PromptFamily, PromptCase, Prompt
from transformer_lens import HookedTransformer
from typing import List, Dict
from prompt_registry import PROMPT_REGISTRY
from wrap_registry import WRAP_REGISTRY
import numpy as np


class ListPromptFamily(PromptFamily):
    def __init__(
        self,
        min_val: int = 0,
        max_val: int = 10,
        list_size: int = 5,
        append_min: int = 0,
        append_max: int = 20,
        index_range: int = None,
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.list_size = list_size
        self.append_min = append_min
        self.append_max = append_max
        self.index_range = list_size if index_range is None else index_range

        def rand_list():
            return np.random.randint(self.min_val, self.max_val, size=self.list_size).tolist()

        # Clone and inject random_input_fn into each Prompt
        prompt_templates = PROMPT_REGISTRY["list"]
        prompts = []

        for name, base_prompt in prompt_templates.items():
            # Shallow copy of Prompt with custom random_input_fn
            prompt = Prompt(
                name=base_prompt.name,
                prompt_fn=base_prompt.prompt_fn,
                transform_fn=base_prompt.transform_fn,
                random_input_fn=self._make_random_input_fn(name, rand_list)
            )
            prompts.append(prompt)

        super().__init__(name="list", prompts=prompts, wraps=WRAP_REGISTRY)

    def _make_random_input_fn(self, name: str, rand_list_fn) -> callable:
        if name == "print":
            return lambda: [rand_list_fn()]

        elif name in {"append", "add_all", "insert_middle"}:
            return lambda: [
                rand_list_fn(),
                np.random.randint(self.append_min, self.append_max),
            ]

        elif name == "swap_indices":
            def make_swap_inputs():
                lst = rand_list_fn()
                indexing = np.random.choice(["zero", "one"])

                if indexing == "zero":
                    i1 = np.random.randint(0, len(lst))
                    i2 = np.random.randint(0, len(lst))
                else:  # one-indexed (1 to len inclusive)
                    i1 = np.random.randint(1, len(lst) + 1)
                    i2 = np.random.randint(1, len(lst) + 1)

                return [lst, i1, i2, indexing]

            return make_swap_inputs

        else:
            raise ValueError(f"Unrecognized prompt name: {name}")


    def analyze_tokens(self, case: PromptCase, model: HookedTransformer) -> Dict:
        tokens = model.to_tokens(case.prompt)[0]
        token_strs = model.to_str_tokens(tokens)

        number_token_spans = {}
        split_tokens = []
        numbers = case.metadata.get("inputs", [])[0] if case.metadata.get("inputs") else []

        for num in numbers:
            num_str = str(num)
            tokenized = model.to_tokens(num_str)[0]
            number_token_spans[num] = len(tokenized)
            if len(tokenized) > 1:
                split_tokens.append(num)

        return {
            "total_tokens": len(tokens),
            "token_ids": tokens.tolist(),
            "token_strs": token_strs,
            "number_token_spans": number_token_spans,
            "split_tokens": split_tokens,
            "num_splits": len(split_tokens)
        }

    def test_behavior(self, case: PromptCase, model_output: str) -> Dict:
        return {
            "exact_match": str(case.ground_truth).strip() == model_output.strip(),
            "substring": str(case.ground_truth).strip() in model_output.strip()
        }
