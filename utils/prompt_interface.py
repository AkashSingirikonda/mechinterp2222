from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class PromptCase:
    task_id: str
    inputs: List[Any]
    prompt_fn: Callable[[List[Any]], str]
    ground_truth: Any
    metadata: Dict = field(default_factory=dict)
    wrap_fn: Optional[Callable[[str, List[Any]], str]] = None  # NEW

    @property
    def prompt(self) -> str:
        core = self.prompt_fn(self.inputs)
        return self.wrap_fn(core, self.inputs) if self.wrap_fn else core


@dataclass
class Prompt:
    name: str
    prompt_fn: Callable[[List[Any]], str]
    transform_fn: Optional[Callable[[List[Any]], Any]] = None
    random_input_fn: Optional[Callable[[], List[Any]]] = None
    case_counter: int = field(default=0, init=False)

    def create_case(
        self,
        inputs: Optional[List[Any]] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> PromptCase:
        inputs = inputs or self.random_inputs()
        ground_truth = self.transform_fn(inputs) if self.transform_fn else inputs
        task_id = task_id or f"{self.name}-{self.case_counter}"
        self.case_counter += 1

        return PromptCase(
            task_id=task_id,
            inputs=inputs,
            prompt_fn=self.prompt_fn,
            ground_truth=ground_truth,
            metadata={
                "prompt_name": self.name,
                "inputs": inputs,
                **(metadata or {}),
            },
            wrap_fn=None,
        )

    def create_cases(self, n: int) -> List[PromptCase]:
        return [self.create_case() for _ in range(n)]

    def random_inputs(self) -> List[Any]:
        if not self.random_input_fn:
            raise NotImplementedError(f"No random_input_fn provided for prompt '{self.name}'")
        return self.random_input_fn()

class PromptFamily:
    def __init__(
        self,
        name: str,
        prompts: List[Prompt],
        wraps: Dict[str, Callable[[str, List], str]],
    ):
        self._name = name
        self.prompts: Dict[str, Prompt] = {p.name: p for p in prompts}
        self.wraps: Dict[str, Callable[[str, List], str]] = wraps
        self.cases: List[PromptCase] = []

    def name(self) -> str:
        return self._name

    def generate(self, prompt_name: str, wrap_name: str, n: int) -> List[PromptCase]:
        prompt = self.prompts[prompt_name]
        wrap_fn = self.wraps[wrap_name]

        cases = prompt.create_cases(n)
        for case in cases:
            case.wrap_fn = wrap_fn
            case.metadata["wrap_name"] = wrap_name
        self.cases.extend(cases)
        return cases

    def generate_all(self, n: int) -> List[PromptCase]:
        self.cases = []
        for prompt_name in self.prompts:
            for wrap_name in self.wraps:
                self.generate(prompt_name, wrap_name, n)
        return self.cases

    def random_inputs(self, **kwargs) -> List[Any]:
        raise NotImplementedError("Implement random input generation for your family.")

    def test_behavior(self, case: PromptCase, model_output: str) -> Dict:
        raise NotImplementedError()

    def analyze_tokens(self, case: PromptCase, tokenizer) -> Dict:
        raise NotImplementedError()
