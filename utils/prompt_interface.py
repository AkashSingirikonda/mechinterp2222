from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from transformer_lens import HookedTransformer
from tqdm import tqdm 

@dataclass
class PromptCase:
    task_id: str
    inputs: List[Any]
    prompt_fn: Callable[[List[Any]], str]
    ground_truth: Any
    metadata: Dict = field(default_factory=dict)
    wrap_fn: Optional[Callable[[str, List[Any]], str]] = None
    generated_output: str = None 
    evaluation_result: Dict = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        core = self.prompt_fn(self.inputs)
        return self.wrap_fn(core, self.inputs) if self.wrap_fn else core

    def run_model(self, model: HookedTransformer, max_tokens: int = 30) -> Dict:
        tokens = model.to_tokens(self.prompt)
        generated = model.generate(
            tokens,
            max_new_tokens=max_tokens,
            temperature=0.0,
            top_k=0,
        )
        decoded = model.tokenizer.decode(generated[0]).strip()
        self.generated_output = decoded

        self.evaluation_result = {
            "exact_match": str(self.ground_truth).strip() == decoded,
            "substring_match": str(self.ground_truth).strip() in decoded,
            "output": decoded,
        }
        return self.evaluation_result
    
    def copy(self, **overrides) -> "PromptCase":
        return PromptCase(
            task_id=overrides.get("task_id", self.task_id),
            inputs=overrides.get("inputs", self.inputs.copy()),
            prompt_fn=overrides.get("prompt_fn", self.prompt_fn),
            ground_truth=overrides.get("ground_truth", self.ground_truth),
            metadata={**self.metadata, **overrides.get("metadata", {})},
            wrap_fn=overrides.get("wrap_fn", self.wrap_fn),
            generated_output=overrides.get("generated_output", self.generated_output),
            evaluation_result=overrides.get("evaluation_result", self.evaluation_result.copy()),
        )

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
        self.cases = []

        prompt_names = list(self.prompts.keys()) if prompt_name == "all" else [prompt_name]
        wrap_names = list(self.wraps.keys()) if wrap_name == "all" else [wrap_name]

        for p_name in prompt_names:
            for w_name in wrap_names:
                prompt = self.prompts[p_name]
                wrap_fn = self.wraps[w_name]

                cases = prompt.create_cases(n)
                for case in cases:
                    case.wrap_fn = wrap_fn
                    case.metadata["wrap_name"] = w_name
                self.cases.extend(cases)

        return self.cases

    def generate_all(self, n: int) -> List[PromptCase]:
        return self.generate(prompt_name="all", wrap_name="all", n=n)
    
    def evaluate_all(self, model: HookedTransformer, max_tokens: int = 30, eval_type = "substring_match") -> List[Dict]:
        results = []

        for case in tqdm(self.cases, desc="Evaluating prompt cases"):
            result = case.run_model(model, max_tokens=max_tokens)
            results.append({
                "task_id": case.task_id,
                "prompt": case.prompt,
                "ground_truth": case.ground_truth,
                **result
            })

        results.sort(key=lambda r: not r[eval_type])

        total = len(results)
        successes = sum(1 for r in results if r[eval_type])
        print(f"\nEvaluation Summary: {successes}/{total} correct ({(successes/total)*100:.1f}%)")
        
        for r in results:
            status = "Success" if r[eval_type] else "❌ Failure"
            print(f"\n{status}: {r['task_id']}")
            print("Prompt:\n", r["prompt"])
            print("Expected:", r["ground_truth"])
            print("Got     :", r["output"])

        return results

    
    def random_inputs(self, **kwargs) -> List[Any]:
        raise NotImplementedError("Implement random input generation for your family.")

    def test_behavior(self, case: PromptCase, model_output: str) -> Dict:
        raise NotImplementedError()

    def analyze_tokens(self, case: PromptCase, tokenizer) -> Dict:
        raise NotImplementedError()
