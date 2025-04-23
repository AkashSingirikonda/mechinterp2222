from collections import defaultdict
from typing import Tuple, Any, List, Dict
from prompt_interface import PromptCase

def collect_failure_modes(
    cases: List[PromptCase],
    pivot_index: int,
    success_key: str = "substring_match"
) -> List[Tuple[PromptCase, PromptCase]]:
    """
    For each set of PromptCase in `cases` that differ *only* at inputs[pivot_index],
    collect the pairs (clean, corrupted) where one succeeds and the other fails.

    Returns a list of (clean_case, corrupted_case) tuples, aligned so that
    clean_case.inputs[pivot_index] != corrupted_case.inputs[pivot_index].
    """
    # 1) Bucket by all inputs except the pivot_index
    buckets: Dict[Tuple[Any, ...], List[PromptCase]] = defaultdict(list)
    for case in cases:
        # make a key of inputs without the pivot
        key = tuple(
            v for i, v in enumerate(case.inputs) if i != pivot_index
        )
        buckets[key].append(case)

    pairs = []
    # 2) For each bucket, look for two cases that differ at pivot and one success/one failure
    for key, bucket in buckets.items():
        # further group by the pivot value
        by_pivot = defaultdict(list)
        for case in bucket:
            by_pivot[case.inputs[pivot_index]].append(case)

        # only interested in groups with at least two different pivot values
        pivot_values = list(by_pivot.keys())
        if len(pivot_values) < 2:
            continue

        # try every pair of pivot values
        for v1, v2 in zip(pivot_values, pivot_values[1:]):
            for c1 in by_pivot[v1]:
                for c2 in by_pivot[v2]:
                    # ensure they only differ at pivot_index
                    if all(
                        (i == pivot_index) or (c1.inputs[i] == c2.inputs[i])
                        for i in range(len(c1.inputs))
                    ):
                        # pick which is “clean” vs. “corrupted” by success_key
                        ok1 = c1.evaluation_result.get(success_key, False)
                        ok2 = c2.evaluation_result.get(success_key, False)
                        if ok1 and not ok2:
                            pairs.append((c1, c2))
                        elif ok2 and not ok1:
                            pairs.append((c2, c1))
    return pairs