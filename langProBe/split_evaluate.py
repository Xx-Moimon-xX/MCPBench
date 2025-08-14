"""
Utilities to decouple generation and evaluation for DSPy programs.

This module provides a small, self-contained class `SplitEvaluate` that mirrors
`dspy.Evaluate`'s parallel execution for generation, while allowing you to score
later (or with different metrics) without re-generating.

Usage:

    split_eval = SplitEvaluate(devset=my_devset, metric=my_metric, num_threads=8)
    predictions = split_eval.generate(program=my_program, display_progress=True)
    score, results = split_eval.score(predictions)  # results: [(example, pred, score), ...]

This does not modify DSPy; it simply reuses the same parallel executor to run
your program over the dataset and then score separately.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import dspy, pdb


class SplitEvaluate:
    """
    Split generation and evaluation for a DSPy program.

    - generate(): runs the program over the devset in parallel and returns predictions
    - score(): computes metric over (example, prediction) pairs and returns (avg%, detailed list)
    """

    def __init__(
        self,
        *,
        devset: List["dspy.Example"],
        metric: Optional[Callable] = None,
        num_threads: Optional[int] = None,
        max_errors: Optional[int] = None,
        provide_traceback: Optional[bool] = None,
    ) -> None:
        self.devset = devset
        self.metric = metric
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.provide_traceback = provide_traceback

    def _make_executor(self, display_progress: bool):
        # Lazy import to match DSPy's structure
        from dspy.utils.parallelizer import ParallelExecutor

        return ParallelExecutor(
            num_threads=self.num_threads,
            disable_progress_bar=not display_progress,
            max_errors=self.max_errors if self.max_errors is not None else dspy.settings.max_errors,
            provide_traceback=self.provide_traceback,
            compare_results=False,
        )

    def generate(
        self,
        *,
        program: "dspy.Module",
        display_progress: bool = False,
    ) -> List["dspy.Prediction"]:
        """
        Run the program on each example in `devset` in parallel and return predictions only.
        """

        print(f"In split_evaluate.generate")
        # pdb.set_trace(header="inside split_eval.generate")
        breakpoint()
        executor = self._make_executor(display_progress=display_progress)

        def process_item(example: "dspy.Example") -> "dspy.Prediction":
            return program(**example.inputs())

        # pdb.set_trace(header="running through the examples")
        breakpoint()

        predictions = executor.execute(process_item, self.devset)
        # Replace failed items (None) with empty predictions to keep alignment
        predictions = [p if p is not None else dspy.Prediction() for p in predictions]

        # pdb.set_trace(header="returning predictions from split_eval generate()")
        breakpoint()

        return predictions

    def score(
        self,
        predictions: List["dspy.Prediction"],
        *,
        metric: Optional[Callable] = None,
    ) -> Tuple[float, List[tuple]]:
        """
        Compute metric(example, prediction) per example.

        Returns:
            (average_percent_score, detailed_results)
        where:
            - average_percent_score is a float like 67.3
            - detailed_results is a list of (example, prediction, score_float)
        """
        if metric is None:
            metric = self.metric
        assert metric is not None, "metric must be provided either at init or call-time"

        assert len(self.devset) == len(predictions), "devset and predictions must align in length"
        detailed_results = []
        total = 0.0
        for example, prediction in zip(self.devset, predictions):
            score_val = metric(example, prediction)
            # Normalize to float
            score_float = float(score_val)
            total += score_float
            detailed_results.append((example, prediction, score_float))

        avg_percent = round(100.0 * total / len(self.devset), 2) if self.devset else 0.0
        return avg_percent, detailed_results


