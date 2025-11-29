"""Language identification baselines package."""

from .cli import parse_args
from .data import SentenceExample, iter_conllu_sentences, load_multilingual_dataset
from .evaluation import BaselineResult, evaluate_logistic_regression, evaluate_rule_based, evaluate_xlmr
from .reporting import compare_models, print_results

__all__ = [
    "BaselineResult",
    "compare_models",
    "evaluate_logistic_regression",
    "evaluate_rule_based",
    "evaluate_xlmr",
    "iter_conllu_sentences",
    "load_multilingual_dataset",
    "parse_args",
    "print_results",
    "SentenceExample",
]
