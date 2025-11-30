"""XLM-RoBERTa fine-tuning baseline using Hugging Face transformers."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Dict, List, Sequence, Tuple

from packaging import version

TransformersStack = Tuple[
    object,  # torch module
    type,  # Dataset class
    type,  # AutoModelForSequenceClassification
    type,  # AutoTokenizer
    type,  # Trainer
    type,  # TrainingArguments
]


def transformers_available() -> bool:
    """Return ``True`` if torch, datasets, and transformers are installed."""

    return all(find_spec(mod) is not None for mod in ("torch", "datasets", "transformers"))


def import_transformers_stack() -> TransformersStack:
    """Import and return the transformer dependencies needed for training."""

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    torch = import_module("torch")
    datasets = import_module("datasets")
    transformers = import_module("transformers")

    if not hasattr(transformers.utils, "is_torch_greater_or_equal"):
        def _is_torch_greater_or_equal(min_version: str) -> bool:
            if torch is None:
                return False
            return version.parse(torch.__version__) >= version.parse(min_version)

        transformers.utils.is_torch_greater_or_equal = _is_torch_greater_or_equal

    from transformers import (  # type: ignore[attr-defined]
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    return (
        torch,
        datasets.Dataset,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )


@dataclass
class XLMRClassifier:
    """Fine-tunes an XLM-RoBERTa sequence classification model."""

    model_name: str = "xlm-roberta-base"
    output_dir: str = "./xlmr_language_id"
    num_train_epochs: float = 1.0
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    seed: int = 13

    def __post_init__(self) -> None:
        (
            self._torch,
            self._dataset_cls,
            self._auto_model_cls,
            self._tokenizer_cls,
            self._trainer_cls,
            self._training_args_cls,
        ) = import_transformers_stack()
        self.tokenizer = self._tokenizer_cls.from_pretrained(self.model_name)
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.model = None
        self.trainer = None

    def _encode_dataset(self, dataset: object) -> object:
        def tokenize_function(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )

        return dataset.map(tokenize_function, batched=True)

    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[str],
        eval_texts: Sequence[str],
        eval_labels: Sequence[str],
    ) -> None:
        unique_labels = sorted(set(train_labels) | set(eval_labels))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        train_dataset = self._dataset_cls.from_dict(
            {"text": list(train_texts), "label": [self.label2id[label] for label in train_labels]}
        )
        eval_dataset = self._dataset_cls.from_dict(
            {"text": list(eval_texts), "label": [self.label2id[label] for label in eval_labels]}
        )

        encoded_train = self._encode_dataset(train_dataset)
        encoded_eval = self._encode_dataset(eval_dataset)

        self.model = self._auto_model_cls.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            label2id=self.label2id,
            id2label=self.id2label,
        )

        args_signature = inspect.signature(self._training_args_cls.__init__).parameters
        training_kwargs = dict(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            seed=self.seed,
        )

        optional_args = {
            "evaluation_strategy": "epoch",
            "logging_strategy": "epoch",
            "save_strategy": "no",
            "load_best_model_at_end": False,
        }
        for name, value in optional_args.items():
            if name in args_signature:
                training_kwargs[name] = value

        if "evaluation_strategy" not in training_kwargs and "evaluate_during_training" in args_signature:
            training_kwargs["evaluate_during_training"] = True

        training_args = self._training_args_cls(**training_kwargs)

        trainer = self._trainer_cls(
            model=self.model,
            args=training_args,
            train_dataset=encoded_train,
            eval_dataset=encoded_eval,
        )
        trainer.train()
        self.trainer = trainer

    def predict(self, texts: Sequence[str]) -> List[str]:
        if self.model is None or self.trainer is None:
            raise RuntimeError("The model has not been trained yet.")
        dataset = self._dataset_cls.from_dict({"text": list(texts)})
        encoded_dataset = self._encode_dataset(dataset)
        predictions = self.trainer.predict(encoded_dataset).predictions
        predicted_ids = predictions.argmax(axis=-1)
        return [self.id2label[idx] for idx in predicted_ids]
