"""XLM-RoBERTa fine-tuning baseline."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = None

try:  # pragma: no cover - heavy dependency initialisation
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    import torch
    from datasets import Dataset
    import transformers

    if not hasattr(transformers.utils, "is_torch_greater_or_equal"):
        try:
            from packaging import version
        except Exception:  # pragma: no cover - packaging is part of std envs
            version = None

        def _is_torch_greater_or_equal(min_version: str) -> bool:
            if torch is None:
                return False
            if version is None:
                def _parse(ver: str) -> tuple[int, ...]:
                    return tuple(int(part) for part in ver.split(".") if part.isdigit())

                return _parse(torch.__version__) >= _parse(min_version)

            return version.parse(torch.__version__) >= version.parse(min_version)

        transformers.utils.is_torch_greater_or_equal = _is_torch_greater_or_equal

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
except Exception as exc:  # pragma: no cover - optional dependency
    torch = None
    Dataset = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    Trainer = None
    TrainingArguments = None
    TRANSFORMERS_IMPORT_ERROR = exc


class XLMRClassifier:
    """Fine-tunes an XLM-RoBERTa sequence classification model."""

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        output_dir: Path = Path("./xlmr_language_id"),
        num_train_epochs: float = 1.0,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        seed: int = 13,
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "The transformers baseline requires PyTorch and transformers to be installed."
            )
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.model: Optional[AutoModelForSequenceClassification] = None

    def _encode_dataset(self, dataset: Dataset) -> Dataset:
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

        train_dataset = Dataset.from_dict(
            {"text": list(train_texts), "label": [self.label2id[label] for label in train_labels]}
        )
        eval_dataset = Dataset.from_dict(
            {"text": list(eval_texts), "label": [self.label2id[label] for label in eval_labels]}
        )

        encoded_train = self._encode_dataset(train_dataset)
        encoded_eval = self._encode_dataset(eval_dataset)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            label2id=self.label2id,
            id2label=self.id2label,
        )

        args_signature = inspect.signature(TrainingArguments.__init__).parameters
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

        training_args = TrainingArguments(**training_kwargs)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded_train,
            eval_dataset=encoded_eval,
        )
        trainer.train()
        self.trainer = trainer

    def predict(self, texts: Sequence[str]) -> List[str]:
        if self.model is None:
            raise RuntimeError("The model has not been trained yet.")
        dataset = Dataset.from_dict({"text": list(texts)})
        encoded_dataset = self._encode_dataset(dataset)
        predictions = self.trainer.predict(encoded_dataset).predictions
        predicted_ids = predictions.argmax(axis=-1)
        return [self.id2label[idx] for idx in predicted_ids]
