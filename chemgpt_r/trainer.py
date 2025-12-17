"""Custom Trainer for ChemGPT-R model.

This module provides a custom Trainer class that extends HuggingFace's
Trainer to support both regression (MSE) and DAPT (causal LM) modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn


@dataclass
class TrainerOutput:
    """Output from trainer methods.

    Args:
        loss: Training/evaluation loss.
        predictions: Model predictions (for regression).
        metrics: Additional metrics dictionary.
    """

    loss: Optional[float] = None
    predictions: Optional[np.ndarray] = None
    metrics: Optional[Dict[str, float]] = None


class ChemGPTRTrainer:
    """Custom trainer for ChemGPT-R models.

    Supports both regression (MSE loss) and DAPT (causal LM loss) training modes.
    Wraps HuggingFace Trainer for easy integration.

    Args:
        model: ChemGPTR model instance.
        args: Training arguments (HuggingFace TrainingArguments).
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset (optional).
        data_collator: Data collator for batching.
        compute_metrics: Function for computing evaluation metrics.
    """

    def __init__(
        self,
        model: Any,
        args: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        compute_metrics: Optional[Callable[[Any], Dict[str, float]]] = None,
    ):
        self.chemgptr_model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics

        # Create wrapper model for HuggingFace Trainer
        self._trainer: Optional[Any] = None

    def _create_trainer(self) -> Any:
        """Create the underlying HuggingFace Trainer."""
        from transformers import Trainer

        # Create a wrapper module that the Trainer can use
        wrapper = _ChemGPTRWrapper(self.chemgptr_model)

        trainer = Trainer(
            model=wrapper,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self._wrap_compute_metrics(),
        )

        return trainer

    def _wrap_compute_metrics(self) -> Optional[Callable]:
        """Wrap compute_metrics to handle model outputs."""
        if self.compute_metrics is None:
            return None

        def wrapped(eval_pred: Any) -> Dict[str, float]:
            predictions, labels = eval_pred
            return self.compute_metrics(eval_pred)

        return wrapped

    def train(self, **kwargs: Any) -> Any:
        """Train the model.

        Args:
            **kwargs: Additional arguments passed to Trainer.train().

        Returns:
            Training output from HuggingFace Trainer.
        """
        if self._trainer is None:
            self._trainer = self._create_trainer()
        return self._trainer.train(**kwargs)

    def evaluate(self, eval_dataset: Optional[Any] = None, **kwargs: Any) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            eval_dataset: Evaluation dataset (uses init dataset if not provided).
            **kwargs: Additional arguments passed to Trainer.evaluate().

        Returns:
            Evaluation metrics dictionary.
        """
        if self._trainer is None:
            self._trainer = self._create_trainer()
        return self._trainer.evaluate(eval_dataset=eval_dataset, **kwargs)

    def predict(self, test_dataset: Any, **kwargs: Any) -> Any:
        """Generate predictions on a dataset.

        Args:
            test_dataset: Test dataset.
            **kwargs: Additional arguments passed to Trainer.predict().

        Returns:
            Predictions output from HuggingFace Trainer.
        """
        if self._trainer is None:
            self._trainer = self._create_trainer()
        return self._trainer.predict(test_dataset, **kwargs)

    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Save the model.

        Args:
            output_dir: Directory to save to (uses args.output_dir if not provided).
        """
        save_dir = output_dir or self.args.output_dir
        self.chemgptr_model.save_pretrained(save_dir)

    @property
    def state(self) -> Any:
        """Get trainer state."""
        if self._trainer is not None:
            return self._trainer.state
        return None


class _ChemGPTRWrapper(nn.Module):
    """Wrapper module for HuggingFace Trainer compatibility.

    This wrapper adapts the ChemGPTR model to work with the standard
    HuggingFace Trainer interface, handling the custom inputs and outputs.
    """

    def __init__(self, chemgptr_model: Any):
        super().__init__()
        self.chemgptr_model = chemgptr_model

    @property
    def config(self) -> Any:
        """Return base model config for Trainer compatibility."""
        if self.chemgptr_model.base_model is not None:
            return self.chemgptr_model.base_model.config
        return None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass with Trainer-compatible output format.

        Returns a tuple where the first element is the loss (if labels provided)
        and subsequent elements depend on the mode.
        """
        outputs = self.chemgptr_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_features=graph_features,
            labels=labels,
            **kwargs,
        )

        if self.chemgptr_model.mode == "regression":
            loss = outputs.get("loss")
            predictions = outputs.get("predictions")

            if loss is not None:
                return (loss, predictions)
            return (predictions,)

        else:  # dapt mode
            loss = outputs.get("loss")
            logits = outputs.get("logits")

            if loss is not None:
                return (loss, logits)
            return (logits,)

    def parameters(self, recurse: bool = True):
        """Return all parameters from the wrapped model."""
        # Yield base model parameters
        if self.chemgptr_model.base_model is not None:
            yield from self.chemgptr_model.base_model.parameters(recurse=recurse)

        # Yield prefix projector parameters
        if self.chemgptr_model.prefix_projector is not None:
            yield from self.chemgptr_model.prefix_projector.parameters(recurse=recurse)

        # Yield regression head parameters
        if self.chemgptr_model.regression_head is not None:
            yield from self.chemgptr_model.regression_head.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Return all named parameters from the wrapped model."""
        # Yield base model parameters
        if self.chemgptr_model.base_model is not None:
            for name, param in self.chemgptr_model.base_model.named_parameters(
                prefix=f"{prefix}base_model." if prefix else "base_model.",
                recurse=recurse,
            ):
                yield name, param

        # Yield prefix projector parameters
        if self.chemgptr_model.prefix_projector is not None:
            for name, param in self.chemgptr_model.prefix_projector.named_parameters(
                prefix=f"{prefix}prefix_projector." if prefix else "prefix_projector.",
                recurse=recurse,
            ):
                yield name, param

        # Yield regression head parameters
        if self.chemgptr_model.regression_head is not None:
            for name, param in self.chemgptr_model.regression_head.named_parameters(
                prefix=f"{prefix}regression_head." if prefix else "regression_head.",
                recurse=recurse,
            ):
                yield name, param


def compute_regression_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute regression metrics.

    Args:
        eval_pred: Tuple of (predictions, labels).

    Returns:
        Dictionary of metrics (MSE, RMSE, MAE, R2).
    """
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()

    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - labels))

    # R-squared with epsilon to avoid division by zero/near-zero
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    eps = 1e-10
    r2 = 1 - (ss_res / (ss_tot + eps)) if ss_tot > eps else 0.0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def create_training_args(
    output_dir: str = "./output",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    logging_steps: int = 10,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    fp16: bool = False,
    bf16: bool = True,
    dataloader_num_workers: int = 4,
    seed: int = 42,
    **kwargs: Any,
) -> Any:
    """Create HuggingFace TrainingArguments.

    Args:
        output_dir: Directory for saving checkpoints.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Training batch size per device.
        per_device_eval_batch_size: Evaluation batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        warmup_ratio: Warmup ratio.
        logging_steps: Logging frequency.
        eval_strategy: Evaluation strategy ('no', 'steps', 'epoch').
            Maps to TrainingArguments.evaluation_strategy.
        save_strategy: Save strategy ('no', 'steps', 'epoch').
        fp16: Use FP16 mixed precision.
        bf16: Use BF16 mixed precision.
        dataloader_num_workers: Dataloader workers.
        seed: Random seed.
        **kwargs: Additional arguments.

    Returns:
        TrainingArguments instance.
    """
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        evaluation_strategy=eval_strategy,
        save_strategy=save_strategy,
        fp16=fp16,
        bf16=bf16,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
        remove_unused_columns=False,  # Important for custom inputs
        **kwargs,
    )
