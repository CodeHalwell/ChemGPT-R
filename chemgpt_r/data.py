"""Data pipeline for ChemGPT-R model training.

This module provides dataset classes and collators for integrating
graph features with text tokenization for the Graph-Prefix Decoder Regressor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .graph_features import GraphFeatureExtractor


@dataclass
class ChemGPTRSample:
    """A single sample for ChemGPT-R training.

    Args:
        smiles: SMILES string representation of the molecule.
        target: Target value for regression (optional for DAPT).
        text: Additional text context (optional).
    """

    smiles: str
    target: Optional[float] = None
    text: Optional[str] = None


class ChemGPTRDataset(Dataset):
    """Dataset for ChemGPT-R model.

    Args:
        samples: List of ChemGPTRSample or dicts with 'smiles', 'target', 'text' keys.
        tokenizer: HuggingFace tokenizer.
        graph_extractor: Graph feature extractor instance.
        max_length: Maximum sequence length for tokenization.
        mode: Training mode ('regression' or 'dapt').
    """

    def __init__(
        self,
        samples: Sequence[Union[ChemGPTRSample, Dict[str, Any]]],
        tokenizer: Any,
        graph_extractor: Optional[GraphFeatureExtractor] = None,
        max_length: int = 512,
        mode: str = "regression",
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.graph_extractor = graph_extractor or GraphFeatureExtractor()
        self.max_length = max_length
        self.mode = mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Handle both dataclass and dict formats
        if isinstance(sample, dict):
            smiles = sample["smiles"]
            target = sample.get("target")
            text = sample.get("text")
        else:
            smiles = sample.smiles
            target = sample.target
            text = sample.text

        # Extract graph features
        graph_features = self.graph_extractor(smiles)

        # Prepare text input
        if text:
            input_text = f"SMILES: {smiles}\n{text}"
        else:
            input_text = f"SMILES: {smiles}"

        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result: Dict[str, Any] = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "graph_features": torch.from_numpy(graph_features).float(),
        }

        if target is not None:
            result["labels"] = torch.tensor(target, dtype=torch.float32)

        # For DAPT mode, use input_ids as labels (shifted internally)
        if self.mode == "dapt":
            result["labels"] = encoding["input_ids"].squeeze(0).clone()

        return result


@dataclass
class ChemGPTRDataCollator:
    """Data collator for ChemGPT-R batches.

    Handles batching of tokenized inputs, graph features, and labels.

    Args:
        tokenizer: HuggingFace tokenizer (for padding token ID).
        mode: Training mode ('regression' or 'dapt').
    """

    tokenizer: Any
    mode: str = "regression"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of features.

        Args:
            features: List of feature dictionaries from dataset.

        Returns:
            Batched tensors dictionary.
        """
        batch: Dict[str, torch.Tensor] = {}

        # Stack input_ids
        input_ids = torch.stack([f["input_ids"] for f in features])
        batch["input_ids"] = input_ids

        # Stack attention masks
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        batch["attention_mask"] = attention_mask

        # Stack graph features
        graph_features = torch.stack([f["graph_features"] for f in features])
        batch["graph_features"] = graph_features

        # Handle labels
        if "labels" in features[0]:
            if self.mode == "regression":
                labels = torch.stack([f["labels"] for f in features])
            else:
                # DAPT mode: labels are token IDs
                labels = torch.stack([f["labels"] for f in features])
            batch["labels"] = labels

        return batch


def create_datasets(
    train_data: Sequence[Union[ChemGPTRSample, Dict[str, Any]]],
    tokenizer: Any,
    graph_extractor: Optional[GraphFeatureExtractor] = None,
    eval_data: Optional[Sequence[Union[ChemGPTRSample, Dict[str, Any]]]] = None,
    max_length: int = 512,
    mode: str = "regression",
) -> Dict[str, ChemGPTRDataset]:
    """Create train and eval datasets.

    Args:
        train_data: Training samples.
        tokenizer: HuggingFace tokenizer.
        graph_extractor: Graph feature extractor (created if not provided).
        eval_data: Evaluation samples (optional).
        max_length: Maximum sequence length.
        mode: Training mode.

    Returns:
        Dictionary with 'train' and optionally 'eval' datasets.
    """
    extractor = graph_extractor or GraphFeatureExtractor()

    datasets = {
        "train": ChemGPTRDataset(
            samples=train_data,
            tokenizer=tokenizer,
            graph_extractor=extractor,
            max_length=max_length,
            mode=mode,
        )
    }

    if eval_data is not None:
        datasets["eval"] = ChemGPTRDataset(
            samples=eval_data,
            tokenizer=tokenizer,
            graph_extractor=extractor,
            max_length=max_length,
            mode=mode,
        )

    return datasets


def load_csv_data(
    csv_path: str,
    smiles_column: str = "smiles",
    target_column: Optional[str] = None,
    text_column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load data from a CSV file.

    Args:
        csv_path: Path to the CSV file.
        smiles_column: Name of the SMILES column.
        target_column: Name of the target column (optional).
        text_column: Name of additional text column (optional).

    Returns:
        List of sample dictionaries.
    """
    import csv

    samples = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample: Dict[str, Any] = {"smiles": row[smiles_column]}
            if target_column and target_column in row:
                sample["target"] = float(row[target_column])
            if text_column and text_column in row:
                sample["text"] = row[text_column]
            samples.append(sample)
    return samples
