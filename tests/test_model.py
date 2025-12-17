"""Tests for ChemGPT-R model components.

These tests validate the model architecture components without loading
a full LLM to keep tests fast and not require GPU/large downloads.
"""

import numpy as np
import pytest
import torch

from chemgpt_r.config import (
    ChemGPTRConfig,
    GraphPrefixConfig,
    LoraConfig,
    RegressionHeadConfig,
    TrainingConfig,
)
from chemgpt_r.model import (
    GraphPrefixProjector,
    RegressionHead,
)
from chemgpt_r.data import (
    ChemGPTRSample,
    ChemGPTRDataCollator,
)


class TestGraphPrefixConfig:
    """Tests for GraphPrefixConfig."""

    def test_default_values(self):
        config = GraphPrefixConfig()
        assert config.graph_feature_dim == 11
        assert config.num_prefix_tokens == 8
        assert config.hidden_dim == 256
        assert config.dropout == 0.1

    def test_custom_values(self):
        config = GraphPrefixConfig(
            graph_feature_dim=16,
            num_prefix_tokens=4,
            hidden_dim=128,
            dropout=0.2,
        )
        assert config.graph_feature_dim == 16
        assert config.num_prefix_tokens == 4
        assert config.hidden_dim == 128
        assert config.dropout == 0.2


class TestLoraConfig:
    """Tests for LoraConfig."""

    def test_default_target_modules(self):
        config = LoraConfig()
        expected_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        assert config.target_modules == expected_modules

    def test_default_values(self):
        config = LoraConfig()
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"


class TestChemGPTRConfig:
    """Tests for ChemGPTRConfig."""

    def test_default_mode_is_regression(self):
        config = ChemGPTRConfig()
        assert config.mode == "regression"

    def test_nested_configs(self):
        config = ChemGPTRConfig(
            prefix=GraphPrefixConfig(num_prefix_tokens=4),
            lora=LoraConfig(r=8),
        )
        assert config.prefix.num_prefix_tokens == 4
        assert config.lora.r == 8

    def test_quantization_options(self):
        config = ChemGPTRConfig(load_in_4bit=True, load_in_8bit=False)
        assert config.load_in_4bit is True
        assert config.load_in_8bit is False


class TestGraphPrefixProjector:
    """Tests for GraphPrefixProjector."""

    def test_output_shape(self):
        batch_size = 4
        graph_feature_dim = 11
        embed_dim = 768
        num_prefix_tokens = 8

        projector = GraphPrefixProjector(
            graph_feature_dim=graph_feature_dim,
            embed_dim=embed_dim,
            num_prefix_tokens=num_prefix_tokens,
        )

        input_features = torch.randn(batch_size, graph_feature_dim)
        output = projector(input_features)

        assert output.shape == (batch_size, num_prefix_tokens, embed_dim)

    def test_different_configurations(self):
        configs = [
            (8, 512, 4, 128),   # Small config
            (16, 1024, 16, 512),  # Medium config
            (32, 4096, 32, 1024),  # Large config
        ]

        for graph_dim, embed_dim, n_tokens, hidden_dim in configs:
            projector = GraphPrefixProjector(
                graph_feature_dim=graph_dim,
                embed_dim=embed_dim,
                num_prefix_tokens=n_tokens,
                hidden_dim=hidden_dim,
            )

            x = torch.randn(2, graph_dim)
            output = projector(x)
            assert output.shape == (2, n_tokens, embed_dim)

    def test_gradients_flow(self):
        projector = GraphPrefixProjector(
            graph_feature_dim=11,
            embed_dim=256,
            num_prefix_tokens=4,
        )

        x = torch.randn(2, 11, requires_grad=True)
        output = projector(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestRegressionHead:
    """Tests for RegressionHead."""

    def test_output_shape(self):
        batch_size = 4
        seq_len = 16
        hidden_dim = 768
        num_prefix_tokens = 8

        head = RegressionHead(hidden_dim=hidden_dim)
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        output = head(hidden_states, num_prefix_tokens)

        assert output.shape == (batch_size, 1)

    def test_mean_pooling(self):
        head = RegressionHead(hidden_dim=64, pooling="mean")
        hidden_states = torch.ones(2, 10, 64)
        output = head(hidden_states, num_prefix_tokens=5)
        assert output.shape == (2, 1)

    def test_max_pooling(self):
        head = RegressionHead(hidden_dim=64, pooling="max")
        hidden_states = torch.randn(2, 10, 64)
        output = head(hidden_states, num_prefix_tokens=5)
        assert output.shape == (2, 1)

    def test_first_pooling(self):
        head = RegressionHead(hidden_dim=64, pooling="first")
        hidden_states = torch.randn(2, 10, 64)
        output = head(hidden_states, num_prefix_tokens=5)
        assert output.shape == (2, 1)

    def test_invalid_pooling_raises(self):
        head = RegressionHead(hidden_dim=64, pooling="invalid")
        hidden_states = torch.randn(2, 10, 64)
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            head(hidden_states, num_prefix_tokens=5)

    def test_gradients_flow(self):
        head = RegressionHead(hidden_dim=64)
        hidden_states = torch.randn(2, 10, 64, requires_grad=True)
        output = head(hidden_states, num_prefix_tokens=5)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None


class TestChemGPTRSample:
    """Tests for ChemGPTRSample dataclass."""

    def test_required_smiles(self):
        sample = ChemGPTRSample(smiles="CCO")
        assert sample.smiles == "CCO"
        assert sample.target is None
        assert sample.text is None

    def test_with_target(self):
        sample = ChemGPTRSample(smiles="CCO", target=1.5)
        assert sample.smiles == "CCO"
        assert sample.target == 1.5

    def test_with_text(self):
        sample = ChemGPTRSample(smiles="CCO", text="Ethanol molecule")
        assert sample.text == "Ethanol molecule"


class TestChemGPTRDataCollator:
    """Tests for ChemGPTRDataCollator."""

    def test_collate_batch(self):
        # Create mock tokenizer
        class MockTokenizer:
            pass

        collator = ChemGPTRDataCollator(tokenizer=MockTokenizer(), mode="regression")

        features = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "attention_mask": torch.tensor([1, 1, 1, 0]),
                "graph_features": torch.tensor([0.1, 0.2, 0.3]),
                "labels": torch.tensor(1.5),
            },
            {
                "input_ids": torch.tensor([5, 6, 7, 8]),
                "attention_mask": torch.tensor([1, 1, 0, 0]),
                "graph_features": torch.tensor([0.4, 0.5, 0.6]),
                "labels": torch.tensor(2.5),
            },
        ]

        batch = collator(features)

        assert batch["input_ids"].shape == (2, 4)
        assert batch["attention_mask"].shape == (2, 4)
        assert batch["graph_features"].shape == (2, 3)
        assert batch["labels"].shape == (2,)


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        config = TrainingConfig()
        assert config.output_dir == "./output"
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 4
        assert config.bf16 is True
        assert config.fp16 is False

    def test_custom_values(self):
        config = TrainingConfig(
            output_dir="/custom/path",
            num_train_epochs=10,
            learning_rate=1e-5,
        )
        assert config.output_dir == "/custom/path"
        assert config.num_train_epochs == 10
        assert config.learning_rate == 1e-5
