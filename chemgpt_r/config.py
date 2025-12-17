"""Configuration dataclasses for ChemGPT-R model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class GraphPrefixConfig:
    """Configuration for graph feature prefix projection.

    Args:
        graph_feature_dim: Dimension of input graph features.
        num_prefix_tokens: Number of prefix tokens (m) to project to.
        hidden_dim: Hidden dimension for the MLP projector.
        dropout: Dropout rate in the projector MLP.
    """

    graph_feature_dim: int = 11  # Default: 8 eigenvalues + 3 descriptors
    num_prefix_tokens: int = 8
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class LoraConfig:
    """Configuration for QLoRA adapters.

    Args:
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout rate for LoRA layers.
        target_modules: List of module names to apply LoRA to.
        bias: Bias handling strategy ('none', 'all', 'lora_only').
        task_type: Task type for PEFT configuration.
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class RegressionHeadConfig:
    """Configuration for the regression head.

    Args:
        hidden_dim: Hidden dimension for the regression MLP.
        dropout: Dropout rate.
        pooling: Pooling strategy for prefix hidden states.
    """

    hidden_dim: int = 256
    dropout: float = 0.1
    pooling: Literal["mean", "max", "first"] = "mean"


@dataclass
class ChemGPTRConfig:
    """Main configuration for ChemGPT-R model.

    Args:
        model_name_or_path: HuggingFace model identifier or path.
        prefix: Graph prefix configuration.
        lora: QLoRA adapter configuration.
        regression_head: Regression head configuration.
        load_in_4bit: Whether to load model in 4-bit quantization.
        load_in_8bit: Whether to load model in 8-bit quantization.
        torch_dtype: Data type for model weights ('float16', 'bfloat16', 'float32').
        use_gradient_checkpointing: Whether to enable gradient checkpointing.
        max_seq_length: Maximum sequence length for tokenization.
        mode: Training mode ('regression' for MSE, 'dapt' for causal LM).
    """

    model_name_or_path: str = "mistralai/Mistral-7B-v0.1"
    prefix: GraphPrefixConfig = field(default_factory=GraphPrefixConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    regression_head: RegressionHeadConfig = field(default_factory=RegressionHeadConfig)
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    use_gradient_checkpointing: bool = True
    max_seq_length: int = 512
    mode: Literal["regression", "dapt"] = "regression"
    device_map: Optional[str] = "auto"


@dataclass
class TrainingConfig:
    """Configuration for training.

    Args:
        output_dir: Directory for saving model checkpoints.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Training batch size per device.
        per_device_eval_batch_size: Evaluation batch size per device.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        learning_rate: Learning rate.
        weight_decay: Weight decay for optimizer.
        warmup_ratio: Warmup ratio for learning rate scheduler.
        logging_steps: Logging frequency (steps).
        eval_strategy: Evaluation strategy ('no', 'steps', 'epoch').
        save_strategy: Save strategy ('no', 'steps', 'epoch').
        fp16: Use FP16 mixed precision.
        bf16: Use BF16 mixed precision.
        dataloader_num_workers: Number of dataloader workers.
        seed: Random seed.
    """

    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    eval_strategy: Literal["no", "steps", "epoch"] = "epoch"
    save_strategy: Literal["no", "steps", "epoch"] = "epoch"
    fp16: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 4
    seed: int = 42
