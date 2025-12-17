"""Graph-Prefix Decoder Regressor model for ChemGPT-R.

This module implements the main regression architecture with:
- Quantized decoder-only LLM backbone (via transformers/peft with QLoRA)
- Prefix fusion via trainable MLP projecting graph features to prefix tokens
- QLoRA adapters on attention and MLP layers
- Regression head pooling prefix hidden states to scalar output
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .config import ChemGPTRConfig, GraphPrefixConfig, LoraConfig, RegressionHeadConfig


class GraphPrefixProjector(nn.Module):
    """Projects graph features to prefix token embeddings.

    Args:
        graph_feature_dim: Input dimension of graph features.
        embed_dim: Output dimension (LLM embedding dimension).
        num_prefix_tokens: Number of prefix tokens to generate.
        hidden_dim: Hidden dimension for the MLP.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        graph_feature_dim: int,
        embed_dim: int,
        num_prefix_tokens: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.embed_dim = embed_dim

        # Two-layer MLP with GELU activation
        self.projector = nn.Sequential(
            nn.Linear(graph_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_prefix_tokens * embed_dim),
        )

    def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
        """Project graph features to prefix embeddings.

        Args:
            graph_features: Tensor of shape (batch_size, graph_feature_dim).

        Returns:
            Tensor of shape (batch_size, num_prefix_tokens, embed_dim).
        """
        batch_size = graph_features.size(0)
        projected = self.projector(graph_features)
        return projected.view(batch_size, self.num_prefix_tokens, self.embed_dim)


class RegressionHead(nn.Module):
    """Regression head that pools prefix hidden states and projects to scalar.

    Args:
        hidden_dim: LLM hidden dimension.
        intermediate_dim: Intermediate dimension for the MLP.
        dropout: Dropout rate.
        pooling: Pooling strategy ('mean', 'max', 'first').
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()
        self.pooling = pooling
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1),
        )

    def forward(
        self, hidden_states: torch.Tensor, num_prefix_tokens: int
    ) -> torch.Tensor:
        """Pool prefix hidden states and project to scalar.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim).
            num_prefix_tokens: Number of prefix tokens to pool over.

        Returns:
            Tensor of shape (batch_size, 1).
        """
        # Extract prefix hidden states
        prefix_hidden = hidden_states[:, :num_prefix_tokens, :]

        # Pool prefix hidden states
        if self.pooling == "mean":
            pooled = prefix_hidden.mean(dim=1)
        elif self.pooling == "max":
            pooled = prefix_hidden.max(dim=1).values
        elif self.pooling == "first":
            pooled = prefix_hidden[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return self.head(pooled)


class GraphPrefixDecoderRegressor(nn.Module):
    """Main Graph-Prefix Decoder Regressor model.

    This model combines:
    - A quantized decoder-only LLM backbone with QLoRA adapters
    - Graph feature prefix projection
    - Regression head for scalar output

    Args:
        config: Model configuration.
        base_model: Pre-loaded base language model (optional).
        tokenizer: Pre-loaded tokenizer (optional).
    """

    def __init__(
        self,
        config: ChemGPTRConfig,
        base_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        super().__init__()
        self.config = config

        # Store references (initialized via from_pretrained or set externally)
        self.base_model = base_model
        self.tokenizer = tokenizer

        # These will be initialized once we have the LLM hidden dim
        self.prefix_projector: Optional[GraphPrefixProjector] = None
        self.regression_head: Optional[RegressionHead] = None

        # Track if model is fully initialized
        self._initialized = False

    @property
    def num_prefix_tokens(self) -> int:
        """Number of prefix tokens."""
        return self.config.prefix.num_prefix_tokens

    @property
    def mode(self) -> str:
        """Training mode ('regression' or 'dapt')."""
        return self.config.mode

    def _get_hidden_dim(self) -> int:
        """Get the hidden dimension from the base model config."""
        if self.base_model is None:
            raise RuntimeError("Base model not initialized.")
        model_config = self.base_model.config
        # Try common attribute names for hidden dimension
        for attr in ("hidden_size", "n_embd", "d_model"):
            if hasattr(model_config, attr):
                return getattr(model_config, attr)
        raise AttributeError(
            f"Cannot determine hidden dimension from model config. "
            f"Expected 'hidden_size', 'n_embd', or 'd_model' attribute."
        )

    def _initialize_components(self) -> None:
        """Initialize prefix projector and regression head based on LLM config."""
        if self._initialized:
            return

        hidden_dim = self._get_hidden_dim()
        prefix_cfg = self.config.prefix
        reg_cfg = self.config.regression_head

        self.prefix_projector = GraphPrefixProjector(
            graph_feature_dim=prefix_cfg.graph_feature_dim,
            embed_dim=hidden_dim,
            num_prefix_tokens=prefix_cfg.num_prefix_tokens,
            hidden_dim=prefix_cfg.hidden_dim,
            dropout=prefix_cfg.dropout,
        )

        self.regression_head = RegressionHead(
            hidden_dim=hidden_dim,
            intermediate_dim=reg_cfg.hidden_dim,
            dropout=reg_cfg.dropout,
            pooling=reg_cfg.pooling,
        )

        self._initialized = True

    @classmethod
    def from_pretrained(
        cls,
        config: ChemGPTRConfig,
        **kwargs: Any,
    ) -> "GraphPrefixDecoderRegressor":
        """Load model from a pretrained HuggingFace model.

        Args:
            config: Model configuration.
            **kwargs: Additional arguments for model loading.

        Returns:
            Initialized GraphPrefixDecoderRegressor instance.
        """
        # Lazy imports to avoid requiring transformers at module load
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

        # Configure quantization
        quantization_config = None
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=kwargs.get("trust_remote_code", True),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": kwargs.get("trust_remote_code", True),
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if config.device_map is not None:
            model_kwargs["device_map"] = config.device_map

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            **model_kwargs,
        )

        # Enable gradient checkpointing if requested
        if config.use_gradient_checkpointing:
            base_model.gradient_checkpointing_enable()

        # Apply LoRA adapters
        base_model = cls._apply_lora(base_model, config.lora)

        # Create instance
        model = cls(config, base_model=base_model, tokenizer=tokenizer)
        model._initialize_components()

        # Move new components to same device/dtype as base model
        device = next(base_model.parameters()).device
        if model.prefix_projector is not None:
            model.prefix_projector = model.prefix_projector.to(device, dtype=torch_dtype)
        if model.regression_head is not None:
            model.regression_head = model.regression_head.to(device, dtype=torch_dtype)

        return model

    @staticmethod
    def _apply_lora(
        model: Any,
        lora_config: LoraConfig,
    ) -> Any:
        """Apply QLoRA adapters to the model.

        Args:
            model: Base language model.
            lora_config: LoRA configuration.

        Returns:
            Model with LoRA adapters applied.
        """
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model, prepare_model_for_kbit_training

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Create PEFT LoRA config
        peft_config = PeftLoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
        )

        # Apply LoRA
        model = get_peft_model(model, peft_config)
        return model

    def get_input_embeddings(self) -> nn.Module:
        """Get the input embedding layer from base model."""
        if self.base_model is None:
            raise RuntimeError("Base model not initialized.")
        return self.base_model.get_input_embeddings()

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare inputs by concatenating prefix embeddings with token embeddings.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            graph_features: Graph features of shape (batch_size, graph_feature_dim).

        Returns:
            Tuple of (inputs_embeds, attention_mask) with prefix prepended.
        """
        if self.prefix_projector is None:
            raise RuntimeError("Prefix projector not initialized.")

        batch_size = input_ids.size(0)

        # Get token embeddings
        token_embeddings = self.get_input_embeddings()(input_ids)

        # Project graph features to prefix embeddings
        prefix_embeddings = self.prefix_projector(graph_features)

        # Concatenate prefix with token embeddings
        inputs_embeds = torch.cat([prefix_embeddings, token_embeddings], dim=1)

        # Extend attention mask for prefix tokens
        prefix_attention = torch.ones(
            batch_size,
            self.num_prefix_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        return inputs_embeds, extended_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        graph_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for the model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            graph_features: Graph features of shape (batch_size, graph_feature_dim).
            labels: Target values for regression or next-token labels for DAPT.
            return_dict: Whether to return a dictionary.
            **kwargs: Additional arguments passed to base model.

        Returns:
            Dictionary containing loss and predictions.
        """
        if self.base_model is None:
            raise RuntimeError("Base model not initialized.")
        if input_ids is None:
            raise ValueError("input_ids is required.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if graph_features is None:
            raise ValueError("graph_features is required.")

        # Prepare inputs with prefix
        inputs_embeds, extended_attention_mask = self._prepare_inputs(
            input_ids, attention_mask, graph_features
        )

        # Forward through base model
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        result: Dict[str, torch.Tensor] = {}

        if self.mode == "regression":
            # Get last hidden states and apply regression head
            hidden_states = outputs.hidden_states[-1]
            predictions = self.regression_head(hidden_states, self.num_prefix_tokens)
            result["predictions"] = predictions.squeeze(-1)

            if labels is not None:
                loss_fn = nn.MSELoss()
                result["loss"] = loss_fn(predictions.squeeze(-1), labels.float())

        elif self.mode == "dapt":
            # DAPT mode: causal language modeling
            # Shift logits and labels for next-token prediction
            logits = outputs.logits
            result["logits"] = logits

            if labels is not None:
                # Extend labels with -100 for prefix tokens (ignored in loss)
                batch_size = labels.size(0)
                prefix_labels = torch.full(
                    (batch_size, self.num_prefix_tokens),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                extended_labels = torch.cat([prefix_labels, labels], dim=1)

                # Compute causal LM loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = extended_labels[..., 1:].contiguous()
                loss_fn = nn.CrossEntropyLoss()
                result["loss"] = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

        return result

    def save_pretrained(self, save_directory: str) -> None:
        """Save the model to a directory.

        Args:
            save_directory: Directory to save the model.
        """
        import json
        import os
        from dataclasses import asdict

        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        # Save base model (with LoRA adapters)
        if self.base_model is not None:
            self.base_model.save_pretrained(os.path.join(save_directory, "base_model"))

        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))

        # Save prefix projector and regression head
        if self.prefix_projector is not None:
            torch.save(
                self.prefix_projector.state_dict(),
                os.path.join(save_directory, "prefix_projector.pt"),
            )
        if self.regression_head is not None:
            torch.save(
                self.regression_head.state_dict(),
                os.path.join(save_directory, "regression_head.pt"),
            )

    @classmethod
    def load_pretrained(
        cls,
        load_directory: str,
        **kwargs: Any,
    ) -> "GraphPrefixDecoderRegressor":
        """Load model from a saved directory.

        Args:
            load_directory: Directory containing the saved model.
            **kwargs: Additional arguments.

        Returns:
            Loaded GraphPrefixDecoderRegressor instance.
        """
        import json
        import os

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Load config
        config_path = os.path.join(load_directory, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Reconstruct config objects
        from .config import (
            ChemGPTRConfig,
            GraphPrefixConfig,
            LoraConfig,
            RegressionHeadConfig,
        )

        prefix_config = GraphPrefixConfig(**config_dict["prefix"])
        lora_config = LoraConfig(**config_dict["lora"])
        reg_config = RegressionHeadConfig(**config_dict["regression_head"])

        config = ChemGPTRConfig(
            model_name_or_path=config_dict["model_name_or_path"],
            prefix=prefix_config,
            lora=lora_config,
            regression_head=reg_config,
            load_in_4bit=config_dict.get("load_in_4bit", True),
            load_in_8bit=config_dict.get("load_in_8bit", False),
            torch_dtype=config_dict.get("torch_dtype", "bfloat16"),
            use_gradient_checkpointing=config_dict.get(
                "use_gradient_checkpointing", True
            ),
            max_seq_length=config_dict.get("max_seq_length", 512),
            mode=config_dict.get("mode", "regression"),
            device_map=config_dict.get("device_map", "auto"),
        )

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.torch_dtype, torch.bfloat16)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(load_directory, "tokenizer")
        )

        # Configure quantization for base model
        quantization_config = None
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if config.device_map is not None:
            model_kwargs["device_map"] = config.device_map

        # Load base model with LoRA adapters
        base_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                config.model_name_or_path,
                **model_kwargs,
            ),
            os.path.join(load_directory, "base_model"),
        )

        # Create instance
        model = cls(config, base_model=base_model, tokenizer=tokenizer)
        model._initialize_components()

        # Load prefix projector weights
        prefix_path = os.path.join(load_directory, "prefix_projector.pt")
        if os.path.exists(prefix_path) and model.prefix_projector is not None:
            model.prefix_projector.load_state_dict(torch.load(prefix_path))

        # Load regression head weights
        reg_path = os.path.join(load_directory, "regression_head.pt")
        if os.path.exists(reg_path) and model.regression_head is not None:
            model.regression_head.load_state_dict(torch.load(reg_path))

        # Move to appropriate device
        device = next(base_model.parameters()).device
        if model.prefix_projector is not None:
            model.prefix_projector = model.prefix_projector.to(device, dtype=torch_dtype)
        if model.regression_head is not None:
            model.regression_head = model.regression_head.to(device, dtype=torch_dtype)

        return model

    def print_trainable_parameters(self) -> None:
        """Print the number of trainable parameters."""
        trainable = 0
        total = 0

        # Count base model parameters
        if self.base_model is not None:
            for param in self.base_model.parameters():
                total += param.numel()
                if param.requires_grad:
                    trainable += param.numel()

        # Count prefix projector parameters
        if self.prefix_projector is not None:
            for param in self.prefix_projector.parameters():
                total += param.numel()
                if param.requires_grad:
                    trainable += param.numel()

        # Count regression head parameters
        if self.regression_head is not None:
            for param in self.regression_head.parameters():
                total += param.numel()
                if param.requires_grad:
                    trainable += param.numel()

        print(
            f"trainable params: {trainable:,d} || "
            f"all params: {total:,d} || "
            f"trainable%: {100 * trainable / total:.4f}"
        )
