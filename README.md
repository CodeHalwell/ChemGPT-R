# ChemGPT-R
Chemistry decoder using GPT - A Graph-Prefix Decoder Regressor for molecular property prediction.

## Overview

ChemGPT-R combines graph neural network features with large language models for accurate molecular property prediction. The architecture uses:

- **Quantized LLM backbone**: Mistral-7B or Llama-3-8B with QLoRA adapters for efficient fine-tuning
- **Graph prefix fusion**: MLP projects graph features (Laplacian eigenvalues + molecular descriptors) to prefix tokens
- **Regression head**: Pools prefix hidden states and projects to scalar output
- **Mixed precision**: bfloat16/4-bit quantization with gradient checkpointing for memory efficiency

## Installation

```bash
pip install -r requirements.txt
```

Note: RDKit is required for SMILES parsing. Install via conda:
```bash
conda install -c conda-forge rdkit
```

## Graph Feature Extraction

Utilities for turning SMILES strings into feature vectors are available in `chemgpt_r.graph_features`.

```python
from chemgpt_r import GraphFeatureExtractor

extractor = GraphFeatureExtractor(k_eigen=8, use_descriptors=True)
features = extractor("CCO")  # numpy array with Laplacian spectrum + descriptors
```

## Model Architecture

### Configuration

```python
from chemgpt_r import ChemGPTRConfig, GraphPrefixConfig, LoraConfig

config = ChemGPTRConfig(
    model_name_or_path="mistralai/Mistral-7B-v0.1",
    prefix=GraphPrefixConfig(
        graph_feature_dim=11,  # 8 eigenvalues + 3 descriptors
        num_prefix_tokens=8,
        hidden_dim=256,
    ),
    lora=LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
    ),
    load_in_4bit=True,
    torch_dtype="bfloat16",
    use_gradient_checkpointing=True,
    mode="regression",  # or "dapt" for causal LM pretraining
)
```

### Loading the Model

```python
from chemgpt_r import GraphPrefixDecoderRegressor

model = GraphPrefixDecoderRegressor.from_pretrained(config)
model.print_trainable_parameters()
```

## Training

### Prepare Data

```python
from chemgpt_r import (
    ChemGPTRDataset,
    ChemGPTRDataCollator,
    create_datasets,
    load_csv_data,
)

# Load from CSV
data = load_csv_data("molecules.csv", smiles_column="smiles", target_column="activity")

# Or create samples manually
from chemgpt_r import ChemGPTRSample
samples = [
    ChemGPTRSample(smiles="CCO", target=1.5),
    ChemGPTRSample(smiles="CC(=O)O", target=2.3),
]

# Create datasets
datasets = create_datasets(
    train_data=samples,
    tokenizer=model.tokenizer,
    max_length=512,
    mode="regression",
)
```

### Train with Custom Trainer

```python
from chemgpt_r import (
    ChemGPTRTrainer,
    ChemGPTRDataCollator,
    compute_regression_metrics,
    create_training_args,
)

# Create training arguments
training_args = create_training_args(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
)

# Create data collator
collator = ChemGPTRDataCollator(tokenizer=model.tokenizer, mode="regression")

# Create trainer
trainer = ChemGPTRTrainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets.get("eval"),
    data_collator=collator,
    compute_metrics=compute_regression_metrics,
)

# Train
trainer.train()

# Save model
trainer.save_model("./trained_model")
```

## Training Modes

### Regression Mode (MSE Loss)
For molecular property prediction with scalar targets:
```python
config = ChemGPTRConfig(mode="regression")
```

### DAPT Mode (Causal LM Loss)
For domain-adaptive pretraining on SMILES:
```python
config = ChemGPTRConfig(mode="dapt")
```

## Example Inference

```python
import torch
from chemgpt_r import GraphFeatureExtractor

# Extract features
extractor = GraphFeatureExtractor()
smiles = "CCO"
graph_features = torch.tensor(extractor(smiles)).unsqueeze(0).float()

# Tokenize
inputs = model.tokenizer(f"SMILES: {smiles}", return_tensors="pt")

# Predict
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        graph_features=graph_features.to(model.base_model.device),
    )
    prediction = outputs["predictions"]
    print(f"Predicted property: {prediction.item():.4f}")
```

## Dependencies

- PyTorch >= 2.0.0
- Transformers >= 4.36.0
- PEFT >= 0.7.0
- Accelerate >= 0.25.0
- BitsAndBytes >= 0.41.0
- RDKit (for SMILES parsing/descriptors)
- NumPy, NetworkX

See `requirements.txt` for full Python dependencies.
