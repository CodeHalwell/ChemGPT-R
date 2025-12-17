# ChemGPT-R
Chemistry decoder using GPT

## Graph Feature Extraction

Utilities for turning SMILES strings into feature vectors are available in `chemgpt_r.graph_features`.

Example:
```
from chemgpt_r import GraphFeatureExtractor

extractor = GraphFeatureExtractor(k_eigen=8, use_descriptors=True)
features = extractor("CCO")  # numpy array with Laplacian spectrum + descriptors
```

Dependencies: RDKit (for SMILES parsing/descriptors), NumPy, NetworkX. See `requirements.txt` for Python dependencies.
