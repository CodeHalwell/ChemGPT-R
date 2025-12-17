from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

DescriptorFn = Callable[[Chem.Mol], float]
DescriptorSpec = Sequence[Tuple[str, DescriptorFn]]

DEFAULT_DESCRIPTORS: DescriptorSpec = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
]


def smiles_to_mol(smiles: str) -> Chem.Mol:
    """Parse a SMILES string into an RDKit Mol with explicit validation."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES string must be a non-empty string.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return mol


def mol_to_graph(mol: Chem.Mol) -> nx.Graph:
    """Convert an RDKit Mol to an undirected NetworkX graph."""
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        graph.add_edge(i, j, order=bond.GetBondTypeAsDouble())
    return graph


def _normalized_laplacian_matrix(graph: nx.Graph) -> np.ndarray:
    """Compute a normalized Laplacian without requiring SciPy."""
    n = graph.number_of_nodes()
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    adjacency = nx.to_numpy_array(graph, dtype=float)
    degrees = adjacency.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt_deg = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    d_mat = np.diag(inv_sqrt_deg)
    laplacian = np.eye(n, dtype=float) - d_mat @ adjacency @ d_mat
    zero_degree = degrees == 0
    if np.any(zero_degree):
        laplacian[np.ix_(zero_degree, zero_degree)] = 0.0
    return laplacian


def normalized_laplacian_eigenvalues(
    graph: nx.Graph, k: int, tol: float = 1e-8
) -> np.ndarray:
    """
    Return the k smallest non-zero eigenvalues of the normalized Laplacian,
    padding with zeros when necessary.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    laplacian = _normalized_laplacian_matrix(graph)
    if laplacian.size == 0:
        return np.zeros(k, dtype=float)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = np.sort(np.real(eigenvalues))
    non_zero = eigenvalues[eigenvalues > tol]
    top_k = non_zero[:k]
    if len(top_k) < k:
        top_k = np.pad(top_k, (0, k - len(top_k)), constant_values=0.0)
    return top_k


def compute_descriptors(
    mol: Chem.Mol, descriptor_fns: DescriptorSpec | None = None
) -> np.ndarray:
    """Compute a vector of RDKit descriptors."""
    descriptors = descriptor_fns if descriptor_fns is not None else DEFAULT_DESCRIPTORS
    values: List[float] = []
    for _, fn in descriptors:
        values.append(float(fn(mol)))
    return np.asarray(values, dtype=float)


def extract_feature_vector(
    smiles: str,
    k_eigen: int = 8,
    use_descriptors: bool = True,
    descriptor_fns: DescriptorSpec | None = None,
    eigenvalue_tol: float = 1e-8,
) -> np.ndarray:
    """High level pipeline: SMILES -> feature vector."""
    mol = smiles_to_mol(smiles)
    graph = mol_to_graph(mol)
    spectrum = normalized_laplacian_eigenvalues(graph, k_eigen, tol=eigenvalue_tol)
    parts = [spectrum]
    if use_descriptors:
        parts.append(compute_descriptors(mol, descriptor_fns))
    return np.concatenate(parts)


@dataclass
class GraphFeatureExtractor:
    """Callable extractor suitable for training and inference."""

    k_eigen: int = 8
    use_descriptors: bool = True
    descriptor_fns: DescriptorSpec | None = None
    eigenvalue_tol: float = 1e-8

    def transform(self, smiles: str) -> np.ndarray:
        return extract_feature_vector(
            smiles=smiles,
            k_eigen=self.k_eigen,
            use_descriptors=self.use_descriptors,
            descriptor_fns=self.descriptor_fns,
            eigenvalue_tol=self.eigenvalue_tol,
        )

    def __call__(self, smiles: str) -> np.ndarray:
        return self.transform(smiles)
