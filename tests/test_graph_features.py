import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from chemgpt_r.graph_features import (
    compute_descriptors,
    extract_feature_vector,
    normalized_laplacian_eigenvalues,
    mol_to_graph,
    smiles_to_mol,
)


def test_invalid_smiles_raises_value_error():
    with pytest.raises(ValueError):
        smiles_to_mol("not_a_smiles")


def test_single_atom_spectrum_padding():
    features = extract_feature_vector("C", k_eigen=4, use_descriptors=False)
    assert np.allclose(features, np.zeros(4))


def test_descriptor_values_match_rdkit():
    smiles = "CCO"
    mol = Chem.MolFromSmiles(smiles)
    expected = np.array(
        [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol)]
    )
    features = extract_feature_vector(smiles, k_eigen=1, use_descriptors=True)
    descriptor_part = features[1:]  # first item is eigenvalue
    assert descriptor_part.shape[0] == expected.shape[0]
    assert np.allclose(descriptor_part, expected, atol=1e-6)


def test_smallest_nonzero_eigenvalues_filtered_and_padded():
    mol = smiles_to_mol("CC")
    graph = mol_to_graph(mol)
    eigenvalues = normalized_laplacian_eigenvalues(graph, k=3)
    # normalized laplacian for a two-node chain has eigenvalues [0, 2]
    assert np.allclose(eigenvalues, np.array([2.0, 0.0, 0.0]))
