import numpy as np
import pytest
import torch
from pymatgen.core import Composition, Element

from app.featurizer import NOBLE_GAS_FALLBACKS
from tests import structures

# Node feature layout: [Z, electronegativity, atomic radius, ionization energy, group, coordination number]
X_COL, R_COL, IE_COL, GROUP_COL, COORD_COL = 1, 2, 3, 4, 5


# ---------------- formula -> tabular ----------------
def test_formula_to_tabular_shape_dtype_and_finite(featurizer):
    tabular = featurizer.formula_to_tabular("NaCl")
    assert tuple(tabular.shape) == (1, 132)
    assert tabular.dtype == torch.float32
    assert torch.isfinite(tabular).all()


def test_different_formulas_give_different_vectors(featurizer):
    assert not torch.allclose(featurizer.formula_to_tabular("NaCl"), featurizer.formula_to_tabular("Fe2O3"))


def test_tabular_features_are_standardized_with_the_training_scaler(featurizer):
    # Regression guard: the model was trained on scaled Magpie features; feeding raw ones was a real bug.
    scaled = featurizer.formula_to_tabular("Fe2O3").numpy()[0]
    raw = np.array(featurizer.magpie.featurize(Composition("Fe2O3")), dtype=float)
    assert not np.allclose(scaled, raw)
    unscaled = scaled * featurizer.tabular_scaler.scale_ + featurizer.tabular_scaler.mean_
    np.testing.assert_allclose(unscaled, raw, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("bad_formula", ["", "Xx9", "@@"])
def test_bad_formula_raises_value_error(featurizer, bad_formula):
    with pytest.raises(ValueError, match="Failed to featurize formula"):
        featurizer.formula_to_tabular(bad_formula)


# ---------------- CIF -> graph ----------------
def test_cif_to_graph_shapes_and_dtypes(featurizer, nacl_2atom_cscl_type_cif):
    graph = featurizer.cif_to_graph(nacl_2atom_cscl_type_cif)
    n_nodes, n_edges = graph.x.shape[0], graph.edge_index.shape[1]
    assert n_nodes == 2
    assert graph.x.shape == (n_nodes, 6) and graph.x.dtype == torch.float32
    assert graph.edge_index.shape == (2, n_edges) and graph.edge_index.dtype == torch.long
    assert graph.edge_attr.shape == (n_edges, 2) and graph.edge_attr.dtype == torch.float32
    assert graph.edge_attr[:, 0].max() <= featurizer.max_radius
    assert graph.edge_index.max() < n_nodes
    assert torch.isfinite(graph.x).all() and torch.isfinite(graph.edge_attr).all()


@pytest.mark.parametrize("bad_cif", ["not a cif at all", "", "data_x\n_cell_length_a 5.6\n"])
def test_bad_cif_raises_value_error(featurizer, bad_cif):
    with pytest.raises(ValueError):
        featurizer.cif_to_graph(bad_cif)


def test_every_node_keeps_exactly_k_nearest_neighbours(featurizer):
    graph = featurizer.cif_to_graph(structures.to_cif(structures.cu_fcc_supercell()))
    out_degree = torch.bincount(graph.edge_index[0], minlength=graph.x.shape[0])
    assert graph.x.shape[0] == 108
    assert set(out_degree.tolist()) == {featurizer.k}


def test_isolated_atom_gives_empty_edge_tensors(featurizer):
    graph = featurizer.cif_to_graph(structures.to_cif(structures.isolated_atom("Ar")))
    assert tuple(graph.edge_index.shape) == (2, 0)
    assert tuple(graph.edge_attr.shape) == (0, 2)
    assert graph.x[0, COORD_COL] == 0


def test_noble_gas_uses_hardcoded_fallbacks_and_median_electronegativity(featurizer, monkeypatch):
    monkeypatch.setitem(featurizer.impute_stats, "median_X", 99.0)  # sentinel so the imputed value is observable
    fallback = NOBLE_GAS_FALLBACKS["Ar"]
    node = featurizer.cif_to_graph(structures.to_cif(structures.isolated_atom("Ar"))).x[0]
    assert node[X_COL].item() == pytest.approx(99.0)
    assert node[R_COL].item() == pytest.approx(fallback["r"])
    assert node[IE_COL].item() == pytest.approx(fallback["ie"], rel=1e-5)
    assert node[GROUP_COL].item() == pytest.approx(fallback["group"])


def test_missing_elemental_data_is_imputed_with_medians(featurizer, monkeypatch):
    def _missing(value):
        return value is None or (isinstance(value, float) and np.isnan(value))

    # Fr/At lack some pymatgen elemental properties. Find which, then check those columns get the sentinel.
    element = next(
        e for e in (Element("Fr"), Element("At")) if any(_missing(v) for v in (e.X, e.atomic_radius, e.ionization_energy))
    )
    for key in ("median_X", "median_r", "median_ie", "median_group"):
        monkeypatch.setitem(featurizer.impute_stats, key, 99.0)

    node = featurizer.cif_to_graph(structures.to_cif(structures.isolated_atom(element.symbol))).x[0]

    for col, value in ((X_COL, element.X), (R_COL, element.atomic_radius), (IE_COL, element.ionization_energy)):
        if _missing(value):
            assert node[col].item() == pytest.approx(99.0)
        else:
            assert node[col].item() == pytest.approx(float(value), rel=1e-5)
    assert torch.isfinite(node).all()


# ---------------- combined ----------------
def test_process_input_attaches_tabular_features_and_batch(featurizer, nacl_2atom_cscl_type_cif):
    data = featurizer.process_input("NaCl", nacl_2atom_cscl_type_cif)
    assert tuple(data.tabular_features.shape) == (1, 132)
    assert data.batch.dtype == torch.long
    assert data.batch.shape[0] == data.x.shape[0]
    assert (data.batch == 0).all()
