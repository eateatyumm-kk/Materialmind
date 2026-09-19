"""Small, deterministic crystal structures shared by the tests and the golden-value regenerator."""

from pymatgen.core import Lattice, Structure


def nacl_2atom_cscl_type() -> Structure:
    # Na (0,0,0) + Cl (1/2,1/2,1/2) in a simple-cubic cell is CsCl-type, NOT rock salt. This is the
    # structure the golden values were generated from, so it keeps its historical shape.
    return Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def nacl_rocksalt() -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def cu_fcc_supercell() -> Structure:
    structure = Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.61), ["Cu"], [[0, 0, 0]])
    structure.make_supercell([3, 3, 3])
    return structure


def isolated_atom(symbol: str) -> Structure:
    return Structure(Lattice.cubic(20.0), [symbol], [[0, 0, 0]])


def to_cif(structure: Structure) -> str:
    return structure.to(fmt="cif")
