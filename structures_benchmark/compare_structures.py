#!/usr/bin/env python3
"""Compare pygamlab GAM_architectures structures with ASE reference geometries.

Accuracy against the ASE geometry:
- 2D sheets: RMSD of each pygamlab atom to the ASE crystal lattice
- nanotubes and nanoparticles: RMSD after rigid alignment of the two clusters
Bond-length error is the absolute difference of the mean nearest-neighbor distances.
Performance is the median build time and the atom count.

Reference builders
------------------
graphene     ase.build.graphene
silicene     ase.build.graphene(formula="Si2", thickness=buckling)
nanotube     ase.build.nanotube
nanoparticle ase.build.bulk (FCC sphere) and ase.cluster.Octahedron
"""

from __future__ import annotations

import csv
import json
import time
from math import gcd, sqrt
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, graphene, nanotube
from ase.cluster import Octahedron
from ase.io import write
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from pygamlab.structures.GAM_architectures import (
    Graphene,
    Nanoparticle_Generator,
    Nanotube_Generator,
    Silicene,
)
from pygamlab.structures.GAM_architectures.GAM_Graphene import graphene_unit_cell
from pygamlab.structures.GAM_architectures.GAM_sillicene import silicene_unit_cell

import ase
import pygamlab

OUT_DIR = Path(__file__).resolve().parent / "outputs"
XYZ_DIR = OUT_DIR / "xyz"

GRAPHENE_A = 2.46
SILICENE_A = 3.87
SILICENE_BUCKLING = 0.44
NANOTUBE_BOND = 1.42
AU_A = 4.08
REPEATS = 5


def median_time_ms(build, repeats: int = REPEATS):
    """Warm up once, then return the last structure and the median build time."""
    build()
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = build()
        samples.append((time.perf_counter() - start) * 1000.0)
    return result, float(np.median(samples))


def as_atoms(symbols, positions, cell=None, pbc=None) -> Atoms:
    atoms = Atoms(symbols=list(symbols), positions=np.asarray(positions, dtype=float))
    if cell is not None:
        atoms.set_cell(cell)
    if pbc is not None:
        atoms.set_pbc(pbc)
    return atoms


def positions_of(atoms: Atoms) -> np.ndarray:
    return np.asarray(atoms.get_positions(), dtype=float)


def honeycomb_supercell(basis, a1, a2, nx: int, ny: int, symbol: str, vacuum: float = 20.0) -> Atoms:
    """Tile a 2-atom hexagonal basis the same way a finite supercell is repeated."""
    points = []
    for i in range(nx):
        for j in range(ny):
            origin = i * a1 + j * a2
            for site in basis:
                points.append(origin + site)
    points = np.asarray(points, dtype=float)
    cell = np.array([nx * a1, ny * a2, [0.0, 0.0, vacuum]], dtype=float)
    return as_atoms([symbol] * len(points), points, cell=cell, pbc=(True, True, False))


def pygamlab_graphene_supercell(nx: int, ny: int, lattice: float = GRAPHENE_A) -> Atoms:
    basis, vectors = graphene_unit_cell(lattice)
    return honeycomb_supercell(basis, vectors[0], vectors[1], nx, ny, "C")


def ase_graphene_supercell(nx: int, ny: int, lattice: float = GRAPHENE_A) -> Atoms:
    atoms = graphene(a=lattice, size=(nx, ny, 1))
    cell = atoms.cell.array.copy()
    cell[2] = [0.0, 0.0, 20.0]
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, False))
    return atoms


def pygamlab_silicene_supercell(nx: int, ny: int) -> Atoms:
    basis, vectors = silicene_unit_cell(SILICENE_A)
    buckled = []
    for sublattice, site in enumerate(basis):
        z_offset = SILICENE_BUCKLING / 2.0 if sublattice == 0 else -SILICENE_BUCKLING / 2.0
        buckled.append(np.asarray(site, dtype=float) + np.array([0.0, 0.0, z_offset]))
    return honeycomb_supercell(buckled, vectors[0], vectors[1], nx, ny, "Si")


def ase_silicene_supercell(nx: int, ny: int) -> Atoms:
    atoms = graphene(
        formula="Si2",
        a=SILICENE_A,
        thickness=SILICENE_BUCKLING,
        size=(nx, ny, 1),
    )
    cell = atoms.cell.array.copy()
    cell[2] = [0.0, 0.0, 20.0]
    atoms.set_cell(cell)
    atoms.set_pbc((True, True, False))
    return atoms


def nanotube_period(n: int, m: int, bond: float = NANOTUBE_BOND) -> float:
    """Translational period shared by ase.build.nanotube and Nanotube_Generator."""
    if n < m:
        n, m = m, n
    sq3 = sqrt(3.0)
    a = sq3 * bond
    l2 = n * n + m * m + n * m
    nd = gcd(n, m)
    ndr = 3 * nd if (n - m) % (3 * nd) == 0 else nd
    c = a * sqrt(l2)
    return abs(sq3 * c / ndr)


def ase_fcc_sphere(symbol: str, lattice: float, radius: float) -> Atoms:
    """FCC sphere cut from ase.build.bulk, centered the same way as a lattice-site sphere."""
    n = int(np.ceil((2.0 * radius) / lattice)) + 3
    block = bulk(symbol, "fcc", a=lattice, cubic=True).repeat((n, n, n))
    center = block.cell.diagonal() / 2.0
    origin = block.positions[np.argmin(np.linalg.norm(block.positions - center, axis=1))]
    block.positions -= origin
    keep = np.linalg.norm(block.positions, axis=1) <= radius + 1e-8
    sphere = block[keep]
    sphere.positions -= sphere.positions.mean(axis=0)
    sphere.set_pbc(False)
    sphere.set_cell([0.0, 0.0, 0.0])
    return sphere


def gam_to_atoms(structure) -> Atoms:
    return structure.to_ase()


def neighbor_bonds(atoms: Atoms) -> np.ndarray:
    """Unique nearest-neighbor distances, using the minimum image when the cell is periodic."""
    n = len(atoms)
    if n < 2:
        return np.array([])
    use_mic = atoms.cell.rank >= 2 and np.any(atoms.pbc)
    if use_mic:
        matrix = atoms.get_all_distances(mic=True)
    else:
        matrix = cdist(atoms.positions, atoms.positions)
    np.fill_diagonal(matrix, np.inf)
    nearest = matrix.min(axis=1)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        return np.array([])
    cutoff = 1.25 * float(np.median(finite))
    upper = matrix[np.triu_indices(n, k=1)]
    bonds = upper[(upper > 0.3) & (upper <= cutoff)]
    return bonds


def kabsch(mobile: np.ndarray, target: np.ndarray):
    """Return the rigid map that takes mobile onto target."""
    centroid_m = mobile.mean(axis=0)
    centroid_t = target.mean(axis=0)
    covariance = (mobile - centroid_m).T @ (target - centroid_t)
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    if np.linalg.det(vt_mat.T @ u_mat.T) < 0:
        vt_mat = vt_mat.copy()
        vt_mat[-1, :] *= -1
    rotation = vt_mat.T @ u_mat.T
    return rotation, centroid_m, centroid_t


def apply_kabsch(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    rotation, centroid_m, centroid_t = kabsch(mobile, target)
    return (mobile - centroid_m) @ rotation.T + centroid_t


def assignment_rmsd(mobile: np.ndarray, target: np.ndarray):
    """RMSD of the optimal one-to-one match. Extra atoms on the longer side are left out."""
    distances = cdist(mobile, target)
    rows, cols = linear_sum_assignment(distances)
    delta = mobile[rows] - target[cols]
    rmsd = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    return rmsd, int(len(rows))


def icp(mobile: np.ndarray, target: np.ndarray, iters: int = 4) -> np.ndarray:
    current = mobile.copy()
    for _ in range(iters):
        nearest = cdist(current, target).argmin(axis=1)
        current = apply_kabsch(current, target[nearest])
    return current


def orientation_candidates(mobile: np.ndarray, target: np.ndarray):
    """Rigid orientations to try before ICP. Flat sheets get an in-plane sweep."""
    yield mobile
    for axis in range(3):
        mirrored = mobile.copy()
        mirrored[:, axis] *= -1.0
        yield mirrored

    # In-plane sweep matters for zigzag flakes, which are rotated relative to ASE.
    z_span = mobile[:, 2].max() - mobile[:, 2].min()
    xy_span = np.linalg.norm(mobile[:, :2].max(axis=0) - mobile[:, :2].min(axis=0))
    if z_span < 0.5 * max(xy_span, 1.0):
        step = 15 if len(mobile) <= 400 else 30
        for mirror in (1.0, -1.0):
            base = mobile.copy()
            base[:, 0] *= mirror
            for angle in np.deg2rad(np.arange(0, 180, step)):
                c, s = np.cos(angle), np.sin(angle)
                rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                yield base @ rotation.T


def lattice_rmsd(positions: np.ndarray, primitive: Atoms) -> float:
    """RMSD of a point set to the ASE crystal.

    Translation is removed by measuring every atom relative to the first.
    A site matches if it lands on any ASE basis atom, modulo the primitive cell.
    Flat sheets also try in-plane rotations and a mirror, so a zigzag flake is
    not penalized for its orientation. Cell skew (60° vs 120°) does not matter.
    """
    points = np.asarray(positions, dtype=float)
    cell = np.asarray(primitive.cell.array, dtype=float)
    inverse = np.linalg.inv(cell)
    basis = np.asarray(primitive.positions, dtype=float) @ inverse

    def score(coords: np.ndarray) -> float:
        relative = (coords - coords[0]) @ inverse
        best = None
        for origin in basis:
            error = []
            for fractional in relative:
                delta = fractional - (basis - origin)
                delta -= np.round(delta)
                cartesian = delta @ cell
                error.append(np.sum(cartesian * cartesian, axis=1).min())
            rmsd = float(np.sqrt(np.mean(error)))
            if best is None or rmsd < best:
                best = rmsd
        return best

    span = points.max(axis=0) - points.min(axis=0)
    flat = span[2] < max(span[0], span[1], 1.0)
    angles = np.deg2rad(np.arange(0, 180, 15)) if flat else [0.0]
    best = None
    for mirror in (1.0, -1.0):
        mirrored = points.copy()
        mirrored[:, 0] *= mirror
        for angle in angles:
            cosine, sine = np.cos(angle), np.sin(angle)
            rotation = np.array([
                [cosine, -sine, 0.0],
                [sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ])
            rmsd = score(mirrored @ rotation.T)
            if best is None or rmsd < best:
                best = rmsd
            if best < 1e-8:
                return best
    return best


def rmsd_against(candidate: Atoms, reference: Atoms):
    """RMSD in Å of candidate atoms against the ASE reference after rigid alignment."""
    mobile = positions_of(candidate)
    target = positions_of(reference)
    mobile = mobile - mobile.mean(axis=0)
    target = target - target.mean(axis=0)
    if len(mobile) == 0 or len(target) == 0:
        return None, 0

    best_rmsd = None
    best_matched = 0
    for oriented in orientation_candidates(mobile, target):
        aligned = icp(oriented, target)
        rmsd, matched = assignment_rmsd(aligned, target)
        if best_rmsd is None or rmsd < best_rmsd:
            best_rmsd = rmsd
            best_matched = matched
        if best_rmsd < 1e-6:
            break
    return best_rmsd, best_matched


def ideal_bond_error(bonds: np.ndarray, ideal: float | None):
    if ideal is None or bonds.size == 0:
        return None
    return float(np.mean(np.abs(bonds - ideal)))


def compare_case(
    task,
    reference,
    parameters,
    build_gam,
    build_ase,
    ideal_bond,
    periodic_bonds=False,
    primitive=None,
    equalize_atom_counts=False,
):
    gam_atoms, gam_ms = median_time_ms(build_gam)

    def build_equal_ase():
        atoms = build_ase()
        if len(atoms) < len(gam_atoms):
            raise ValueError(
                f"ASE reference has {len(atoms)} atoms, fewer than pygamlab's {len(gam_atoms)}"
            )
        if len(atoms) == len(gam_atoms):
            return atoms
        center = atoms.positions.mean(axis=0)
        distances = np.linalg.norm(atoms.positions - center, axis=1)
        return atoms[np.argsort(distances)[:len(gam_atoms)]]

    timed_ase_builder = build_equal_ase if equalize_atom_counts else build_ase
    ase_atoms, ase_ms = median_time_ms(timed_ase_builder)
    if not periodic_bonds:
        gam_atoms = gam_atoms.copy()
        ase_atoms = ase_atoms.copy()
        gam_atoms.set_pbc(False)
        ase_atoms.set_pbc(False)

    gam_bonds = neighbor_bonds(gam_atoms)
    ase_bonds = neighbor_bonds(ase_atoms)
    gam_mean = float(gam_bonds.mean()) if gam_bonds.size else None
    ase_mean = float(ase_bonds.mean()) if ase_bonds.size else None
    if gam_mean is None or ase_mean is None:
        bond_error = None
    else:
        bond_error = abs(gam_mean - ase_mean)

    if primitive is None:
        rmsd, matched = rmsd_against(gam_atoms, ase_atoms)
    else:
        rmsd = lattice_rmsd(positions_of(gam_atoms), primitive())
        matched = len(gam_atoms)
    record = {
        "task": task,
        "reference": reference,
        "parameters": parameters,
        "n_atoms_pygamlab": int(len(gam_atoms)),
        "n_atoms_ase": int(len(ase_atoms)),
        "matched_atoms": int(matched),
        "rmsd_A": None if rmsd is None else round(rmsd, 6),
        "bond_length_error_A": None if bond_error is None else round(bond_error, 6),
        "mean_bond_pygamlab_A": None if gam_mean is None else round(gam_mean, 6),
        "mean_bond_ase_A": None if ase_mean is None else round(ase_mean, 6),
        "ideal_bond_A": ideal_bond,
        "bond_mae_vs_ideal_pygamlab_A": _round(ideal_bond_error(gam_bonds, ideal_bond)),
        "bond_mae_vs_ideal_ase_A": _round(ideal_bond_error(ase_bonds, ideal_bond)),
        "time_pygamlab_ms": round(gam_ms, 3),
        "time_ase_ms": round(ase_ms, 3),
    }
    return record, gam_atoms, ase_atoms


def _round(value):
    if value is None:
        return None
    return round(float(value), 6)


def save_xyz(task: str, gam_atoms: Atoms, ase_atoms: Atoms):
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in task)
    slug = "_".join(part for part in slug.split("_") if part)
    for atoms, suffix in ((gam_atoms, "pygamlab"), (ase_atoms, "ase")):
        clean = atoms.copy()
        clean.info.clear()
        write(XYZ_DIR / f"{slug}_{suffix}.xyz", clean, format="xyz")


def graphene_cases():
    ideal = GRAPHENE_A / sqrt(3.0)
    cases = []
    for nx, ny in ((2, 2), (4, 4), (8, 8), (16, 16)):
        cases.append(dict(
            task="graphene_supercell",
            reference=f"graphene_unit_cell(lattice_constant={GRAPHENE_A})  # supercell {nx}x{ny}",
            parameters=f"a={GRAPHENE_A} A, size={nx}x{ny}",
            build_gam=lambda nx=nx, ny=ny: pygamlab_graphene_supercell(nx, ny),
            build_ase=lambda nx=nx, ny=ny: ase_graphene_supercell(nx, ny),
            ideal_bond=ideal,
            periodic_bonds=True,
            primitive=lambda: ase_graphene_supercell(1, 1),
        ))
    for edge in ("armchair", "zigzag"):
        for width in (8.0, 12.0, 20.0):
            def build_gam(width=width, edge=edge):
                return gam_to_atoms(Graphene(
                    lattice_constant=GRAPHENE_A,
                    width=width,
                    length=width,
                    edge_type=edge,
                    rotation=0.0,
                ))

            atom_count = len(build_gam())
            ase_side = int(np.ceil(np.sqrt(atom_count / 2.0)))

            def build_ase(ase_side=ase_side):
                return ase_graphene_supercell(ase_side, ase_side)

            cases.append(dict(
                task="graphene_flake",
                reference=(
                    f"Graphene(lattice_constant={GRAPHENE_A}, width={width}, "
                    f"length={width}, edge_type='{edge}')"
                ),
                parameters=f"a={GRAPHENE_A} A, {edge}, width=length={width} A",
                build_gam=build_gam,
                build_ase=build_ase,
                ideal_bond=ideal,
                periodic_bonds=False,
                primitive=lambda: ase_graphene_supercell(1, 1),
                equalize_atom_counts=True,
            ))
    return cases


def silicene_cases():
    in_plane = SILICENE_A / sqrt(3.0)
    ideal = float(np.hypot(in_plane, SILICENE_BUCKLING))
    cases = [dict(
        task="silicene_supercell",
        reference=(
            f"silicene_unit_cell(lattice_constant={SILICENE_A})  "
            f"# buckling_height={SILICENE_BUCKLING}, supercell 4x4"
        ),
        parameters=f"a={SILICENE_A} A, buckling={SILICENE_BUCKLING} A, size=4x4",
        build_gam=lambda: pygamlab_silicene_supercell(4, 4),
        build_ase=lambda: ase_silicene_supercell(4, 4),
        ideal_bond=ideal,
        periodic_bonds=True,
        primitive=lambda: ase_silicene_supercell(1, 1),
    )]

    def build_flake():
        return gam_to_atoms(Silicene(
            lattice_constant=SILICENE_A,
            width=12.0,
            length=12.0,
            buckling_height=SILICENE_BUCKLING,
            rotation=0.0,
        ))

    atom_count = len(build_flake())
    ase_side = int(np.ceil(np.sqrt(atom_count / 2.0)))
    cases.append(dict(
        task="silicene_flake",
        reference=(
            f"Silicene(lattice_constant={SILICENE_A}, width=12.0, length=12.0, "
            f"buckling_height={SILICENE_BUCKLING})"
        ),
        parameters=f"a={SILICENE_A} A, buckling={SILICENE_BUCKLING} A, width=length=12 A",
        build_gam=build_flake,
        build_ase=lambda: ase_silicene_supercell(ase_side, ase_side),
        ideal_bond=ideal,
        periodic_bonds=False,
        primitive=lambda: ase_silicene_supercell(1, 1),
        equalize_atom_counts=True,
    ))
    return cases


def nanotube_cases():
    cases = []
    for n, m, repeats in ((6, 0, 2), (5, 5, 2), (8, 2, 1)):
        period = nanotube_period(n, m, NANOTUBE_BOND)
        length_A = (repeats + 0.5) * period

        def build_gam(n=n, m=m, length_A=length_A):
            return gam_to_atoms(Nanotube_Generator(n=n, m=m, length=length_A, atom_type="C"))

        def build_ase(n=n, m=m, repeats=repeats):
            atoms = nanotube(n, m, length=repeats, bond=NANOTUBE_BOND, symbol="C")
            atoms.set_pbc(False)
            return atoms

        cases.append(dict(
            task="nanotube",
            reference=(
                f"Nanotube_Generator(n={n}, m={m}, length={length_A:.4f}, atom_type='C')"
            ),
            parameters=f"({n},{m}), cells={repeats}, bond={NANOTUBE_BOND} A",
            build_gam=build_gam,
            build_ase=build_ase,
            ideal_bond=NANOTUBE_BOND,
            periodic_bonds=False,
        ))
    return cases


def nanoparticle_cases():
    ideal = AU_A / sqrt(2.0)
    radius = 1.5 * 10.0 / 2.0

    def build_sphere():
        particle = Nanoparticle_Generator(
            element="Au",
            size_nm=1.5,
            shape="sphere",
            crystal_structure="FCC",
            lattice_constant=AU_A,
        )
        return gam_to_atoms(particle)

    def build_octahedron():
        particle = Nanoparticle_Generator(
            element="Au",
            size_nm=1.5,
            shape="octahedron",
            crystal_structure="FCC",
            lattice_constant=AU_A,
        )
        return gam_to_atoms(particle)

    return [
        dict(
            task="nanoparticle_sphere",
            reference=(
                f"Nanoparticle_Generator(element='Au', size_nm=1.5, shape='sphere', "
                f"crystal_structure='FCC', lattice_constant={AU_A})"
            ),
            parameters=f"Au FCC a={AU_A} A, radius={radius} A",
            build_gam=build_sphere,
            build_ase=lambda: ase_fcc_sphere("Au", AU_A, radius),
            ideal_bond=ideal,
            periodic_bonds=False,
        ),
        dict(
            task="nanoparticle_octahedron",
            reference=(
                f"Nanoparticle_Generator(element='Au', size_nm=1.5, shape='octahedron', "
                f"crystal_structure='FCC', lattice_constant={AU_A})"
            ),
            parameters=f"Au FCC a={AU_A} A, pygamlab size_nm=1.5, ASE length=3",
            build_gam=build_octahedron,
            build_ase=lambda: Octahedron("Au", length=3, latticeconstant=AU_A),
            ideal_bond=ideal,
            periodic_bonds=False,
        ),
    ]


def write_csv(records):
    path = OUT_DIR / "results.csv"
    fieldnames = list(records[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


def write_summary(records):
    path = OUT_DIR / "summary.txt"
    header = (
        f"{'task':<24} {'n_gam':>6} {'n_ase':>6} {'RMSD_A':>10} "
        f"{'bond_err_A':>11} {'t_gam_ms':>10} {'t_ase_ms':>10}  reference"
    )
    lines = [
        "pygamlab GAM_architectures vs ASE",
        f"pygamlab {pygamlab.__version__}    ASE {ase.__version__}",
        "",
        "Accuracy: RMSD of pygamlab atoms to the ASE crystal (2D sheets), or rigid RMSD",
        "for nanotubes and nanoparticles. Bond-length error is |mean NN bond difference|.",
        "Performance: median build time after one warmup, and atom count.",
        "",
        header,
        "-" * len(header),
    ]
    for row in records:
        rmsd = "n/a" if row["rmsd_A"] is None else f"{row['rmsd_A']:.4f}"
        bond = "n/a" if row["bond_length_error_A"] is None else f"{row['bond_length_error_A']:.4f}"
        lines.append(
            f"{row['task']:<24} {row['n_atoms_pygamlab']:6d} {row['n_atoms_ase']:6d} "
            f"{rmsd:>10} {bond:>11} {row['time_pygamlab_ms']:10.3f} "
            f"{row['time_ase_ms']:10.3f}  {row['reference']}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    XYZ_DIR.mkdir(parents=True, exist_ok=True)
    for path in XYZ_DIR.glob("*.xyz"):
        path.unlink()

    cases = []
    cases.extend(graphene_cases())
    cases.extend(silicene_cases())
    cases.extend(nanotube_cases())
    cases.extend(nanoparticle_cases())

    records = []
    for case in cases:
        print(f"running {case['task']}: {case['parameters']}")
        record, gam_atoms, ase_atoms = compare_case(**case)
        records.append(record)
        save_xyz(f"{case['task']}_{case['parameters']}", gam_atoms, ase_atoms)
        print(
            f"  atoms {record['n_atoms_pygamlab']}/{record['n_atoms_ase']}  "
            f"RMSD {record['rmsd_A']} A  bond error {record['bond_length_error_A']} A  "
            f"time {record['time_pygamlab_ms']} / {record['time_ase_ms']} ms"
        )

    payload = {
        "pygamlab_version": pygamlab.__version__,
        "ase_version": ase.__version__,
        "accuracy_metrics": ["rmsd_A", "bond_length_error_A"],
        "performance_metrics": ["time_pygamlab_ms", "time_ase_ms", "n_atoms_pygamlab", "n_atoms_ase"],
        "results": records,
    }
    json_path = OUT_DIR / "results.json"
    json_path.write_text(json.dumps(payload, indent=2))
    csv_path = write_csv(records)
    summary_path = write_summary(records)
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print(f"wrote xyz files in {XYZ_DIR}")


if __name__ == "__main__":
    main()
