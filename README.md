# kicad2ems-fem

The finite-element solvers of [kicad2ems](https://github.com/lowprocess/kicad2ems), as kicad2ems-fdtd hosts its
FDTD ones: each an external program behind its own interface, sharing the board's geometry (`geometry.py`) and
gmsh meshing. Neither program is a Python dependency; each is found when it runs.

| Solver | For | Interface |
|---|---|---|
| [Palace](https://github.com/awslabs/palace) (AWS) | S-parameters, frequency domain | `kicad2ems board.toml --solvers palace` (`palace/`) |
| [Elmer](https://www.elmerfem.org) (CSC) | DC IR drop of a net's copper, 3D | `python -m kicad2ems_fem.elmer.dc` (`elmer/`) |

Both are **experimental**.

## Elmer: DC IR drop

```sh
python -m kicad2ems_fem.elmer.dc board.kicad_pcb --net VHB --source L3 --load Q1=0.67 --load Q2=0.67 --out dc_vhb
```

- **Model:** the net's copper is extruded through its thickness, outer layers outward. Each via is a cylinder
  whose conductivity gives its plated annulus's conductance (`--plating`).
- **Terminals:** the source part's pads on the net are held at 0 V. Each load draws its current as one
  equipotential terminal: Elmer's `Potential Constant` with a flux.
- **Several `--net`** are one conductor. Series parts (sense resistors, beads) are not modelled: rail trees
  and the return are pcbqs.pdn's job, and this checks their copper in 3D.
- **Output:** `<out>/dc.json` (each load's voltage and drop, the copper's loss, the mesh) and Elmer's case in
  `<out>/elmer`.
- **Mesh:** `--h` (default 0.25 mm) in the copper, `--h-fine` (0.05 mm) at vias and pad edges. Refine until the
  drop stops moving.

Against pcbqs's 2D DC solve this agreed within 0.2% on the wpthb VHB rail and 1.5% on Jetson +1V15
(pcbqs/validation). wpthb VHB, L3 to the two high-side FETs, at h 0.1 / 0.02 mm (1.1 M tetrahedra): Q1 0.3835 mV,
Q2 0.6179 mV.

Elmer is found as `K2E_ELMER` (the folder with `ElmerSolver` and `ElmerGrid`) or on `PATH`. The mesh is solved
with Elmer's direct solver (Umfpack), because the iterative ones diverged on thin copper. A million tetrahedra
take about 3.5 GB.

## Palace: S-parameters

The board is meshed in tetrahedra by gmsh (`palace/model.py`):
- the dielectric slabs inside the board outline, in an air box;
- copper as PEC sheets of zero thickness on each copper layer;
- vias as PEC holes;
- lumped ports as rectangles, across the trace from the reference copper or across a gap.

Palace solves it in the frequency domain, exciting every port in turn, with its adaptive frequency sweep
(`palace/config.py`).

Not modelled yet: lumped parts, conductor loss and roughness, edge walls, the antenna mode's far field.

### Palace itself

Palace is not on PyPI. `run.py` finds it as

- `K2E_PALACE`: its `palace` executable (the MPI wrapper), or `palace` on `PATH`, e.g. a
  [Spack build](https://awslabs.github.io/palace/stable/install/) (`spack install palace`);
- `K2E_PALACE_ROOTFS` (default `~/opt/palace/rootfs`): an unpacked Palace container image, run with rootless
  podman with that directory as its root, no network, as the user. The development machine uses a community
  Apptainer image of Palace 0.18.1 (gds2palace's `palace_0181`) unpacked this way, since there is no root
  for Apptainer.

### Settings

| Key (`[solver.palace]`) | Default | |
|---|---|---|
| `order` | 2 | element order |
| `tol` | 1e-3 | adaptive sweep tolerance (0: every point solved) |
| `feature` | 1.0 | cells across the smallest feature at the copper edges |
| `wave` | 8.0 | cells per wavelength at fmax away from the copper |
| `margin` | λ/4 at fmax | air around the board, mm |
| `np` | 8 | MPI processes |
