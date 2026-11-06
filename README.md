# kicad2ems-palace

[Palace](https://github.com/awslabs/palace) (AWS's 3D finite-element electromagnetics solver) as a kicad2ems
solver: `kicad2ems board.toml --solvers palace`. **Experimental.**

The board is meshed in tetrahedra by gmsh (`model.py`): the dielectric slabs inside the board outline and an
air box around them, copper as PEC sheets of zero thickness on each copper layer, vias as PEC holes, lumped
ports as rectangles (across the trace from the reference copper, or across a gap). Palace solves it in the
frequency domain, every port excited in turn, with its adaptive frequency sweep (`config.py`).

Not modelled yet: lumped parts, conductor loss and roughness, edge walls, the antenna mode's far field.

## Palace itself

Palace is not on PyPI. `run.py` finds it as

- `K2E_PALACE`: its `palace` executable (the MPI wrapper), or `palace` on `PATH`, e.g. a
  [Spack build](https://awslabs.github.io/palace/stable/install/) (`spack install palace`);
- `K2E_PALACE_ROOTFS` (default `~/opt/palace/rootfs`): an unpacked Palace container image, run with rootless
  podman with that directory as its root, no network, as the user. The development machine uses a community
  Apptainer image of Palace 0.18.1 (gds2palace's `palace_0181`) unpacked this way, since there is no root
  for Apptainer.

## Settings

| Key (`[solver.palace]`) | Default | |
|---|---|---|
| `order` | 2 | element order |
| `tol` | 1e-3 | adaptive sweep tolerance (0: every point solved) |
| `feature` | 1.0 | cells across the smallest feature at the copper edges |
| `wave` | 8.0 | cells per wavelength at fmax away from the copper |
| `margin` | λ/4 at fmax | air around the board, mm |
| `np` | 8 | MPI processes |
