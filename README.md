# kicad2ems-fem

The finite-element solvers of [kicad2ems](https://github.com/lowprocess/kicad2ems), as kicad2ems-fdtd hosts its
FDTD ones: each an external program behind its own interface, sharing the board's geometry (`geometry.py`) and
gmsh meshing. Neither program is a Python dependency; each is found when it runs.

| Solver | For | Interface |
|---|---|---|
| [Palace](https://github.com/awslabs/palace) (AWS) | S-parameters, frequency domain; antennas | `kicad2ems board.toml --solvers palace` (`palace/`) |
| Palace, electrostatic | Capacitance between nets on the real stack | `python -m kicad2ems_fem.palace.cap` (`palace/cap.py`) |
| [Elmer](https://www.elmerfem.org) (CSC) | DC IR drop of a net's copper, 3D | `python -m kicad2ems_fem.elmer.dc` (`elmer/`) |
| Elmer, heat | The board's steady temperature: parts' heat, a rail's own, a circuit run's | `python -m kicad2ems_fem.elmer.thermal` |
| Elmer, magnetostatic | Wound coils beside ferrite: inductance and coupling (the study's *Coils and ferrite*) | `python -m kicad2ems_fem.elmer.mag` |
| Elmer, electrostatic | Capacitance between nets: Palace's model, a second solver | `python -m kicad2ems_fem.elmer.cap` |

All are **experimental**. Each is a second opinion on a faster solver of the tool, by a different method:

| Area of the tool | Fast solver | Here |
|---|---|---|
| S-parameters of a routed board, with its lumped parts | mom25d, the line model, FDTD | Palace, driven |
| Antenna: match, gain, efficiency, patterns | gfdtd, openEMS, mom25d | Palace, driven, with its far field |
| Capacitance between nets (pcbqs's PEEC charge) | pcbqs | Palace, electrostatic: the finite FR-4 and the copper's thickness; Elmer on the same model |
| DC IR drop of a rail | pcbqs.pdn | Elmer |
| The power circuit's losses (pcbqs's ngspice run) | none | Elmer: the temperature they give the board |
| A wireless-power coil pair's L and k (the power circuit's coil model) | none: a datasheet figure | Elmer: the coils with their ferrite |

The first four are minutes where the fast solver is seconds: for checking a result that matters, not for tuning.
The last two have no fast solver in the tool: they supply what it otherwise takes as given.

## As engines of a kicad2ems study

kicad2ems lists what can be asked of a board as questions, each with the engines that answer it
(`kicad2ems.study`, `kicad2ems --check`). Five of the tools here register as engines (`analyses.py`, the
`kicad2ems.analyses` entry points), beside the faster ones:

| Question | Engine here | Beside |
|---|---|---|
| a rail's DC drop | `elmer_dc` | pcbqs.pdn |
| the board's temperature | `elmer_thermal` | pcbqs.thermal |
| capacitance between nets | `elmer_cap`, `palace_cap` | |
| coils and ferrite (their inductances and coupling) | `elmer_mag` | |

A front end builds each one's settings from the tool's own command-line options, runs it as its command and shows
its report. The nets and the source part can be picked on the board there (the options' `pick`, kicad2ems API 2.11). Asked for the temperature without any heat typed, `elmer_thermal` takes the losses of the latest
circuit run's heaviest case. Each result says which tool wrote it, so a `thermal.json` of Elmer's is not taken for
pcbqs's.

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
- **Mesh:** the layered mesh the thermal tool solves on (`--mesh wedge`, the default): the board's triangles,
  cropped to the net's copper and 1 mm round it, swept through the stack as wedges, the net's copper and via
  barrels its bodies; `--h` (default 0.25 mm) the triangles' size. `--mesh tet` meshes the copper in 3D with gmsh
  instead (`--h-fine`, 0.05 mm, at vias and pad edges). Refine until the drop stops moving.

Against pcbqs's 2D DC solve this agreed within 0.2% on the wpthb VHB rail and 1.5% on Jetson +1V15
(pcbqs/validation). wpthb VHB, L3 to the two high-side FETs, at h 0.1 / 0.02 mm (1.1 M tetrahedra): Q1 0.3835 mV,
Q2 0.6179 mV; on the layered mesh at its default (39,681 wedges, 0.8 s and 120 MB, R-D23) 0.3842 and 0.6194 mV.

Elmer is found as `K2E_ELMER` (the folder with `ElmerSolver` and `ElmerGrid`), the folder kicad2ems's installer
recorded (`elmer` in `kicad2ems-site.toml`), or on `PATH`; a serial build with OpenMP is enough for every tool
here. kicad2ems's `install.py` finds one, takes a folder you give (`--elmer PATH`) or builds one from source
(`--elmer build`, kicad2ems's `tools/build_elmer.sh`: a few minutes). The mesh is solved with Elmer's direct solver (Umfpack), because the
iterative ones diverged on thin copper. A million tetrahedra take about 3.5 GB (`--mesh tet`).

## Palace: S-parameters

The board is meshed in tetrahedra by gmsh (`palace/model.py`):
- the dielectric slabs inside the board outline, in an air box with an absorbing boundary (or metal: `box`);
- copper as PEC sheets of zero thickness on each copper layer (`loss`: lossy sheets);
- vias as PEC holes;
- lumped ports as rectangles, across the trace from the reference copper or across a gap;
- lumped R / L / C parts (`pos`, `[parts]`) as sheets across their pads' gap, one per series element, each with
  a terminal onto its pads as wide as the part. Without the terminals a rounded pad feeds the sheet over part
  of its width and the part reads 30% high;
- edge walls (`edge_walls`) as vertical PEC sheets between the planes, the keepout in from the board's edge.

The mesh follows the copper edges (`feature`), the wavelength in the dielectric inside the board and the
free-space wavelength in the air (`wave`). The ground does not ask for the fine size: a ground plane's own
edge, a via on a ground net (a hexagonal hole of the barrel's cross-section) and an edge wall take the size the
mesh has grown to from the signal copper, the ports and the parts. On `sma_launch` at one cell a feature that
was 64,000 tetrahedra instead of 262,000, with S within 0.006 of the fine mesh's (0.03 at a resonance);
`fine_ground` puts it back.

`feature` is 4 cells across the smallest feature because a line's impedance needs it. With one cell across
`sma_launch`'s 0.2 mm dielectric the line reads 5% low, which there happens to be 50 Ω: S11 is −57 dB at 1 GHz where
the converged answer is −27 dB, and the return-loss null of the launch is missing. S21 does not move.

| `feature` on `sma_launch` | Edge size, mm | Tetrahedra | S11 at 1.0 / 2.4 / 3.2 / 5.0 GHz, dB | Line, Ω |
|---|---|---|---|---|
| 1 | 0.2 | 64,000 | −56.7 / −34.7 / −32.8 / −23.6 | 50.4 |
| 2 | 0.1 | 108,000 | −34.9 / −39.5 / −35.0 / −24.5 | 51.0 |
| 4 | 0.05 | 190,000 | −29.1 / −32.0 / −39.6 / −24.3 | 52.1 |
| 8 | 0.025 | 327,000 | −27.4 / −29.7 / −41.2 / −24.0 | 52.6 |

openEMS has the line at 52.9 Ω (S11 −26.1 dB at 1 GHz). The finer mesh costs 2 to 7 times the time and 3 to 4
times the memory on the examples below; `feature = 1` is enough for insertion loss, a filter's corner or a
first look.

Palace solves it in the frequency domain, exciting every port in turn, with its adaptive frequency sweep
(`palace/config.py`). Two of its linear-solver defaults are changed, both timed on these examples:
- the multigrid's coarse level is solved with the true complex matrix (`ComplexCoarseSolve`), not a real
  stand-in: 7 to 10 GMRES iterations a frequency instead of 46 to 152 on the antennas, 3.5 times less solve time,
  the same S to 5 digits, a quarter more memory;
- its error indicator, which only its mesh refinement uses, is found loosely (`EstimatorTol`): at the default it
  was 40% of the time on `sma_launch`.

Not modelled: copper thickness, roughness, surface finish, dispersion (er and tand at one value).

Checked against the tool's other solvers on its examples (i7-13700K, 8 processes):

| Example | Palace | Other solver | Time |
|---|---|---|---|
| `demo_lowpass`: two 0402 inductors and a capacitor on a microstrip, to 3 GHz | series arms 10.71 / 10.85 nH, shunt 9.04 pF; −3 dB at 0.77 GHz, notch 2.11 GHz | mom25d: 10.80 / 10.92 nH, 9.02 pF; 0.77 GHz, 2.07 GHz | 143,000 tetrahedra, 30 min, 19 GB (mom25d 2 s); at `feature = 1` 34,000, 4.5 min, 4.9 GB and 10.64 / 10.75 nH, 9.15 pF |
| A series part in a 50 Ω microstrip, 0.1 GHz (`tests/test_palace_run.py`) | 20 Ω + 8.2 nH: 20.0 Ω, 10.8 nH with the line's 2.6 nH; 10 pF: 10.0 pF | the circuit | 18,000 tetrahedra, 12 s |
| `sma_launch`: an end-launch SMA jack onto a 50 Ω line on a 4-layer board, edge walls, to 6 GHz | S21 −0.09 / −0.26 / −0.55 dB at 1 / 3 / 6 GHz; S11 −29.1 dB at 1 GHz, its null at 2.98 GHz, under −24 dB to 6 GHz | S21: mom25d −0.12 / −0.28 / −0.55 dB, openEMS −0.32 / −0.30 / −0.38 dB. S11: mom25d −22.1 dB, null 3.35 GHz; openEMS −26.1 dB, null 3.11 GHz | 190,000 tetrahedra, 7 min, 18 GB |

Without its edge walls `sma_launch` has two dips near 5 GHz (−5 dB at 5.06 GHz): the plane pair beyond the
last row of vias resonating, the field in the two corner pockets beside the connector. The walls stand 0.5 mm in
from the outline, inside the board, and an earlier model builder dropped any sheet that was not the face of a
volume, so the walls with them. `tests/test_palace.py` holds one now, and a run's record (`*_run.json`) says
whether the model has its walls (`edge_walls`).

Palace can save the fields too: `Solver.Driven.Save` (frequencies) or `SaveStep` writes ParaView files with E,
B, the energy densities and the Poynting vector in the volume, and the surface current and charge on the
boundaries (86 MB a frequency on a 61,000-tetrahedron model). The plugin does not ask for them yet, so the
tool's field-dump settings do nothing here.

### Antenna mode

`mode = "antenna"` in the board's TOML: the finite board in air, the absorbing faces half a wavelength (band
centre) away. A second run solves 12 frequencies with the antenna's port alone driven (the first of `excite`,
else the first port) and has Palace integrate the far field (`palace/antenna.py`). The result is the other
solvers': match, directivity, efficiency, gain, polarisation, the pattern cuts and the sphere, with the same
plots, JSON and report section. A phased drive (`drive`) is not modelled.

| Example | Palace | gfdtd | mom25d |
|---|---|---|---|
| `antenna_dipole`, 1.5 to 3.5 GHz | 2.430 GHz, S11 −20.3 dB, Zin 62.1 + j7.3 Ω, band 2.260 to 2.640 GHz; D 2.12 dBi, efficiency 98%, gain 2.02 dBi | 2.435 GHz, −21.0 dB, 63.0 + j6.8 Ω, 2.260 to 2.645 GHz; 2.10 dBi, 95%, 1.89 dBi | (no ground plane) |
| `antenna_patch`, 2 to 3 GHz | 2.4375 GHz, −37.6 dB, band 2.410 to 2.467 GHz; D 7.03 dBi, efficiency 41%, gain 3.16 dBi | 2.4375 GHz, −34.6 dB, 2.408 to 2.470 GHz; 7.02 dBi, 39%, 2.89 dBi | 2.450 GHz, −30.5 dB, 2.425 to 2.480 GHz; 6.33 dBi, 36%, 1.94 dBi |

The dipole takes 3.8 min (86,000 tetrahedra, 8.2 GB) and the patch 4.0 min (94,000, 8.6 GB); gfdtd takes
seconds. At `feature = 1` they took 71 s and 84 s, with the patch's resonance 0.7% lower (2.420 GHz). The
efficiency reads up to 0.5% over 100% on the dipole away from resonance, and the boundary's order is not why: the
second-order one (`abc = 2`) moves the match a little (S11 at resonance −20.5 dB to −22.7 dB at `feature = 1`)
and the efficiency not at all (100.4% at 2.9 GHz either way). The excess is in the far field's integration over
the box's faces.

### In a can: resonances

`box = "pec"` closes the air box in metal: a can on all six sides, `margin` its distance from the board.
`modes = N` adds an eigenmode run after the driven one, on the same mesh: the model's N resonances above
`fmin`, each one's frequency and Q, with the ports as their resistors so the Q is the loaded one, and each
port's share (Q_ext, Palace's `port-Q.csv`). They are logged, written as `<name>_palace_modes.json` and shown
as a table in the report's Palace section. In an open box (`box = "absorbing"`) the Q includes what radiates.
A mode with Q under 5 is a feed line damped by its port's resistor, not a resonance, and is marked so.

`rf_shielded` (kicad2ems's example: a half-wave resonator side-coupled to two lines in a can 4 mm round and
above the board), `box = "pec"`, `margin = 4`, `modes = 3`, 118,000 tetrahedra, 5.7 min for both runs, 10.7 GB:

| | Palace | other solvers |
|---|---|---|
| driven: S21 peak | 4.657 GHz, −4.1 dB; −3 dB band 4.644 to 4.672 GHz | EMerge (converged) 4.621 GHz; openEMS (4× mesh) and gfdtd 4.625 GHz; mom25d, without the can, 4.648 GHz |
| mode 1, the resonator | 4.6574 GHz, Q 157.5 (f/Q 29.6 MHz); Q_ext P1 457, P2 554 | |
| modes 2 and 3 | 5.36 and 6.95 GHz, Q 1.1 and 2.1: the feed lines, each damped by its port | |
| mode 4, the can | 7.115 GHz, Q 559 | |

The eigenmode's frequency is the driven peak's to 5 digits. Both sit 0.8 % above EMerge's 4.621 GHz, and the
mesh is not why: Palace's own refinement (`amr`) from `feature = 1` goes 4.626, 4.607, 4.635, 4.651 GHz (165,000
to 547,000 unknowns; 417 s, 7.0 GB), towards the default mesh's 4.657 GHz (813,000 unknowns; 195 s, 10 GB for
the eigenmode run). openEMS's refinement climbs the same way (4.555, 4.588, 4.626 GHz at 1x, 2x and 4x its
default mesh) and has not settled; mom25d, without the can, has 4.648 GHz. Which is right is open.

Refinement (`amr = N`) applies to the eigenmode run too, refining on the modes' error, and stops at `amr_max`
unknowns: by default what 70 % of the memory available holds at Palace's 12 MB a thousand unknowns (a pass can
double the mesh). On `rf_shielded` it took twice the time of the default mesh for a slightly coarser answer, so
the default stays `feature = 4`; `amr` is for checking that an answer has settled.

### A rail's impedance

`python -m kicad2ems_fem.palace.pdn --pdn <a pcbqs PDN run> --out pdn_palace`, and in a study the engine
*Rail impedance (Palace)* of the power-integrity question, which takes the board's newest pcbqs PDN run: its rail
solved again in 3D. Each part on the rail is a lumped port across its copper's gap between rail and ground at the
part (a pour's edge to the next, or pad to pad; a series part's between its pads), so the copper from the pads to
the planes is Palace's own; the bare copper's port matrix is closed after the solve by the run's parts (the decaps'
C, ESR and ESL or their vendor files, the regulator's R + L or its file), the loads left open. A part with no gap of
its own (a regulator's output pad whose nearest ground lies past a decap's) is left out, open as in pcbqs's bare
matrix. `pdn_palace.json` has each load's |Z| beside pcbqs's and, where the run wrote `--ports-touchstone`, the bare
matrix against pcbqs's. On pcbqs's buck example's output rail (two layers, the out and GND pours 0.4 mm apart on
F.Cu) Palace reads the bare rail's |Z| 0.68 of pcbqs's and the loaded J2's inductance 0.45 of it, steadily from 1 to
550 MHz: the ground pour beside the rail on its own layer, which pcbqs's plane pair (rail on one layer, ground on
another, vias between) leaves out. 42,000 tetrahedra, 36 s.

### A converter's input loop

`python -m kicad2ems_fem.palace.loop --mqs <a pcbqs near-field run> --out loop`, and in a study the engine
*Converter loop (Palace)* of the near field's question, which takes the board's newest pcbqs near-field run: the
loop pcbqs found (its converter, input net, ground, input capacitors and crop) solved again in 3D. The copper is
kicad2ems's board of the same KiCad file, cropped as pcbqs cropped it, as PEC sheets and holes; the input
capacitors are metal straps (pcbqs's copper inductance has the parts' ESL apart); the source a lumped port across
the converter's closest input and ground pads, its reference resistance near the loop's own |Z| (against 50 Ω, a
milliohm loop's S11 is within 1e-4 of −1 and Palace read 1.43 Ω of loss out of PEC). `loop.json` has L at each of
the run's harmonics beside pcbqs's, and both engines name them `loop L at F MHz`, so a study shows them side by
side.

On the Jetson baseboard's U69 (a SOT-563 buck from +5V, C35 and C76 at its input): Palace 0.219 to 0.198 nH from
0.6 to 12 MHz (123,000 tetrahedra at `hmin` 0.25 mm, 2.3 min, ~12 GB); pcbqs 0.420 to 0.338 nH. On the validation
suite's simple loops the two agree within 2% (`pcbqs/validation/mqs_palace.py`), so the 40% here is in how each
models this loop, which is open: zero-thickness PEC against 35 µm copper (1.7 skin depths at 12 MHz) in a loop
this small, the capacitors as straps in the copper plane against pcbqs's filaments, and pcbqs's source across all
the converter's input and ground pads against one port. The finest mesh (`hmin` 0.02 mm, the board's narrowest
gap) is 2.2 M tetrahedra, past this machine.

### Palace itself

Palace is not on PyPI. `run.py` finds it as

- `K2E_PALACE`: its `palace` executable (the MPI wrapper), then the one kicad2ems's installer recorded (`palace`
  in `kicad2ems-site.toml`), then `palace` on `PATH`. It runs with its own folder first on `PATH`, where a Spack
  view has its `mpirun`. kicad2ems's `install.py --palace spack` (`tools/build_palace.sh`) builds Palace's newest
  release with Spack as Palace's own CI does, into a prefix of its own: several hours and about 10 GB, never by
  default;
- `K2E_PALACE_ROOTFS`, then `palace_rootfs` in `kicad2ems-site.toml` (default `~/opt/palace/rootfs`): an unpacked
  Palace container image, run with rootless
  podman with that directory as its root, no network, as the user. The development machine uses a community
  Apptainer image of Palace 0.18.1 (gds2palace's `palace_0181`) unpacked this way, since there is no root
  for Apptainer.

The tools here were written against Palace 0.18.1. The Spack build is of the newest release, and 0.19.0 built
that way (72 packages, about an hour at 8 jobs, 2.3 GB) runs them too: kicad2ems's `rf_crosstalk` through the
plugin gives the crosstalk of mom25d with lumped ports and of the line model (NEXT -31.0 dB at 1 GHz, FEXT
-17.2 dB at 6 GHz; mom25d -31.7 and -17.5 dB), 103,000 tetrahedra in 6 minutes. `PALACE_TAG=v0.18.1` builds the
older one.

### Settings

| Key (`[solver.palace]`) | Default | |
|---|---|---|
| `order` | 2 | element order |
| `tol` | 1e-3 | adaptive sweep tolerance (0: every point solved) |
| `feature` | 4.0 | cells across the smallest feature at the copper edges; 1 is 2 to 7 times quicker and enough where the match is not the question |
| `wave` | 8.0 | cells per wavelength at fmax: in the dielectric inside the board, in free space in the air |
| `margin` | λ/4 at fmax; antenna mode λ/2 at the band centre | air around the board, mm |
| `np` | 8 | MPI processes |
| `fine_ground` | false | the feature size at the ground's own edges, ground vias (round) and edge walls too |
| `loss` | false | copper loss: the sheets as Palace's finite-thickness conductivity boundary. Good to ~10 MHz, high above (the sheet edges' singular current) |
| `amr`, `amr_tol`, `amr_max` | 0, 1e-2, from the memory | Palace's adaptive mesh refinement: its passes, the error to stop at and the unknowns it stops at (default: what 70% of the memory available holds) |
| `abc` | 1 | the absorbing boundary's order, 1 or 2 |
| `box` | absorbing | the air box's faces: `absorbing` (free space, first order) or `pec` (a metal can, `margin` from the board) |
| `modes` | 0 | after the driven run, an eigenmode run: the N resonances above `fmin`, their frequency, Q and each port's share |

## Elmer: temperature

```sh
python -m kicad2ems_fem.elmer.thermal board.kicad_pcb --heat Q1=0.3 --heat Q2=0.3@0.5 --out th
python -m kicad2ems_fem.elmer.thermal board.kicad_pcb --heat-from run/case.json --out th
python -m kicad2ems_fem.elmer.thermal board.kicad_pcb --net VHB --source L3 --load Q1=3 --load Q2=3 --out th
```

- **Model:** all the board's copper, its vias (barrels of their plated annulus's conductance) and the board
  material (0.8 W/m/K in plane, 0.3 through: `--k-board`), every outer face giving heat to the air at `--htc`
  (default 10 W/m²/K: still air with radiation near 60 °C) above `--ambient`.
- **Heat:** `--heat REF=WATTS` spreads a part's dissipation over the copper under its pads (`REF.PAD`: one pad).
  `WATTS@DUTY` is a part that dissipates for that fraction of the time: the mean heats it.
- **From the circuit:** `--heat-from` takes every part's dissipation from a pcbqs circuit run: its
  `case.json` has each part's `power_W`, its mean terminal power over the reported period. Any JSON
  `{REF: watts}` works too; `FILE@DUTY` is a converter that runs part of the time. Entries that are not parts on
  the board (the copper model's ports, the source, the coils) are listed and left out.
- **A rail's own heat:** `--net`, `--source`, `--load` as the DC tool. The copper's resistance follows its
  temperature (0.393 %/K), so the drops and the loss are the warm copper's; both are solved together.
- **Output:** `<out>/thermal.json` (the hottest point, each part's pad temperature, the drops warm and at ambient,
  the heat balance), `<out>/thermal.png` (both faces) and Elmer's case with a `.vtu` in `<out>/elmer`.
- **Mesh:** the board's outline triangulated with every copper edge in it, swept through the stack as wedges
  (`elmer/layered.py`): one element through each copper layer, where tetrahedra would need millions. `--h`
  (default 0.3 mm) in the copper with current and at pads, `--hmax` (2 mm) elsewhere.

**Heating from cold** (`--transient S`, seconds; parts' heat only, not a net's current). After the steady run,
the board heats from the ambient with the parts' heat on at 0 s: Elmer's BDF2 on pcbqs's own steps (40 a
decade, over 4 decades to S), each part's pads and the hottest point recorded each step, the board's own heat
capacity (copper 3.45, laminate 2.0 MJ/m³/K; not the parts' bodies'). `<out>/thermal-transient.json` (the
curves, each part's time to 63 % of its rise), `thermal-transient.png`, and the report's *Heating from cold*.

| Check | Result |
|---|---|
| A 10 mm square on a 0.2 mm board, heated over its whole top, against one lump (C / hA = 24.87 s) | 63 % at 24.81 s, the curve within 2% at 0.3, 1 and 3 time constants |
| wpthb, 0.3 W in each FET and 0.4 W in L3, to 2000 s, against pcbqs's grid (`pcbqs/validation/thermal_elmer.py --transient`) | 63 % at 82, 85, 90 s (Q1, L3, Q2); pcbqs 89, 88, 97 s (+3 to +9%), within 0.5 K at 200 s. 331,000 wedges, 15 min |

The wpthb run found pcbqs's grid counting an inner layer's copper as laminate too (its laminate share took the
layer's own thickness): 4.8% too much heat capacity, its parts 13% late, now 3 to 9% (pcbqs, R-E21). Elmer's
direct solver refactors at each of the 160 steps, ~5.5 s each there (keeping its factors, `Linear System
Refactorize = False`, changed nothing); an iterative one (CG, ILU0) took ten times as long on the board's thin
wedges.

It is the steady temperature for the mean power: right when the on and off times are short beside the seconds
a part's copper takes to warm. Left out: the parts' own bodies (all a part's heat enters the board, so pad
temperatures are upper bounds), solder mask, heat taken by connectors and wires, air flow other than as `--htc`,
and the losses the circuit run does not have (gate drivers, the controller, regulators).

| Check | Result |
|---|---|
| A 100 × 5 mm strip on a 0.2 mm board, 0.2 W at one end, against the fin equation | within 0.6% along it |
| A 50 × 5 mm strip carrying 10 A | 93 to 109 °C; drop 52.85 mV, 40.14 mV with the copper at ambient; loss = I × drop and heat in = heat out to 1e-6 |
| wpthb VHB at 3 A, wedges against the DC tool's tetrahedra | Q1 0.8552 mV against 0.8535, Q2 1.3782 against 1.3764 (+0.2%, +0.13%) |
| wpthb, whole board, 0.3 W in each FET and L3, 3 A on VHB | 385,000 wedges, 411 vias, 40 s, 1.3 GB; hottest 67.6 °C |
| wpthb from its stored 90 kHz, 10 W circuit run (`--heat-from`) | 17 parts, 0.263 W: hottest 33.5 °C; Q2 and Q4 (92 mW each) 32.7 and 33.0 °C, Q1 and Q3 (21 mW) 31.3 °C |

Two things in Elmer 26.2 that the tool works around. Its nodal Joule heating is weighted per node unless the
solver has `Constant Weights`, and then heats nothing. And the loop that computes it is not thread-safe: with
several OpenMP threads the heating is now and then wrong by orders of magnitude, so a run with current uses one
thread, and the DC tool's loss is the sum of each load's current × drop.

## Elmer: coils with ferrite

```sh
python -m kicad2ems_fem.elmer.mag --coil tx:turns=10,inner=10.25,outer=22,z=0,t=1 \
    --coil rx:turns=14,inner=8,outer=16,z=5,t=0.5 --plate z=-3,t=2.5,size=50x50,mur=2000 --out mag
```

- **Model:** each coil a flat ring of winding (mm: inner and outer radius, `z` of its lower face, thickness `t`,
  centre `x`, `y`), its turns' current spread evenly over the section: litz or fine wire below its resonance.
  Each plate a ferrite slab (`size=AxB` or `r=`, `z`, `t`, `mur`). The coils' axes are z. An air ball `--margin`
  (default 6) times the parts' reach around them.
- **Solve:** the static field of each coil alone and of each pair together; L and M from the field energy.
  Second-order edge elements on ten-node tetrahedra (first-order ones were 2 to 4% low).
- **Output:** `<out>/mag.json`: `L` (H, the matrix), `k`, and `air`: the same coils without ferrite by the ring
  formulas (Maxwell's for coaxial circles, Neumann's integral off axis), to read the result against.
- **Mesh:** `--h` at the coils and plates (default from their sections).

**In a study.** The tool is also the engine of the question *Coils and ferrite* (`elmer_mag`), which needs no
board: in kicad2ems's window, tick it under *Power*, describe each coil and each plate on a line of its own
(`tx: turns=10 inner=10.25 outer=22 z=0 t=1`; spaces or commas between), and run. Its report has the
inductances, the coupling and the same coils in air, and its result offers *Use these coils in the circuit*:
each coil's inductance for the circuit model's inductor of the same name, and each pair's coupling for the model's
coupling of those two (`pcbqs.circuit --set tx.value=... --set rx.value=... --set couplings.tx+rx.k=...`: name the
coils after the model's inductors), so a circuit's run uses the coils as solved, not a datasheet's figure. A
capacitor tuned to the old inductance is not retuned. In a config:
`[analysis.elmer_mag] coil = ["tx: turns=10 ...", "rx: ..."]`.

**At the operating frequency** (`--freq F`, Hz; `freq` in the window). Each coil is solved again alone in
the time-harmonic field at F (Elmer's WhitneyAVHarmonicSolver), and `mag.json` gains `ac`: its inductance there
and its series resistance, in three parts, and its Q:
- the ferrite's loss, from a plate's `mur_im` (μ″, the datasheet's at F): (ω/2)·μ0·μ″·|H|² a unit volume, from
  the integral of |B|² over the plate (its elemental field; the nodal one, averaged across the plate's faces,
  read 10% low). Elmer's own harmonic loss reads 0 in Elmer 26.2, its test case's too;
- a metal plate's eddy currents (`sigma=` S/m instead of `mur`: a shield, a battery's can). It is a hole in the
  air whose faces are a surface impedance, so its skin depth must be well under its thickness. The static run
  leaves it out (a non-magnetic metal is transparent at DC);
- the winding's own resistance at F, given (`r=` ohms on the coil: its datasheet's or a measurement). Its skin
  and proximity loss is not solved: the turns' current is spread over the section.

| Check, the 10-turn ring at 100 kHz | Result |
|---|---|
| A plate of µr 10, µ″ 0.1, 3 mm under it: the loss against the static field's energy, mu′ ±5% | R 4.680 mΩ; 4.685 from the energies (0.1%) |
| A copper disc 3 mm under it (r 80 mm, 2 mm thick, skin depth 0.21 mm) against its mirror image | L 1.980 µH (image 1.983), R 22.9 mΩ (the image's infinite plane 23.5); 78,000 tetrahedra, 101 s |
| An aluminium plate (3.5e7 S/m, r 40 mm, 4 mm under it) | L 1.98 µH, R 29.3 mΩ, Q 42 |
| A ferrite sheet (µr 200, µ″ 5, 1 mm thick, r 30 mm, 0.5 mm under it) | L 6.28 µH, R 3.09 mΩ, Q 1,280 |
| The sheet with the aluminium behind it | L 6.10 µH, R 5.74 mΩ (ferrite 3.43, metal 2.31), Q 667: the sheet shields the metal, which pushes more flux into the sheet |

Left out: the winding's skin and proximity loss, the ferrite's saturation, a metal plate thinner than a few skin
depths, and coils drawn as board copper, which pcbqs solves in air. The result is what the power circuit's coil
model takes: L1, L2, k; the resistance at F is reported, not proposed to the circuit (its `AC_R_ohm` is a
winding's curve over frequency, one frequency's value would stand for all of it).

| Check | Result |
|---|---|
| A 10-turn ring, 10.25 to 22 mm radius, in air, at h 2 / 1 / 0.5 mm | 3.731 / 3.736 / 3.737 µH; ring formulas 3.741 (−0.1%); 3 / 8 / 28 s |
| With a 14-turn ring 5 mm above it | L −0.12%, −0.06%, M −0.09%; k 0.4736 both; 77,000 tetrahedra, 63 s |
| The upper ring 5 / 10 / 20 mm off axis | M −0.1% / −0.1% / −0.3% against Neumann's integral; k 0.434 / 0.329 / 0.073 |
| The 10-turn ring 0.5 mm above a ferrite disc, µr 2000, 2.5 mm thick, radius 25 / 35 / 50 / 80 / 120 mm | 6.26 / 6.61 / 6.72 / 6.75 / 6.76 µH, towards the image of a thick wide plate, 6.78 (−0.26% at 120) |
| The same at µr 200, and at 0.5 mm thick | 6.65 and 6.71 µH at 120 mm (−1.6%, −1.0%): the plate carries less flux |
| The example above; and with an 0.8 mm, µr 800 plate over the receiver too | 6.45 and 6.80 µH, k 0.567 (in air 3.74, 5.77, 0.474); 8.23 and 12.86 µH, k 0.657; 74 and 114 s |

## Capacitance: Palace, and Elmer on the same model

```sh
python -m kicad2ems_fem.palace.cap board.kicad_pcb --net SW --net VIN --net GND --out cap_sw
python -m kicad2ems_fem.elmer.cap  board.kicad_pcb --net SW --net VIN --net GND --out cap_sw_elmer
```

- **Model:** the dielectric slabs inside the board outline (finite, as built), air around them, each `--net` a
  conductor (`A+B` joins nets). Copper of other nets is left out, not floating. Copper is a sheet, or with
  `--thick` a solid of its thickness; vias are cylinders.
- **Far faces:** `--far open` (no charge: the conductors' charges sum to zero) or `--far ground` (a shield
  `--margin` mm away). The capacitance to the surroundings needs `ground` and falls as 1 / margin.
- **Output:** `<out>/cap.json`: `C`, the Maxwell matrix (F): `C[i][i]` is conductor i with the others grounded,
  `−C[i][j]` the capacitance between i and j.
- **Mesh:** `--h` at the copper edges, where the charge peaks; the answer falls towards the true one as `--h`
  does. Refine until it stops moving.

The two programs share the model and the mesh generator, so they check each other's solve and output reading,
not the mesh. A 0.3 mm trace 20 mm long, 0.2 mm above a 10 mm wide ground, shield 12 mm away, h 0.2 mm
(76,000 tetrahedra, 2 to 5 s each):

| Case | Palace, pF | Elmer, pF | |
|---|---|---|---|
| In air, trace to ground | 0.6838 | 0.6847 | +0.13% |
| In air, open far faces | 0.7026 | 0.7020 | −0.09% |
| On FR-4 (εr 4.4) | 2.282 | 2.280 | −0.09% |
| On FR-4, copper 35 µm thick | 2.322 | 2.324 | +0.09% |
| Two traces 0.1 mm apart on FR-4: between them | 0.4964 | 0.4977 | +0.26% |
| The same: each to ground | 1.944 | 1.945 | +0.05% |

Against the mesh and the closed form (Palace, shield 6 mm away): in air 0.677 / 0.647 / 0.640 pF at h 0.2 / 0.05 /
0.02 mm, Hammerstad and Jensen's line 0.641; on FR-4 2.275 / 2.174 / 2.148 pF, the line 2.094, so 2.6% above it
at 444,000 tetrahedra and still falling. The copper's 35 µm adds 4.3% in air and 1.8% on FR-4 for this trace.
