"""
The Palace config of a model (model.Model): a driven (frequency-domain) run over the band, every port excited in
turn, Palace's adaptive frequency sweep (a reduced-order model fitted to a few full solves) over the run's points.

    cfg = config(mdl, freq, order=2, tol=1e-3)            # dict, written as JSON
    cfg = config(mdl, ffr, tol=0, excite=[0], farfield=angles, out='postpro_ff')    # an antenna's far field
    cfg = config(mdl, freq, box='pec')                    # the board in a metal can: the air box's faces PEC
    cfg = eigen(mdl, n=4, target=4e9, box='pec')          # its resonances above 4 GHz, each one's Q (R-E19)

The model's lumped parts are Palace lumped ports that are never excited: a boundary with the part's R, L or C.
In an eigenmode run the ports are their resistors too, so each mode's Q is the loaded one (the ports' share in
port-Q.csv), with the dielectric's tan d, the copper's loss (loss=True) and, in an open box, what radiates.
"""
import numpy as np

# The multigrid's coarse level solved with the true complex matrix, not Palace's default real stand-in (the real
# and imaginary parts added): GMRES then takes 7-10 iterations a frequency instead of 50-170, for a quarter more
# memory. Timed on the antenna examples (a dipole, 194,000 unknowns: 2.5 s a frequency instead of 8.4 s).
# Palace's error indicator (for its mesh refinement, which is not used here) is found loosely: at Palace's default
# tolerance its solve took 17 s of 40 on a board whose elements differ much in size (examples/sma_launch).
LINEAR = dict(ComplexCoarseSolve=True, EstimatorTol=1e-2)


BOXES = ('absorbing', 'pec')


def config(mdl, freq, order=2, tol=1e-3, out='postpro', verbose=2, max_samples=40, loss=False, sigma=5.8e7,
           plating=0.025, excite=None, farfield=None, amr=0, amr_tol=1e-2, box='absorbing', amr_max=0, abc=1):
    """loss: the copper as Palace's finite-thickness conductivity boundary (each layer's sheets their thickness,
    two-sided; the via walls one-sided, the plating thick) instead of PEC. Against FastHenry's volume filaments on
    a 20 x 10 mm strip loop this gave R within 1% at 1 MHz, 5% at 10 MHz and 30% high at 100 MHz (a zero-thickness
    sheet's edge current is singular; its loss grows with the mesh): fine below ~10 MHz, and for L throughout.

    excite: the ports driven (0-based, board order), each in its own solve; default all, the whole S matrix.
    farfield: (n, 2) angles (theta, phi in degrees) at which Palace writes the far field r E of each solve, from
    the fields on the air box's faces (farfield-rE.csv). amr: passes of Palace's adaptive mesh refinement (its
    Model.Refinement: the cells where its error estimate is largest split, until the estimate is below amr_tol;
    0: none; R-E3: it had never been used); amr_max: the unknowns past which it refines no further (0: no limit).
    box: the air box's faces, 'absorbing' (the board in free space) or 'pec' (a closed metal can, the model's
    margin its distance from the board on every side). abc: the absorbing boundary's order, 1 or 2 (R-E20: the
    second takes the curvature of the field on the faces too)."""
    if abc not in (1, 2):
        raise ValueError(f"abc: the absorbing boundary's order, 1 or 2, not {abc!r}")
    if farfield is not None and box != 'absorbing':
        raise ValueError("a far field needs the absorbing box (box='absorbing')")
    f = np.asarray(freq, float) / 1e9                       # GHz
    driven_ports = set(range(len(mdl.ports))) if excite is None else {int(k) for k in excite}
    ports = _ports(mdl, driven_ports)
    materials = _materials(mdl)
    uniform = len(f) > 1 and np.allclose(np.diff(f), (f[-1] - f[0]) / (len(f) - 1), rtol=1e-6, atol=0)
    if uniform:
        samples = [dict(Type='Linear', MinFreq=float(f[0]), MaxFreq=float(f[-1]), NSample=int(len(f)))]
    else:
        samples = [dict(Type='Point', Freq=[float(v) for v in f])]
    driven = dict(Samples=samples, Save=[])
    if tol and len(f) > 2:
        driven.update(AdaptiveTol=tol, AdaptiveMaxSamples=max_samples)
    boundaries = dict(**_boundaries(mdl, loss, sigma, plating, box, abc), LumpedPort=ports)
    if farfield is not None:
        boundaries['Postprocessing'] = dict(FarField=dict(
            Attributes=[mdl.absorb], ThetaPhis=[[float(t), float(p)] for t, p in np.asarray(farfield, float)]))
    return {
        'Problem': dict(Type='Driven', Verbose=verbose, Output=out),
        'Model': _model(mdl, amr, amr_tol, amr_max),
        'Domains': dict(Materials=materials),
        'Boundaries': boundaries,
        'Solver': dict(Order=order, Device='CPU', Driven=driven,
                       Linear=dict(Type='Default', KSPType='GMRES', Tol=1e-8, MaxIts=400, **LINEAR)),
    }


def eigen(mdl, n, target, order=2, out='postpro_eig', verbose=2, loss=False, sigma=5.8e7, plating=0.025,
          box='pec', tol=1e-6, amr=0, amr_tol=1e-2, amr_max=0):
    """The Palace config of an eigenmode run (R-E19): the model's n resonances above `target` Hz, each one's
    complex frequency and Q (eig.csv) and the ports' share of its loss (port-Q.csv, the ports as their
    resistors). box as config's: 'pec' a can's cavity modes, 'absorbing' the open board's (its radiation then in
    Q; the boundary first order). amr, amr_tol, amr_max as config's: the mesh refined on the modes' error."""
    if box not in BOXES:
        raise ValueError(f"box: one of {', '.join(BOXES)}, not {box!r}")
    return {
        'Problem': dict(Type='Eigenmode', Verbose=verbose, Output=out),
        'Model': _model(mdl, amr, amr_tol, amr_max),
        'Domains': dict(Materials=_materials(mdl)),
        'Boundaries': dict(**_boundaries(mdl, loss, sigma, plating, box), LumpedPort=_ports(mdl, set())),
        'Solver': dict(Order=order, Device='CPU',
                       Eigenmode=dict(N=int(n), Tol=float(tol), Target=float(target) / 1e9, Save=0),
                       Linear=dict(Type='Default', KSPType='GMRES', Tol=1e-8, MaxIts=400, **LINEAR)),
    }


def _model(mdl, amr, amr_tol, amr_max):
    """config's Model: the mesh, mm, and Palace's adaptive refinement where asked (amr passes)."""
    out = dict(Mesh=mdl.path.rsplit('/', 1)[-1], L0=1e-3)
    if amr:
        out['Refinement'] = dict(MaxIts=int(amr), Tol=float(amr_tol), SaveAdaptIterations=False,
                                 **({'MaxSize': int(amr_max)} if amr_max else {}))
    return out


def _ports(mdl, driven):
    """The lumped ports (those in `driven`, 0-based, each its own excitation) and the parts' passive boundaries."""
    ports = []
    for k, p in enumerate(mdl.ports):
        d = dict(Index=p['index'], Attributes=[p['attr']], Direction=p['direction'], R=p['R'])
        if k in driven:
            d['Excitation'] = p['index']                    # each port its own excitation
        ports.append(d)
    for q in getattr(mdl, 'parts', None) or []:             # the lumped parts: passive boundaries
        ports.append(dict(Index=q['index'], Attributes=[q['attr']], Direction=q['direction'],
                          **{k: float(v) for k, v in q['rlc'].items()}))
    return ports


def _materials(mdl):
    materials = [dict(Attributes=[a], Permittivity=float(er), LossTan=float(tand)) for a, er, tand in mdl.diel]
    materials.append(dict(Attributes=[mdl.air], Permittivity=1.0))
    return materials


def _boundaries(mdl, loss, sigma, plating, box, abc=1):
    """The copper (PEC, or lossy sheets) and the air box's faces: absorbing, or PEC (a can)."""
    if box not in BOXES:
        raise ValueError(f"box: one of {', '.join(BOXES)}, not {box!r}")
    out = _metal(mdl, loss, sigma, plating)
    if box == 'pec':
        pec = out.setdefault('PEC', dict(Attributes=[]))
        pec['Attributes'] = list(pec['Attributes']) + [mdl.absorb]
    else:
        out['Absorbing'] = dict(Attributes=[mdl.absorb], Order=int(abc))
    return out


def _metal(mdl, loss, sigma, plating):
    pec = list(mdl.pec) if isinstance(mdl.pec, (list, tuple)) else [mdl.pec]
    if not loss:
        return dict(PEC=dict(Attributes=pec))
    if not getattr(mdl, 'sheets', None):
        raise ValueError('conductor loss needs the model to tag each copper layer (palace.model.build)')
    cond = [dict(Attributes=[a], Conductivity=float(sigma), Thickness=float(t), External=False) for a, t in mdl.sheets]
    if mdl.via_wall is not None:
        cond.append(dict(Attributes=[mdl.via_wall], Conductivity=float(sigma), Thickness=float(plating), External=True))
    out = dict(Conductivity=cond)
    if getattr(mdl, 'edge_wall', None) is not None:        # the edge walls stay PEC
        out['PEC'] = dict(Attributes=[mdl.edge_wall])
    return out
