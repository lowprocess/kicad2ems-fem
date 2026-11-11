"""
The Palace config of a model (model.Model): a driven (frequency-domain) run over the band, every port excited in
turn, Palace's adaptive frequency sweep (a reduced-order model fitted to a few full solves) over the run's points.

    cfg = config(mdl, freq, order=2, tol=1e-3)            # dict, written as JSON
    cfg = config(mdl, ffr, tol=0, excite=[0], farfield=angles, out='postpro_ff')    # an antenna's far field

The model's lumped parts are Palace lumped ports that are never excited: a boundary with the part's R, L or C.
"""
import numpy as np

# The multigrid's coarse level solved with the true complex matrix, not Palace's default real stand-in (the real
# and imaginary parts added): GMRES then takes 7-10 iterations a frequency instead of 50-170, for a quarter more
# memory. Timed on the antenna examples (a dipole, 194,000 unknowns: 2.5 s a frequency instead of 8.4 s).
# Palace's error indicator (for its mesh refinement, which is not used here) is found loosely: at Palace's default
# tolerance its solve took 17 s of 40 on a board whose elements differ much in size (examples/sma_launch).
LINEAR = dict(ComplexCoarseSolve=True, EstimatorTol=1e-2)


def config(mdl, freq, order=2, tol=1e-3, out='postpro', verbose=2, max_samples=40, loss=False, sigma=5.8e7,
           plating=0.025, excite=None, farfield=None):
    """loss: the copper as Palace's finite-thickness conductivity boundary (each layer's sheets their thickness,
    two-sided; the via walls one-sided, the plating thick) instead of PEC. Against FastHenry's volume filaments on
    a 20 x 10 mm strip loop this gave R within 1% at 1 MHz, 5% at 10 MHz and 30% high at 100 MHz (a zero-thickness
    sheet's edge current is singular; its loss grows with the mesh): fine below ~10 MHz, and for L throughout.

    excite: the ports driven (0-based, board order), each in its own solve; default all, the whole S matrix.
    farfield: (n, 2) angles (theta, phi in degrees) at which Palace writes the far field r E of each solve, from
    the fields on the air box's faces (farfield-rE.csv)."""
    f = np.asarray(freq, float) / 1e9                       # GHz
    driven_ports = set(range(len(mdl.ports))) if excite is None else {int(k) for k in excite}
    ports = []
    for k, p in enumerate(mdl.ports):
        d = dict(Index=p['index'], Attributes=[p['attr']], Direction=p['direction'], R=p['R'])
        if k in driven_ports:
            d['Excitation'] = p['index']                    # each port its own excitation
        ports.append(d)
    for q in getattr(mdl, 'parts', None) or []:             # the lumped parts: passive boundaries
        ports.append(dict(Index=q['index'], Attributes=[q['attr']], Direction=q['direction'],
                          **{k: float(v) for k, v in q['rlc'].items()}))
    materials = [dict(Attributes=[a], Permittivity=float(er), LossTan=float(tand)) for a, er, tand in mdl.diel]
    materials.append(dict(Attributes=[mdl.air], Permittivity=1.0))
    uniform = len(f) > 1 and np.allclose(np.diff(f), (f[-1] - f[0]) / (len(f) - 1), rtol=1e-6, atol=0)
    if uniform:
        samples = [dict(Type='Linear', MinFreq=float(f[0]), MaxFreq=float(f[-1]), NSample=int(len(f)))]
    else:
        samples = [dict(Type='Point', Freq=[float(v) for v in f])]
    driven = dict(Samples=samples, Save=[])
    if tol and len(f) > 2:
        driven.update(AdaptiveTol=tol, AdaptiveMaxSamples=max_samples)
    boundaries = dict(**_metal(mdl, loss, sigma, plating), Absorbing=dict(Attributes=[mdl.absorb], Order=1),
                      LumpedPort=ports)
    if farfield is not None:
        boundaries['Postprocessing'] = dict(FarField=dict(
            Attributes=[mdl.absorb], ThetaPhis=[[float(t), float(p)] for t, p in np.asarray(farfield, float)]))
    return {
        'Problem': dict(Type='Driven', Verbose=verbose, Output=out),
        'Model': dict(Mesh=mdl.path.rsplit('/', 1)[-1], L0=1e-3),
        'Domains': dict(Materials=materials),
        'Boundaries': boundaries,
        'Solver': dict(Order=order, Device='CPU', Driven=driven,
                       Linear=dict(Type='Default', KSPType='GMRES', Tol=1e-8, MaxIts=400, **LINEAR)),
    }


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
