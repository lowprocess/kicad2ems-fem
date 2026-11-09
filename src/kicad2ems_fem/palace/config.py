"""
The Palace config of a model (model.Model): a driven (frequency-domain) run over the band, every port excited in
turn, Palace's adaptive frequency sweep (a reduced-order model fitted to a few full solves) over the run's points.

    cfg = config(mdl, freq, order=2, tol=1e-3)            # dict, written as JSON
"""
import numpy as np


def config(mdl, freq, order=2, tol=1e-3, out='postpro', verbose=2, max_samples=40, loss=False, sigma=5.8e7,
           plating=0.025):
    """loss: the copper as Palace's finite-thickness conductivity boundary (each layer's sheets their thickness,
    two-sided; the via walls one-sided, the plating thick) instead of PEC. Against FastHenry's volume filaments on
    a 20 x 10 mm strip loop this gave R within 1% at 1 MHz, 5% at 10 MHz and 30% high at 100 MHz (a zero-thickness
    sheet's edge current is singular; its loss grows with the mesh): fine below ~10 MHz, and for L throughout."""
    f = np.asarray(freq, float) / 1e9                       # GHz
    ports = [dict(Index=p['index'], Attributes=[p['attr']], Direction=p['direction'], R=p['R'],
                  Excitation=p['index']) for p in mdl.ports]   # each port its own excitation: the whole S matrix
    materials = [dict(Attributes=[a], Permittivity=float(er), LossTan=float(tand)) for a, er, tand in mdl.diel]
    materials.append(dict(Attributes=[mdl.air], Permittivity=1.0))
    if len(f) > 1:
        samples = [dict(Type='Linear', MinFreq=float(f[0]), MaxFreq=float(f[-1]), NSample=int(len(f)))]
    else:
        samples = [dict(Type='Point', Freq=[float(f[0])])]
    driven = dict(Samples=samples, Save=[])
    if tol and len(f) > 2:
        driven.update(AdaptiveTol=tol, AdaptiveMaxSamples=max_samples)
    return {
        'Problem': dict(Type='Driven', Verbose=verbose, Output=out),
        'Model': dict(Mesh=mdl.path.rsplit('/', 1)[-1], L0=1e-3),
        'Domains': dict(Materials=materials),
        'Boundaries': dict(**_metal(mdl, loss, sigma, plating), Absorbing=dict(Attributes=[mdl.absorb], Order=1),
                           LumpedPort=ports),
        'Solver': dict(Order=order, Device='CPU', Driven=driven,
                       Linear=dict(Type='Default', KSPType='GMRES', Tol=1e-8, MaxIts=400)),
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
    return dict(Conductivity=cond)
