"""
The Palace config of a model (model.Model): a driven (frequency-domain) run over the band, every port excited in
turn, Palace's adaptive frequency sweep (a reduced-order model fitted to a few full solves) over the run's points.

    cfg = config(mdl, freq, order=2, tol=1e-3)            # dict, written as JSON
"""
import numpy as np


def config(mdl, freq, order=2, tol=1e-3, out='postpro', verbose=2, max_samples=40):
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
        'Boundaries': dict(PEC=dict(Attributes=[mdl.pec]), Absorbing=dict(Attributes=[mdl.absorb], Order=1),
                           LumpedPort=ports),
        'Solver': dict(Order=order, Device='CPU', Driven=driven,
                       Linear=dict(Type='Default', KSPType='GMRES', Tol=1e-8, MaxIts=400)),
    }
