"""
Antenna mode with Palace: the far field of one driven port, and from it what kicad2ems's other solvers report
(kicad2ems.fields.antenna: match, directivity, efficiency, gain, polarisation, pattern cuts and the sphere).

Palace integrates the fields on the air box's faces to the far field r E at given angles (its FarField
postprocessing), in a second run at a few frequencies over the band and the pattern frequency, the antenna's port
alone driven:

    ang = angles(step)                                    # the sphere every 5 deg and the three cuts every `step`
    ...config(mdl, ffr, tol=0, excite=[k], farfield=ang, out='postpro_ff'), run...
    ant, sphere = analyse(ffr, s11, pinc, th, ph, E, step, f_pattern)

The model is the finite board in air: its edges and its back radiate. The accepted power is the incident
wave's less the reflected, so the efficiency counts what the dielectric (and, with --palace-loss, the copper) absorbs.
"""
import numpy as np

ETA0 = 376.730313668
TH, PH = np.arange(0, 181, 5.0), np.arange(0, 361, 5.0)     # the sphere, as the other solvers' (deg)
FREQS = 11                                                  # far-field frequencies over the band (+ the pattern's)


def far_freqs(fmin, fmax, f_pattern=None):
    f = np.linspace(fmin, fmax, FREQS)
    return np.unique(np.r_[f, f_pattern]) if f_pattern else f


def _key(t, p):
    """One key per direction: at the poles every phi is the same one (Palace lists it once)."""
    t = round(float(t), 4)
    return t, 0.0 if t in (0.0, 180.0) else round(float(p) % 360.0, 4)


def cut_angles(step):
    """The pattern cuts: theta from -180 to 180 in the xz and yz planes, phi round at theta = 90 (deg)."""
    return np.arange(-180, 180 + step / 2, step), np.arange(0, 360 + step / 2, step)


def angles(step=2.0):
    """The (theta, phi) pairs (deg) asked of Palace: the sphere and the cuts, each direction once."""
    tc, pa = cut_angles(step)
    want = [(t, p) for p in PH[:-1] for t in TH]
    want += [(abs(t), p0 + (180 if t < 0 else 0)) for p0 in (0.0, 90.0) for t in tc]
    want += [(90.0, p) for p in pa]
    return np.array(list(dict.fromkeys(_key(t, p) for t, p in want)))


def spherical(E, theta, phi):
    """(E_theta, E_phi) of Cartesian far fields E (..., n, 3) at the angles theta, phi (n,), deg."""
    t, p = np.radians(theta), np.radians(phi)
    th = np.stack([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), -np.sin(t)], -1)
    ph = np.stack([-np.sin(p), np.cos(p), np.zeros_like(p)], -1)
    return np.sum(E * th, -1), np.sum(E * ph, -1)


def analyse(ffr, s11, pinc, theta, phi, E, step=2.0, f_pattern=None):
    """The antenna result of a far-field run. ffr (nf,) Hz; s11, pinc (nf,) the driven port's reflection and
    incident power (run.read_incident); theta, phi (n,), E (nf, n, 3): Palace's r E (run.read_farfield). Returns (ant, sphere): the
    result dict without the match (rows, f_pattern, cuts, sphere) and the sphere file's arrays (theta, phi, freq,
    dmax, prad, e_norm (nf, nphi, ntheta))."""
    from kicad2ems.api.solver import gain_table, add_polarisation, pattern_gain, phased
    ffr = np.asarray(ffr, float)
    E = np.asarray(E)
    at = {_key(t, p): i for i, (t, p) in enumerate(zip(theta, phi))}

    def field(t, p):                           # (E_theta, E_phi) (nf, ...) in the directions t, p (deg, any shape)
        t, p = np.broadcast_arrays(np.asarray(t, float), np.asarray(p, float))
        idx = np.array([at[_key(a, b)] for a, b in zip(t.ravel(), p.ravel())])
        et, ep = spherical(E[:, idx], t.ravel(), p.ravel())
        return et.reshape((len(ffr),) + t.shape), ep.reshape((len(ffr),) + t.shape)
    Eth, Eph = field(TH[None, :], PH[:, None])                                               # (nf, nphi, ntheta)
    dmax, prad = [], []
    for i in range(len(ffr)):
        U, pr = phased.radiated(Eth[i], Eph[i], TH, PH)
        dmax.append(4 * np.pi * U.max() / pr)
        prad.append(pr)
    pinc = np.asarray(pinc, float)
    pacc = pinc * np.maximum(1 - np.abs(s11) ** 2, 1e-12)
    rows = add_polarisation(gain_table(ffr, dmax, prad, pacc, pinc), Eth, Eph, TH, PH, phi_first=True)
    ip = int(np.argmin(np.abs(ffr - (f_pattern if f_pattern else ffr[len(ffr) // 2]))))
    def gain_db(t, p):
        et, ep = field(t, p)
        g = 4 * np.pi * (np.abs(et[ip]) ** 2 + np.abs(ep[ip]) ** 2) / (2 * ETA0) / pacc[ip]
        return [float(v) for v in 10 * np.log10(np.maximum(g, 1e-6))]
    tc, pa = cut_angles(step)
    cuts = {}
    for name, p0 in (('xz (phi = 0)', 0.0), ('yz (phi = 90)', 90.0)):
        cuts[name] = dict(angle=list(map(float, tc)), gain_dbi=gain_db(np.abs(tc), p0 + 180.0 * (tc < 0)))
    cuts['xy (theta = 90)'] = dict(angle=list(map(float, pa)), gain_dbi=gain_db(np.full(len(pa), 90.0), pa))
    e_norm = np.sqrt(np.abs(Eth) ** 2 + np.abs(Eph) ** 2)
    sphere = dict(theta=TH, phi=PH, freq=ffr, dmax=np.array(dmax), prad=np.array(prad), e_norm=e_norm)
    ant = dict(rows=rows, f_pattern=float(ffr[ip]), cuts=cuts,
               sphere=dict(theta=TH, phi=PH, gain=pattern_gain(e_norm[ip], dmax[ip], rows[ip]['eff']).T))
    return ant, sphere
