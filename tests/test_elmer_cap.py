"""The Elmer capacitance tool: its .sif and matrix reading without Elmer, a microstrip against the closed form with."""
import numpy as np
import pytest
from shapely.geometry import box, mapping

from kicad2ems_fem.elmer import cap, run as elmer
from kicad2ems_fem.palace import cap as pcap
from test_palace_cap import board


def model():
    return dict(diel=[(pcap.DIEL0, 4.4)], air=pcap.AIR, outer=pcap.OUTER,
                terminals=[('T', pcap.TERM0), ('G', pcap.TERM0 + 1)])


def test_sif_has_a_mode_per_conductor():
    s = cap.sif(model(), far='open')
    assert s.count('Capacitance Body') == 2 and 'Capacitance Body = 2' in s
    assert f'Target Boundaries(1) = {pcap.TERM0 + 1}' in s
    assert 'Relative Permittivity = 4.4' in s and 'Relative Permittivity = 1\n' in s
    assert 'Potential = 0.0' not in s and f'Target Boundaries(1) = {pcap.OUTER}' not in s


def test_sif_grounds_the_far_faces_when_asked():
    s = cap.sif(model(), far='ground')
    assert f'Target Boundaries(1) = {pcap.OUTER}\n  Potential = 0.0' in s
    with pytest.raises(ValueError):
        cap.sif(model(), far='shield')


def test_maxwell_from_elmers_matrix():
    # Elmer: capacitance to the far ground on the diagonal, mutual off it
    C = cap.maxwell([[0.2, 1.0, 0.5], [1.0, 0.3, 0.1], [0.5, 0.1, 0.0]])
    assert np.allclose(C, [[1.7, -1.0, -0.5], [-1.0, 1.4, -0.1], [-0.5, -0.1, 0.6]])
    assert np.allclose(C.sum(axis=1), [0.2, 0.3, 0.0])


def hammerstad_jensen_air(w, h):
    """A microstrip's capacitance per metre in air (Hammerstad and Jensen 1980, zero thickness)."""
    u = w / h
    F = 6 + (2 * np.pi - 6) * np.exp(-(30.666 / u) ** 0.7528)
    z0 = 376.730313668 / (2 * np.pi) * np.log(F / u + np.sqrt(1 + (2 / u) ** 2))
    return 1 / (299792458.0 * z0)


@pytest.mark.skipif(not elmer.found()[0], reason='Elmer not installed')
def test_microstrip_in_air_against_the_closed_form(tmp_path):
    # 0.3 mm wide, 0.2 mm over a 10 mm ground, 20 mm long: the ends add under 1%
    top = [dict(net='T', polygon=mapping(box(-0.15, -10, 0.15, 10)))]
    b = board(er=1.0)
    b['copper'] = {'F.Cu': top, 'B.Cu': [dict(net='G', polygon=mapping(box(-5, -12, 5, 12)))]}
    b['outline'] = mapping(box(-5, -12, 5, 12))
    r = cap.solve(b, ['T', 'G'], str(tmp_path), h=0.08, far='ground', log=lambda *a: None)
    C = np.array(r['C'])
    assert r['names'] == ['T', 'G'] and np.allclose(C, C.T, rtol=1e-6)
    assert C[0, 0] > -C[0, 1] > 0                                # some of the trace's charge ends on the far shield
    assert -C[0, 1] == pytest.approx(20e-3 * hammerstad_jensen_air(0.3, 0.2), rel=0.05)
