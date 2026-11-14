"""
kicad2ems-fem's tools as kicad2ems analysis plugins (the 'kicad2ems.analyses' entry points of pyproject.toml): they
answer questions other engines answer too (kicad2ems.study), in 3D, so a study can ask the same of two programs
and lay the answers side by side.

    elmer_dc        a rail's DC drop                 (python -m kicad2ems_fem.elmer.dc)
    elmer_thermal   the board's temperature          (python -m kicad2ems_fem.elmer.thermal)
    elmer_cap       the capacitance between nets     (python -m kicad2ems_fem.elmer.cap)
    palace_cap      the same, by Palace              (python -m kicad2ems_fem.palace.cap)
    elmer_mag       wound coils beside ferrite       (python -m kicad2ems_fem.elmer.mag): its own question, no board

Each one's settings are its command line's options, read from the tool's own parser, so a new option of a tool
appears in a front end without a change here. Only this module imports kicad2ems.api.analysis.
"""
import json
import math
import os
import re
import sys

from kicad2ems.api.analysis import Analysis

API = (2, 6)
ORDER = ('Coils and plates', 'Nets and loads', 'Heat', 'Air and board', 'Mesh', 'Solver')
GROUPS = dict(coil='Coils and plates', plate='Coils and plates',
              net='Nets and loads', source='Nets and loads', load='Nets and loads', copper='Nets and loads',
              heat='Heat', heat_from='Heat', htc='Air and board', ambient='Air and board', k_board='Air and board',
              far='Air and board', margin='Air and board', region='Air and board',
              h='Mesh', h_fine='Mesh', hmax='Mesh', plating='Mesh', thick='Mesh', order='Solver', np='Solver')


PICK = dict(net='nets', copper='nets', source='part')       # settings a front end may pick on the board (API 2.11)


def describe(parser):
    """A tool's options as kicad2ems's analyses take them, [dict(key, flag, kind, default, help, group, choices,
    required)], from its argparse parser: every option but the board and the output folder, which a front end
    gives."""
    out = []
    for a in parser._actions:
        if not a.option_strings or a.dest in ('help', 'out'):
            continue
        kind = ('bool' if a.nargs == 0 else 'list' if type(a).__name__ == '_AppendAction' else
                'float' if a.type is float else 'int' if a.type is int else 'str')
        out.append(dict(key=a.dest, flag=a.option_strings[0], kind=kind,
                        default=([] if kind == 'list' else False if kind == 'bool' else
                                 '' if a.default is None else a.default),
                        help=a.help or '', group=GROUPS.get(a.dest, 'Solver'), choices=list(a.choices or ()),
                        required=bool(a.required)))
        if a.dest in PICK:
            out[-1]['pick'] = PICK[a.dest]
    return sorted(out, key=lambda o: ORDER.index(o['group']))


def argv(module, options, values, board, out):
    """[python, -m, module, board, --out, out, ...] for values {option key: text, bool or list}: an option left
    empty or at its default is not passed; a list is lines or comma-separated items."""
    line = [sys.executable, '-m', module, board, '--out', out]
    for o in options:
        v = values.get(o['key'])
        if o['kind'] == 'bool':
            if v is True or str(v).strip().lower() in ('1', 'true', 'yes', 'on'):
                line.append(o['flag'])
        elif o['kind'] == 'list':
            text = '\n'.join(map(str, v)) if isinstance(v, (list, tuple)) else str(v or '')
            for item in [s.strip() for s in re.split(r'[\n,]', text) if s.strip()]:
                line += [o['flag'], item]
        else:
            s = '' if v is None else str(v).strip()
            if s and s != str(o['default']):
                line += [o['flag'], s]
    return line


class _Fem(Analysis):
    """One of the package's tools: its module, its parser, the program it needs, its result file."""
    api = API
    module = ''
    program = 'elmer'

    def check(self):
        if self.program == 'palace':
            from .palace.plugin import palace_status
            return palace_status()
        from .elmer.run import found
        return found()

    def _parser(self):
        import importlib
        return importlib.import_module(self.module).parser()

    def options(self):
        return describe(self._parser())

    def owns(self, folder, preview=False):
        """A result of this tool's: its file says whose it is (pcbqs writes a thermal.json too)."""
        return not preview and (self._read(folder) or {}).get('tool') == self.module

    def _read(self, folder):
        try:
            with open(os.path.join(folder or '', self.result), encoding='utf-8') as fh:
                d = json.load(fh)
            return d if isinstance(d, dict) else None
        except (OSError, ValueError):
            return None

    def command(self, values, board, out, preview=False, results=None):
        if not board or not os.path.exists(board):
            raise ValueError('choose a board first' if not board else f'{board}: no such board')
        opts = self.options()
        need = [o['flag'] for o in opts if o['required'] and not str(values.get(o['key']) or '').strip()]
        if need:
            raise ValueError(f"set {', '.join(need)}")
        return argv(self.module, opts, values, board, out)

    def status(self, line):
        return line.split(':', 1)[1].strip()[:160] if line.startswith('fem ') and ':' in line else None

    def curves(self, folder):
        return []


def _seconds(r):
    s = r.get('seconds')
    return sum(s.values()) if isinstance(s, dict) else (s or 0.0) + (r.get('mesh_s') or 0.0)


class ElmerDC(_Fem):
    name, label, engine, tool, module = 'elmer_dc', 'Rail drop (Elmer)', 'Elmer', 'kicad2ems_fem.elmer.dc', 'kicad2ems_fem.elmer.dc'
    about = "a net's DC drop in its 3D copper and via barrels: the check of a rail tool's copper"
    order, result, provides = 25, 'dc.json', ('dc',)

    def tag(self, values):
        v = values.get('net')
        return str((v[0] if isinstance(v, (list, tuple)) and v else v) or 'net').split(',')[0].strip()

    def report(self, folder):
        r = self._read(folder)
        if not r:
            return None
        worst = max(r['drop_mV'].items(), key=lambda kv: kv[1], default=None)
        rows = [[ref, r['loads'][ref], mv] for ref, mv in r['drop_mV'].items()]
        return dict(
            title=f"{' + '.join(r['nets'])}: DC drop from {r['source']}",
            sections=[dict(title='', text=(f"Worst drop {worst[1]:.3f} mV at {worst[0]}. " if worst else '') +
                           f"{r['loss_W'] * 1e3:.3f} mW in the copper."),
                      dict(title='Loads', table=dict(columns=['load', 'current (A)', 'drop (mV)'], rows=rows,
                                                    formats=[None, '.4g', '.4f']))],
            notes=["The net's own copper and vias in 3D; series parts and the ground return are not in it."],
            footer=f"Elmer, {r['tets']:,} tetrahedra, {_seconds(r):.1f} s")

    def summary(self, folder):
        r = self._read(folder)
        return f"worst drop {max(r['drop_mV'].values()):.3f} mV" if r else ''

    def figures(self, folder):
        """Each load's drop in the net's own copper, and the loss in it."""
        r = self._read(folder) or {}
        out = [dict(name=f"drop {ref}", value=float(mv), unit='mV') for ref, mv in (r.get('drop_mV') or {}).items()]
        if isinstance(r.get('loss_W'), (int, float)):
            out.append(dict(name='loss', value=r['loss_W'] * 1e3, unit='mW'))
        return out


class ElmerThermal(_Fem):
    name, label, engine = 'elmer_thermal', 'Temperature (Elmer)', 'Elmer'
    tool = module = 'kicad2ems_fem.elmer.thermal'
    about = ("the board's steady temperature in 3D: all its copper, vias and laminate, the parts' heat and a "
             "net's own current")
    order, result, provides = 55, 'thermal.json', ('thermal',)
    paths = ('heat_from',)

    def tag(self, values):
        return 'heat'

    def command(self, values, board, out, preview=False, results=None):
        values = dict(values)
        if not str(values.get('heat') or '').strip() and not str(values.get('heat_from') or '').strip():
            run = (results or {}).get('circuit')                # the latest circuit run on this board: its losses
            case = self._case(run) if run else None
            if case is None:
                raise ValueError('give the heat (REF=WATTS), or run the circuit first: its losses are taken')
            values['heat_from'] = case
        return super().command(values, board, out)

    @staticmethod
    def _case(run):
        """The case.json of a pcbqs circuit run whose losses to take: its heaviest case's."""
        best = None
        for name in sorted(os.listdir(run)) if os.path.isdir(run) else []:
            path = os.path.join(run, name, 'case.json')
            try:
                with open(path, encoding='utf-8') as fh:
                    parts = json.load(fh).get('parts', {})
            except (OSError, ValueError):
                continue
            loss = sum(p.get('power_W', 0.0) for p in parts.values() if p.get('power_W', 0.0) > 0)
            if best is None or loss > best[0]:
                best = (loss, path)
        return best[1] if best else None

    def figures(self, folder):
        """The copper's hottest point and the board at each part's pads; the coolest point, which is its own."""
        r = self._read(folder) or {}
        if not isinstance(r.get('T_max'), (int, float)):
            return []
        return [dict(name='hottest', value=float(r['T_max']), unit='C')] + \
               [dict(name=ref, value=float(t), unit='C') for ref, t in (r.get('parts') or {}).items()] + \
               ([dict(name='coolest', value=float(r['T_min']), unit='C')] if isinstance(r.get('T_min'), (int, float)) else [])

    def report(self, folder):
        r = self._read(folder)
        if not r:
            return None
        rows = [[ref, r['heat'].get(ref), t, t - r['ambient']] for ref, t in sorted(r['parts'].items(), key=lambda kv: -kv[1])]
        b = r['balance']
        sections = [dict(title='', text=f"Hottest {r['T_max']:.1f} °C on {r['hottest'].get('layer') or 'the board'}; "
                                        f"coolest {r['T_min']:.1f} °C; the air at {r['ambient']:g} °C. "
                                        f"{b['in_W']:.3g} W in, {b['out_W']:.3g} W to the air."),
                    dict(title='Parts', table=dict(columns=['part', 'heat (W)', 'at its pads (°C)', 'rise (K)'], rows=rows,
                                                   formats=[None, '.3g', '.1f', '.1f']))]
        if r.get('drop_mV'):
            sections.append(dict(title='Current', table=dict(
                columns=['load', 'drop, warm (mV)', 'drop, cold (mV)'],
                rows=[[ref, mv, r['drop_ambient_mV'][ref]] for ref, mv in r['drop_mV'].items()], formats=[None, '.4f', '.4f'])))
        picture = os.path.join(folder, 'thermal.png')
        return dict(title=f"Temperature: hottest {r['T_max']:.1f} °C", sections=sections,
                    notes=["A part's heat all enters the board: its pad temperature is an upper bound."],
                    images=[picture] if os.path.exists(picture) else [],
                    footer=f"Elmer, {r['wedges']:,} wedges, {_seconds(r):.1f} s")


class _Cap(_Fem):
    result, provides = 'cap.json', ('capacitance',)

    def tag(self, values):
        v = values.get('net')
        items = v if isinstance(v, (list, tuple)) else re.split(r'[\n,]', str(v or ''))
        return '-'.join(s.strip() for s in items if str(s).strip())[:40] or 'nets'

    def figures(self, folder):
        """The Maxwell matrix by name: a net to all the others (its diagonal), and each two nets (minus the entry
        between them)."""
        r = self._read(folder) or {}
        names, C = r.get('names') or [], r.get('C') or []
        out = [dict(name=f"C {n}", value=C[i][i] * 1e12, unit='pF') for i, n in enumerate(names)]
        return out + [dict(name=f"C {a},{b}", value=-C[i][j] * 1e12, unit='pF')
                      for i, a in enumerate(names) for j, b in enumerate(names) if j > i]

    def report(self, folder):
        r = self._read(folder)
        if not r:
            return None
        names = r['names']
        rows = [[n] + [c * 1e12 for c in row] for n, row in zip(names, r['C'])]
        return dict(
            title=f"Capacitance between {', '.join(names)}",
            sections=[dict(title='Maxwell matrix (pF)', table=dict(columns=[''] + names, rows=rows,
                                                                    formats=[None] + ['.4g'] * len(names)))],
            notes=["A diagonal entry is a conductor's capacitance to all the others together; an off-diagonal one "
                   "is minus the capacitance between the two.",
                   f"Copper as {'solids of its thickness' if r.get('thick') else 'sheets'}; the air box's faces "
                   f"{r.get('far', 'open')}."],
            footer=f"{self.engine}, {r['tets']:,} tetrahedra, {_seconds(r):.1f} s")


class ElmerCap(_Cap):
    name, label, engine = 'elmer_cap', 'Capacitance (Elmer)', 'Elmer'
    tool = module = 'kicad2ems_fem.elmer.cap'
    about = "the capacitance matrix of nets' copper in the board's dielectrics, by Elmer's electrostatics"
    order = 60


class PalaceCap(_Cap):
    name, label, engine, program = 'palace_cap', 'Capacitance (Palace)', 'Palace', 'palace'
    tool = module = 'kicad2ems_fem.palace.cap'
    about = "the capacitance matrix of nets' copper in the board's dielectrics, by Palace's electrostatics"
    order = 61


class ElmerMag(_Fem):
    """Wound coils beside ferrite plates: their inductances and coupling (elmer/mag.py). The coils are parts, not
    board copper: a run needs no board, and its result proposes the coil model of pcbqs's full bridge (the coils'
    inductances and their coupling) to the circuit."""
    name, label, engine = 'elmer_mag', 'Coils with ferrite (Elmer)', 'Elmer'
    tool = module = 'kicad2ems_fem.elmer.mag'
    about = "wound coils beside ferrite plates: their inductances and coupling, for a wireless-power circuit"
    order, result, provides = 45, 'mag.json', ('coils',)

    def options(self):
        opts = super().options()
        for o in opts:
            if o['key'] in ('coil', 'plate'):
                o['level'] = 'basic'
        return opts

    def tag(self, values):
        coils = self._items(values.get('coil'))
        names = [str(c).split(':')[0].strip() if ':' in str(c) else f"coil{k + 1}" for k, c in enumerate(coils)]
        return '_'.join(re.sub(r'[^A-Za-z0-9]+', '-', n) for n in names[:3]) or 'coils'

    @staticmethod
    def _items(v):
        """A repeatable setting's items, one a line (an item has commas or spaces of its own)."""
        v = v if isinstance(v, (list, tuple)) else str(v or '').split('\n')
        return [str(x).strip() for x in v if str(x).strip()]

    def command(self, values, board, out, preview=False, results=None):
        coils = self._items(values.get('coil'))
        if not coils:
            raise ValueError("describe a coil, one a line: NAME: turns=N inner=R outer=R z=Z t=T (mm)")
        cmd = [sys.executable, '-m', self.module, '--out', str(out)]        # (no board: the coils are parts)
        for c in coils:
            cmd += ['--coil', c]
        for p in self._items(values.get('plate')):
            cmd += ['--plate', p]
        for o in self.options():
            v = str(values.get(o['key']) if values.get(o['key']) is not None else '').strip()
            if o['key'] in ('h', 'margin') and v and v != str(o['default']):
                cmd += [o['flag'], v]
        return cmd

    def figures(self, folder):
        r = self._read(folder)
        if not r or r.get('tool') != self.module:
            return []
        n = r['names']
        out = [dict(name=f"L {a}", value=r['L'][i][i] * 1e6, unit='uH') for i, a in enumerate(n)]
        return out + [dict(name=f"k {n[j]},{n[i]}", value=r['k'][i][j], unit='') for i in range(len(n))
                      for j in range(i)]

    def report(self, folder):
        r = self._read(folder)
        if not r or r.get('tool') != self.module:
            return None
        n, L, air = r['names'], r['L'], r['air']
        coils = {c['name']: c for c in r.get('coils', [])}
        sec = [dict(title='Inductances',
                    text="Each coil's own inductance beside the ferrite, and without it by the ring formulas "
                         "(Maxwell's for coaxial circles, Neumann's integral off axis).",
                    table=dict(columns=['coil', 'turns', 'L (uH)', 'in air (uH)', 'with the ferrite'],
                               rows=[[a, coils.get(a, {}).get('turns', ''), L[i][i] * 1e6, air[i][i] * 1e6,
                                      f"x {L[i][i] / air[i][i]:.2f}"] for i, a in enumerate(n)],
                               formats=[None, '.0f', '.3f', '.3f', None]))]
        pairs = [(i, j) for i in range(len(n)) for j in range(i)]
        if pairs:
            ka = lambda i, j: air[i][j] / (air[i][i] * air[j][j]) ** 0.5
            sec.append(dict(title='Coupling',
                            table=dict(columns=['coils', 'M (uH)', 'k', 'k in air'],
                                       rows=[[f"{n[j]} to {n[i]}", L[i][j] * 1e6, r['k'][i][j], ka(i, j)] for i, j in pairs],
                                       formats=[None, '.3f', '.4f', '.4f'])))
        if r.get('plates'):
            sec.append(dict(title='Ferrite plates',
                            table=dict(columns=['z (mm)', 'thickness (mm)', 'mu_r', 'size (mm)'],
                                       rows=[[p.get('z', 0.0), p.get('t', 0.0), p.get('mur', 1.0),
                                              f"r {p['r']:g}" if 'r' in p else 'x'.join(f"{v:g}" for v in p.get('size', ()))]
                                             for p in r['plates']], formats=['.2f', '.2f', '.0f', None])))
        head = ', '.join(f"{a} {L[i][i] * 1e6:.3g} uH" for i, a in enumerate(n))
        if pairs:
            i, j = pairs[0]
            head += f"; k {r['k'][i][j]:.3f} ({n[j]} to {n[i]})"
        return dict(title=f"Coils with ferrite: {', '.join(n)}", summary=head, sections=sec,
                    notes=["Static field: no winding resistance, skin or proximity loss, no ferrite loss or "
                           "saturation, no eddy currents in metal nearby."],
                    footer=f"Elmer, {r.get('tets', 0):,} tetrahedra, {_seconds(r):.0f} s")

    def suggest(self, folder):
        """The first two coils as the wpthb bridge's (pcbqs's data/wpthb_bridge.json): the primary's and the
        secondary's inductance and their coupling. The receiver is left as the model has it (its capacitor is not
        retuned for the new secondary): the note says what would be resonant with it."""
        r = self._read(folder)
        if not r or r.get('tool') != self.module or len(r['names']) < 2:
            return []
        n, L = r['names'], r['L']
        l1, l2, k = L[0][0], L[1][1], abs(r['k'][1][0])
        tuned = lambda f: 1 / ((2 * math.pi * f) ** 2 * l2) * 1e9
        return [dict(to='circuit', label=f"Use these coils in the circuit ({n[0]} {l1 * 1e6:.3g} uH, "
                                         f"{n[1]} {l2 * 1e6:.3g} uH, k {k:.3f})",
                     values=dict(set='\n'.join([f"PRIMARY.value={l1:.6g}", f"SECONDARY.value={l2:.6g}",
                                                f"couplings.0.k={k:.4f}"])), need=[],
                     note="the circuit model's coils PRIMARY and SECONDARY and their coupling (pcbqs's wpthb bridge "
                          "names them so; another model's: edit the references in Set). Its receiver stays as it "
                          f"is: series-resonant with this {n[1]} at 90 kHz would be {tuned(90e3):.3g} nF "
                          f"({tuned(110e3):.3g} nF at 110 kHz), where the bridge's c2 is 130 nF for its 24 uH; add "
                          f"parameters.c2 to Set for that")]
