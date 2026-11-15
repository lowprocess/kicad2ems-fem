"""
kicad2ems-fem's tools as kicad2ems analysis plugins (the 'kicad2ems.analyses' entry points of pyproject.toml): they
answer questions other engines answer too (kicad2ems.study), in 3D, so a study can ask the same of two programs
and lay the answers side by side.

    elmer_dc        a rail's DC drop                 (python -m kicad2ems_fem.elmer.dc)
    elmer_thermal   the board's temperature          (python -m kicad2ems_fem.elmer.thermal)
    elmer_cap       the capacitance between nets     (python -m kicad2ems_fem.elmer.cap)
    palace_cap      the same, by Palace              (python -m kicad2ems_fem.palace.cap)
    elmer_mag       wound coils beside ferrite       (python -m kicad2ems_fem.elmer.mag): its own question, no board
    palace_pdn      a rail's impedance at its loads   (python -m kicad2ems_fem.palace.pdn): the rail pcbqs's PDN run
                    found, each part a port at its pads, closed by its parts after the solve
    palace_loop     a converter's input loop         (python -m kicad2ems_fem.palace.loop): the loop pcbqs's near
                    field found, its inductance again in 3D

Each one's settings are its command line's options, read from the tool's own parser, so a new option of a tool
appears in a front end without a change here. Only this module imports kicad2ems.api.analysis.
"""
import json
import os
import re
import sys

from kicad2ems.api.analysis import Analysis

API = (2, 11)                       # kicad2ems.api it is written against: 2.8 figures, 2.11 pick
ORDER = ('Coils and plates', 'Nets and loads', 'Heat', 'Air and board', 'Mesh', 'Solver')
GROUPS = dict(coil='Coils and plates', plate='Coils and plates',
              net='Nets and loads', source='Nets and loads', load='Nets and loads', copper='Nets and loads',
              heat='Heat', heat_from='Heat', htc='Air and board', ambient='Air and board', k_board='Air and board',
              far='Air and board', margin='Air and board', region='Air and board',
              h='Mesh', h_fine='Mesh', hmax='Mesh', plating='Mesh', mesh='Mesh', thick='Mesh', order='Solver', np='Solver')


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
            footer=f"Elmer, " + (f"{r['tets']:,} tetrahedra" if 'tets' in r else f"{r.get('wedges', 0):,} wedges (layered)")
                   + f", {_seconds(r):.1f} s")

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
               ([dict(name='coolest', value=float(r['T_min']), unit='C')] if isinstance(r.get('T_min'), (int, float)) else []) + \
               [dict(name=f"{ref} to 63 %", value=float(t63), unit='s')
                for ref, t63 in ((r.get('transient') or {}).get('to_63_s') or {}).items()]

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
        tr = r.get('transient')
        if tr:                                                   # (R-E21: --transient)
            sections.append(dict(title=f"Heating from cold, to {tr['end_s']:g} s",
                                 text="The parts' heat on at 0 s, the board at the air's temperature before; the "
                                      "board's own heat capacity, not the parts' bodies'. When each part's pads are "
                                      "63 % of the way to their steady temperature:",
                                 table=dict(columns=['part', 'to 63 % (s)'],
                                            rows=[[ref, t63] for ref, t63 in sorted(tr['to_63_s'].items(),
                                                                                    key=lambda kv: kv[1])],
                                            formats=[None, '.4g'])))
        images = [os.path.join(folder, f) for f in ('thermal.png', 'thermal-transient.png')]
        return dict(title=f"Temperature: hottest {r['T_max']:.1f} °C", sections=sections,
                    notes=["A part's heat all enters the board: its pad temperature is an upper bound."],
                    images=[f for f in images if os.path.exists(f)],
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


def _loop_L(f, L):
    """A loop's figures: its inductance at each frequency, nH (the names pcbqs's near field gives them too)."""
    return [dict(name=f"loop L at {fi / 1e6:g} MHz", value=li * 1e9, unit='nH') for fi, li in zip(f, L)
            if isinstance(li, (int, float))]


class PalaceLoop(_Fem):
    """A converter's input loop in Palace (R-E22): the loop of the board's newest pcbqs near-field run (its converter,
    input net, ground, input capacitors and crop) solved again in 3D, its inductance at the run's harmonics beside
    pcbqs's. Palace stays out of pcbqs: this engine reads pcbqs's result file, not pcbqs."""
    name, label, engine, program = 'palace_loop', 'Converter loop (Palace)', 'Palace', 'palace'
    tool = module = 'kicad2ems_fem.palace.loop'
    about = "a converter's input loop in 3D: its inductance, beside pcbqs's near field's"
    order, result, provides = 35, 'loop.json', ('nearfield',)

    def options(self):
        return [dict(o, required=False) if o['key'] == 'mqs' else o for o in super().options()]

    def missing(self, values, board, results=None):
        if str(values.get('mqs') or '').strip() or (results or {}).get('mqs'):
            return []
        return [dict(key='mqs', text="run the near field (pcbqs) first: its loop is what Palace solves")]

    def command(self, values, board, out, preview=False, results=None):
        run = str(values.get('mqs') or '').strip() or (results or {}).get('mqs')
        if not run:
            raise ValueError("run the near field (pcbqs) first: its loop is what Palace solves")
        cmd = [sys.executable, '-m', self.module, '--mqs', str(run), '--out', str(out)]
        for o in self.options():
            v = str(values.get(o['key']) if values.get(o['key']) is not None else '').strip()
            if o['key'] not in ('mqs', 'out') and v and v != str(o['default']):
                cmd += [o['flag'], v]
        return cmd

    def status(self, line):
        return line.split(':', 1)[1].strip()[:160] if line.startswith('palace loop:') else None

    def figures(self, folder):
        r = self._read(folder)
        return _loop_L(r['frequencies_Hz'], r['L_H']) if r and r.get('tool') == self.module else []

    def report(self, folder):
        r = self._read(folder)
        if not r or r.get('tool') != self.module:
            return None
        sel, pc = r['selection'], (r.get('pcbqs') or {}).get('L_H') or [None] * len(r['L_H'])
        rows = [[fi / 1e6, li * 1e9, p * 1e9 if p else '', f"{li / p:.3f}" if p else '']
                for fi, li, p in zip(r['frequencies_Hz'], r['L_H'], pc)]
        notes = ["Palace's copper is PEC: the high-frequency limit of pcbqs's, whose current leaves the copper's bulk "
                 "for its surfaces over the harmonics; compare at the higher ones.",
                 f"The input capacitors ({', '.join(r['straps']) or 'none'}) are metal straps, as pcbqs's copper "
                 "inductance has them; the source a lumped port across the converter's closest input and ground pads."]
        if r.get('missing_capacitors'):
            notes.append(f"Not on the cropped copper, so left out: {', '.join(r['missing_capacitors'])}")
        return dict(title=f"Converter loop (Palace): {sel['converter']}, {sel['vin']}",
                    summary=f"L {r['L_H'][-1] * 1e9:.3g} nH at {r['frequencies_Hz'][-1] / 1e6:g} MHz"
                            + (f" (pcbqs {pc[-1] * 1e9:.3g} nH)" if pc[-1] else ''),
                    sections=[dict(title='Loop inductance', table=dict(
                        columns=['f (MHz)', 'Palace (nH)', 'pcbqs (nH)', 'Palace / pcbqs'], rows=rows,
                        formats=['.4g', '.4g', '.4g', None]))],
                    notes=notes, footer=f"Palace, {r.get('tets', 0):,} tetrahedra, {_seconds(r):.0f} s")


class PalacePDN(_Fem):
    """A rail's PDN in Palace (R-E22): the rail of the board's newest pcbqs PDN run, each part a lumped port at its
    pads, the bare copper solved in 3D and closed by the run's parts; each load's |Z| beside pcbqs's. Palace stays
    out of pcbqs: this engine reads pcbqs's result file, not pcbqs."""
    name, label, engine, program = 'palace_pdn', 'Rail impedance (Palace)', 'Palace', 'palace'
    tool = module = 'kicad2ems_fem.palace.pdn'
    about = "a rail's impedance at its loads in 3D, beside pcbqs's PDN"
    order, result, provides = 36, 'pdn_palace.json', ('pdn',)

    def options(self):
        return [dict(o, required=False) if o['key'] == 'pdn' else o for o in super().options()]

    def missing(self, values, board, results=None):
        if str(values.get('pdn') or '').strip() or (results or {}).get('pdn'):
            return []
        return [dict(key='pdn', text="run the PDN (pcbqs) first: its rail and parts are what Palace solves")]

    def command(self, values, board, out, preview=False, results=None):
        run = str(values.get('pdn') or '').strip() or (results or {}).get('pdn')
        if not run:
            raise ValueError("run the PDN (pcbqs) first: its rail and parts are what Palace solves")
        cmd = [sys.executable, '-m', self.module, '--pdn', str(run), '--out', str(out)]
        for o in self.options():
            v = str(values.get(o['key']) if values.get(o['key']) is not None else '').strip()
            if o['key'] not in ('pdn', 'out') and v and v != str(o['default']):
                cmd += [o['flag'], v]
        return cmd

    def status(self, line):
        return line.split(':', 1)[1].strip()[:160] if line.startswith('palace pdn:') else None

    def report(self, folder):
        r = self._read(folder)
        if not r or r.get('tool') != self.module:
            return None
        f = r['frequencies_Hz']
        sections = []
        for load, z in r['z'].items():
            pq = z.get('pcbqs_mag') or [None] * len(f)
            rows = [[fi / 1e6, m * 1e3, p * 1e3 if p else '', f"{m / p:.3f}" if p else '']
                    for fi, m, p in zip(f, z['mag'], pq)]
            sections.append(dict(title=f"{load}: |Z|", table=dict(columns=['f (MHz)', 'Palace (mOhm)', 'pcbqs (mOhm)',
                                                                          'Palace / pcbqs'], rows=rows,
                                                                 formats=['.4g', '.4g', '.4g', None])))
        bare = r.get('bare_vs_pcbqs') or {}
        notes = ["Each part is a lumped port across its copper's gap between rail and ground at the part (a series "
                 "part's: between its pads); the decaps and the regulator are closed by the run's values after the "
                 "solve, the loads left open: the copper from the pads to the planes is Palace's own.",
                 "Palace's copper is PEC and its solve full-wave: below a few MHz pcbqs's PDN is the answer."]
        if bare:
            notes.append("The bare network (no part) against pcbqs's, |Z| Palace / pcbqs: " + '; '.join(
                f"{k} {min(v):.2f}..{max(v):.2f}" for k, v in bare.items()))
        if r.get('left_out'):
            notes.append(f"No port of their own (open in both): {', '.join(r['left_out'])}")
        last = next(iter(r['z'].values()))
        return dict(title=f"Rail impedance (Palace): {r['rail']}",
                    summary=f"{next(iter(r['z']))}: |Z| {last['mag'][-1] * 1e3:.3g} mOhm at {f[-1] / 1e6:g} MHz"
                            + (f" (pcbqs {last['pcbqs_mag'][-1] * 1e3:.3g})" if last.get('pcbqs_mag') else ''),
                    sections=sections, notes=notes,
                    footer=f"Palace, {r.get('tets', 0):,} tetrahedra, {_seconds(r):.0f} s")


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
            if o['key'] in ('h', 'margin', 'freq') and v and v not in (str(o['default']), 'None'):
                cmd += [o['flag'], v]
        return cmd

    def figures(self, folder):
        r = self._read(folder)
        if not r or r.get('tool') != self.module:
            return []
        n = r['names']
        out = [dict(name=f"L {a}", value=r['L'][i][i] * 1e6, unit='uH') for i, a in enumerate(n)]
        out += [dict(name=f"k {n[j]},{n[i]}", value=r['k'][i][j], unit='') for i in range(len(n)) for j in range(i)]
        for a in r.get('ac') or []:                              # at the operating frequency (R-E21)
            out += [dict(name=f"R {a['name']} at {r['freq'] / 1e3:g} kHz", value=a['R'] * 1e3, unit='mohm')]
            if a.get('Q'):
                out += [dict(name=f"Q {a['name']} at {r['freq'] / 1e3:g} kHz", value=a['Q'], unit='')]
        return out

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
        if r.get('ac'):
            f = r['freq']
            sec.append(dict(title=f"At {f / 1e3:g} kHz",
                            text="Each coil alone in the time-harmonic field: its inductance with the metal's eddy "
                                 "currents, and its series resistance from the ferrite's loss (mu''), the metal's "
                                 "(its surface impedance) and the winding's own (r=, given). The winding's skin and "
                                 "proximity loss is not solved: the coil is its turns' current spread over the section.",
                            table=dict(columns=['coil', 'L (uH)', 'R ferrite (mohm)', 'R metal (mohm)',
                                                'R winding (mohm)', 'R (mohm)', 'Q'],
                                       rows=[[a['name'], a['L'] * 1e6, a['R_ferrite'] * 1e3, a['R_metal'] * 1e3,
                                              a['R_winding'] * 1e3, a['R'] * 1e3, a['Q'] if a.get('Q') else '']
                                             for a in r['ac']],
                                       formats=[None, '.3f', '.3g', '.3g', '.3g', '.3g', '.4g'])))
        if r.get('plates'):
            sec.append(dict(title='Plates',
                            table=dict(columns=['z (mm)', 'thickness (mm)', 'mu_r', "mu''", 'sigma (S/m)', 'size (mm)'],
                                       rows=[[p.get('z', 0.0), p.get('t', 0.0), p.get('mur', 1.0), p.get('mur_im', ''),
                                              p.get('sigma', ''),
                                              f"r {p['r']:g}" if 'r' in p else 'x'.join(f"{v:g}" for v in p.get('size', ()))]
                                             for p in r['plates']], formats=['.2f', '.2f', '.0f', '.3g', '.3g', None])))
        head = ', '.join(f"{a} {L[i][i] * 1e6:.3g} uH" for i, a in enumerate(n))
        if pairs:
            i, j = pairs[0]
            head += f"; k {r['k'][i][j]:.3f} ({n[j]} to {n[i]})"
        if r.get('ac'):
            head += '; at ' + f"{r['freq'] / 1e3:g} kHz " + ', '.join(
                f"{a['name']} Q {a['Q']:.3g}" for a in r['ac'] if a.get('Q'))
        return dict(title=f"Coils with ferrite: {', '.join(n)}", summary=head, sections=sec,
                    notes=["The inductances and couplings are the static field's (a metal plate is transparent "
                           "to it). " + ("The frequency's run adds the ferrite's and the metal's loss and the "
                                         "metal's eddy currents; " if r.get('ac') else
                                         "Give a frequency for the loss, R and Q there; ") +
                           "the winding's skin and proximity loss and the ferrite's saturation are not modelled."],
                    footer=f"Elmer, {r.get('tets', 0):,} tetrahedra, {_seconds(r):.0f} s")

    def suggest(self, folder):
        """Each coil's inductance for the circuit model's inductor of the same name (name the coils after them:
        --coil REF:...), and each pair's coupling for the model's coupling of those two (couplings.A+B.k): no model's
        names are assumed. (It had written the wpthb bridge's PRIMARY, SECONDARY and couplings.0.)"""
        r = self._read(folder)
        if not r or r.get('tool') != self.module or not r.get('names'):
            return []
        n, L, k = r['names'], r['L'], r['k']
        sets = [f"{a}.value={L[i][i]:.6g}" for i, a in enumerate(n)]
        sets += [f"couplings.{n[j]}+{n[i]}.k={abs(k[i][j]):.4f}" for i in range(len(n)) for j in range(i)
                 if abs(k[i][j]) >= 1e-3]
        return [dict(to='circuit', label=f"Use these coils in the circuit ({', '.join(n)})",
                     values=dict(set='\n'.join(sets)), need=[],
                     note="the circuit model's inductors of these names and their couplings (a coil named otherwise: "
                          "edit the references in Set); a resonant capacitor that was tuned to the old inductance "
                          "is not retuned")]
