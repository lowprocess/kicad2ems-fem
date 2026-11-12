"""
kicad2ems-fem's tools as kicad2ems analysis plugins (the 'kicad2ems.analyses' entry points of pyproject.toml): they
answer questions other engines answer too (kicad2ems.study), in 3D, so a study can ask the same of two programs
and lay the answers side by side.

    elmer_dc        a rail's DC drop                 (python -m kicad2ems_fem.elmer.dc)
    elmer_thermal   the board's temperature          (python -m kicad2ems_fem.elmer.thermal)
    elmer_cap       the capacitance between nets     (python -m kicad2ems_fem.elmer.cap)
    palace_cap      the same, by Palace              (python -m kicad2ems_fem.palace.cap)

Each one's settings are its command line's options, read from the tool's own parser, so a new option of a tool
appears in a front end without a change here. Only this module imports kicad2ems.api.analysis.
"""
import json
import os
import re
import sys

from kicad2ems.api.analysis import Analysis

API = (2, 6)
ORDER = ('Nets and loads', 'Heat', 'Air and board', 'Mesh', 'Solver')
GROUPS = dict(net='Nets and loads', source='Nets and loads', load='Nets and loads', copper='Nets and loads',
              heat='Heat', heat_from='Heat', htc='Air and board', ambient='Air and board', k_board='Air and board',
              far='Air and board', margin='Air and board', region='Air and board',
              h='Mesh', h_fine='Mesh', hmax='Mesh', plating='Mesh', thick='Mesh', order='Solver', np='Solver')


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
