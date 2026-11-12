"""kicad2ems's finite-element solvers, each an external program behind its own interface, sharing gmsh geometry:

    palace/    Palace (AWS's 3D FEM): S-parameters as a kicad2ems solver plugin (`--solvers palace`)
    elmer/     Elmer (CSC's multiphysics FEM): DC IR drop of a net's copper (`python -m kicad2ems_fem.elmer.dc`)
    geometry   the board's copper, pads and vias as shapely / gmsh pieces, shared by both
    analyses   the DC, thermal and capacitance tools as kicad2ems analysis plugins: engines of a study's questions

Neither program is a Python dependency: each is found at run time (palace/run.py, elmer/run.py).
"""
__version__ = '0.2.1'
