'''
Make IRENE importable from this folder (import this module first in every script).
IRENE_ROOT: root of the IRENE repository (default: /home/fenics/shared, as in IRENE's Docker setup).
Search order: this folder (own runtime_arguments / switch_problem / parameters), IRENE's modules, the add-on modules
(stability), IRENE's steady_state/<no_flow or flow> (function_spaces, variational problems, ...).
IRENE reads 'parameters_bc_<problem>.csv' from the working directory, so the working directory is set to this folder
(the mesh and output paths on the command line are made absolute first).
'''
import os
import sys

IRENE_ROOT = os.environ.get("IRENE_ROOT", "/home/fenics/shared")
HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_MODULES = os.path.abspath(os.path.join(HERE, "..", "..", "modules"))

for position, path in enumerate([os.path.join(IRENE_ROOT, "modules"), ADDON_MODULES], start=1):
    if path not in sys.path:
        sys.path.insert(position, path)
# the IRENE solver folder with the same name as this folder (steady_state/no_flow or steady_state/flow)
sys.path.append(os.path.join(IRENE_ROOT, "steady_state", "flow"))   # IRENE's flow solver (this folder shadows its function_spaces)

for k in (2, 3):
    if len(sys.argv) > k and not sys.argv[k].startswith("-"):
        sys.argv[k] = os.path.abspath(sys.argv[k])
os.chdir(HERE)
