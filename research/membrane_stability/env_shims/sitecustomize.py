'''
Compatibility shim for FEniCS installations that ship UFL as `ufl_legacy` (e.g. Ubuntu's python3-dolfin 2019.2),
while IRENE does `import ufl` (as in the quay.io/fenicsproject/stable Docker image).
Put this folder on PYTHONPATH: `ufl` and all its submodules then resolve to the very same `ufl_legacy` modules that
dolfin uses. Not needed inside the FEniCS Docker image.
'''
import importlib
import importlib.abc
import importlib.util
import sys

try:
    import ufl  # noqa: F401  (a real `ufl` is available: nothing to do)
except ImportError:
    try:
        import ufl_legacy
    except ImportError:
        ufl_legacy = None

    if ufl_legacy is not None:
        class _UflAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "ufl" or fullname.startswith("ufl."):
                    return importlib.util.spec_from_loader(fullname, self)
                return None

            def create_module(self, spec):
                return importlib.import_module("ufl_legacy" + spec.name[3:])

            def exec_module(self, module):
                pass

        sys.meta_path.insert(0, _UflAlias())
