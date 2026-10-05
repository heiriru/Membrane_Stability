'''helpers to store FE fields as vertex values (npz) and to plot them with matplotlib'''
import numpy as np
from fenics import FunctionSpace, project, Function
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def vorticity(u, mesh):
    Q = FunctionSpace(mesh, "P", 1)
    return project(u[1].dx(0) - u[0].dx(1), Q, solver_type="mumps")


def vertex_values(f, mesh):
    '''values of a scalar or vector Function at the mesh vertices, shape (n_vertices,) or (n_vertices, dim)'''
    values = f.compute_vertex_values(mesh)
    n = mesh.num_vertices()
    return values.reshape(-1, n).T.squeeze() if values.size > n else values


def mesh_arrays(mesh):
    return dict(points=mesh.coordinates().copy(), cells=mesh.cells().copy())


def triangulation(data):
    return mtri.Triangulation(data["points"][:, 0], data["points"][:, 1], data["cells"])


def plot_field(ax, tri, values, xlim, ylim, cmap="RdBu_r", symmetric=True, vmax=None, levels=41, circle=(0.0, 0.0, 0.5)):
    if vmax is None:
        inside = ((tri.x > xlim[0]) & (tri.x < xlim[1]) & (tri.y > ylim[0]) & (tri.y < ylim[1]))
        vmax = np.abs(values[inside]).max() if symmetric else values[inside].max()
    vmin = -vmax if symmetric else values.min()
    clipped = np.clip(values, vmin, vmax)
    cs = ax.tricontourf(tri, clipped, levels=np.linspace(vmin, vmax, levels), cmap=cmap, extend="both")
    ax.add_patch(plt.Circle(circle[:2], circle[2], color="0.3", zorder=3))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    return cs
