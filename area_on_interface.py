"""Some algorithms for calculating the area of the interface."""

from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
import sys


def read_coordinates(data_path):
    """Read in monomers.

    Args:
        data_path (str or Path): path to the data.
    Returns:
        List of 2D numpy arrays with x,y,z coordinates in columns."""
    monomers = []
    m = []
    with open(data_path, 'r') as f:
        for l in f:
            l = l.strip().split()
            if l[0] == 'ATOM':
                m.append([float(y) for y in l[6:]])
            if 'TER' in l[0] or 'ENDMD' in l[0]:
                if len(m)>0:
                    monomers.append(np.array(m))
                m = []
    return monomers

# project on a sphere
def project_on_sphere(m, R):
    N = np.sqrt((m**2).sum(axis=1))
    M = R * m / N[:, np.newaxis]
    return M, N

def plot3d(W, show=True, *args, **kwds):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(W[:,0], W[:,1], W[:,2], *args, **kwds)
    if show:
        plt.show()

def half_of_3D_hull_area(m, R):
    M, N = project_on_sphere(m, R)
    hull = ConvexHull(M[N >= R])
    return hull.area / 2

# another approach:
# 0. project unto the sphere
# 1. get the mean vector = 2 * this vector is the normal to the sphere.
# 2. get two vectors that span the tangent plane at the normal vector
# 3. project all vectors above the sphere unto the tangent plane
# 4. compute the convex hull of these projections and report its volume (which is 2D).

def volume_of_2D_hull(m, R):
    M, N = project_on_sphere(m, R)
    grad = 2*M.mean(axis=0)[:, np.newaxis].T
    U,S,V= np.linalg.svd(grad, full_matrices=True)
    kernel = V[:,1:]
    M2 = m[N >= R] - grad # non-projected vectors above the plane.
    projections = np.linalg.lstsq(kernel, M2.T, rcond=None)
    hull2d = ConvexHull(projections[0].T)
    return hull2d.volume

def plot_2d_hull(X, hull2d, show=True):
    import matplotlib.pyplot as plt
    plt.scatter(X[:,0], X[:,1])
    plt.plot(X[hull2d.vertices,0], X[hull2d.vertices,1], 'r--', lw=2)
    plt.plot(X[hull2d.vertices,0], X[hull2d.vertices,1], 'ro')
    if show:
        plt.show()

def plot_res(res, show=True):
    import matplotlib.pyplot as plt
    plt.scatter(res[:,0], res[:,1])
    if show:
        plt.show()


if __name__ == "__main__":
    R, path = sys.argv[1:3]
    R = float(R)
    data_path = Path(path).expanduser()
    monomers = read_coordinates(data_path)
    res = np.array([(half_of_3D_hull_area(m,R), volume_of_2D_hull(m,R)) for m in monomers])
    np.savetxt("areas.out", res, delimiter=" ")
