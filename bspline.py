import numpy as np


def bspline_basis(p, n, x):
    """
    Calculate the B-spline basis functions at specified evaluation points.

    Parameters:
        p: int, order of the B-spline
        n: int, number of control points
        x: array, evaluation points

    Returns:
        array, matrix array of B-spline basis function values at x
    """
    t = np.pad(np.linspace(0, 1, n + 1 - p), (p, 2), constant_values=(0, 1))  # knot vector
    basi = np.empty((p + 1, len(x), n + p))
    xb = x[:, None]

    for c in range(p + 1):
        if c == 0:
            basi[c] = np.where(xb != 1, np.where((t[None, :-1] <= xb) & (xb < t[None, 1:]), 1, 0), np.where(t[1:] == 1, 1, 0)[None, :])
        else:
            alpha = np.nan_to_num((xb - t[None, :n + p - c]) / (t[c:n + p] - t[:n + p - c])[None, :])
            beta = np.nan_to_num((t[None, c + 1:n + p + 1] - xb) / (t[c + 1:n + p + 1] - t[1:n + p - c + 1])[None, :])
            basi[c, :, :n + p - c] = alpha * basi[c - 1, :, :n + p - c] + beta * basi[c - 1, :, 1:n + p - c + 1]

    return basi[-1, :, :n]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage:
    n = 5  # number of control points
    p = 2  # order of the B-spline

    eval_points = np.linspace(0, 1, 80)  # evaluation points
    basis_values = bspline_basis(p, n, eval_points)
    plt.plot(basis_values)
    plt.show()
