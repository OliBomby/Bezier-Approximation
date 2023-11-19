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
    u = np.pad(np.linspace(0, 1, n + 1 - p), (p, 2), constant_values=(0, 1))  # knot vector
    basis = np.empty((p + 1, len(x), n + p))
    xb = x[:, None]

    for c in range(p + 1):
        if c == 0:
            basis[c] = np.where(xb != 1, np.where((u[None, :-1] <= xb) & (xb < u[None, 1:]), 1, 0), np.where(u[1:] == 1, 1, 0)[None, :])
        else:
            alpha = np.nan_to_num((xb - u[None, :n + p - c]) / (u[c:n + p] - u[:n + p - c])[None, :])
            beta = np.nan_to_num((u[None, c + 1:n + p + 1] - xb) / (u[c + 1:n + p + 1] - u[1:n + p - c + 1])[None, :])
            basis[c, :, :n + p - c] = alpha * basis[c - 1, :, :n + p - c] + beta * basis[c - 1, :, 1:n + p - c + 1]

    return basis[-1, :, :n]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage:
    n = 5  # number of control points
    p = 2  # order of the B-spline

    eval_points = np.linspace(0, 1, 80)  # evaluation points
    basis_values = bspline_basis(p, n, eval_points)
    plt.plot(basis_values)
    plt.show()
