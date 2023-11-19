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
    u = np.pad(np.linspace(0, 1, n + 1 - p), (p, p), constant_values=(0, 1))  # knot vector
    xb = x[:, None]
    prev_order = np.where(xb != 1, np.where((u[None, p:-p - 1] <= xb) & (xb < u[None, p + 1:-p]), 1, 0), np.where((u[p:-p - 1] < 1) & (u[p + 1:-p] == 1), 1, 0)[None, :])

    for c in range(1, p + 1):
        alpha = (xb - u[None, p - c + 1:n]) / (u[p + 1:n + c] - u[p - c + 1:n])[None, :]
        beta = (u[None, p + 1:n + c] - xb) / (u[p + 1:n + c] - u[p - c + 1:n])[None, :]
        order = np.zeros((len(x), n - p + c))
        order[:, 1:] += alpha * prev_order
        order[:, :-1] += beta * prev_order
        prev_order = order

    return prev_order


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage:
    p = 2  # order of the B-spline
    n = 5  # number of control points
    x = np.linspace(0, 1, 80)  # evaluation points

    basis_values = bspline_basis(p, n, x)
    print(basis_values[-1])
    plt.plot(basis_values)
    plt.show()
