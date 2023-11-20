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
    xb = x[:, None]
    u = np.pad(np.linspace(0, 1, n + 1 - p), (p, p), constant_values=(0, 1))  # knot vector
    prev_order = np.zeros((len(x), n - p))
    prev_order[np.arange(len(x)), np.clip((x * (n - p)).astype(np.int32), 0, n - p - 1)] = 1

    for c in range(1, p + 1):
        alpha = (xb - u[None, p - c + 1:n]) / (u[p + 1:n + c] - u[p - c + 1:n])[None, :]
        order = np.zeros((len(x), n - p + c))
        order[:, 1:] += alpha * prev_order
        order[:, :-1] += (1 - alpha) * prev_order
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
