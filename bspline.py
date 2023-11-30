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
    p = min(max(p, 1), n - 1)
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

def bspline_basis2(p, n, x):
    m = len(x)
    knots = np.zeros(n + p + 1)

    for i in range(p):
        knots[i] = 0
        knots[n + p - i] = 1

    for i in range(p, n + 1):
        knots[i] = (i - p) / (n - p)

    prev_order = np.zeros((m, n))

    for i in range(m):
        prev_order[i, min(max(int(x[i] * (n - p)), 0), n - p - 1)] = 1

    for q in range(1, p + 1):
        for i in range(m):
            prev_alpha = 0

            for j in range(n - p + q - 1):
                alpha = (x[i] - knots[p - q + 1 + j]) / (knots[p + 1 + j] - knots[p - q + 1 + j])
                alpha_val = alpha * prev_order[i, j]
                beta_val = (1 - alpha) * prev_order[i, j]
                prev_order[i, j] = prev_alpha + beta_val
                prev_alpha = alpha_val

            prev_order[i, n - p + q - 1] = prev_alpha

    return prev_order


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example usage:
    p = 4  # order of the B-spline
    n = 8  # number of control points
    x = np.linspace(0, 1, 80)  # evaluation points

    basis_values = bspline_basis(p, n, x)
    print(basis_values[-1])
    plt.plot(basis_values)
    plt.show()
