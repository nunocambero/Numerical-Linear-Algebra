import numpy as np

def back_substitution(R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the upper triangular system Rx = b using back substitution.

    Parameters:
    R (np.ndarray): An upper triangular matrix of shape (n, n).
    b (np.ndarray): A right-hand side vector of shape (n,).

    Returns:
    np.ndarray: The solution vector x of shape (n,).
    """
    n = R.shape[0]
    x = np.zeros(n, dtype=b.dtype)

    if R[n-1, n-1] == 0:
        raise ValueError("Matrix is singular. Cannot perform back substitution.")
    x[n-1] = b[n-1] / R[n-1, n-1]

    for i in range(n-2, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= R[i, j] * x[j]
        if R[i, i] == 0:
            raise ValueError("Matrix is singular. Cannot perform back substitution.")
        x[i] /= R[i, i]

    return x

# Example usage:
if __name__ == "__main__":
    R = np.array([[2, 1, -1],
                  [0, 3, 2],
                  [0, 0, 1]], dtype=float)
    b = np.array([8, 13, 3], dtype=float)
    x = back_substitution(R, b)
    print("Solution x:", x)


