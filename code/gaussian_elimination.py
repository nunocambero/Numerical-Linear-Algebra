import numpy as np

def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Gaussian elimination to convert the system Ax = b into an upper triangular system Rx = b'.

    Parameters:
    A (np.ndarray): A square matrix of shape (n, n).
    b (np.ndarray): A right-hand side vector of shape (n,).

    Returns:
    np.ndarray: An upper triangular matrix R of shape (n, n).
    np.ndarray: The modified right-hand side vector b' of shape (n,).
    """
    n = A.shape[0]
    A = A.astype(float)  # Ensure we are working with floats to avoid integer division
    b = b.astype(float)

    for k in range(n-1):
        d = 1.0 / A[k, k]
        for i in range(k+1, n):
            factor = A[i, k] * d
            b[i] -= factor * b[k]
            A[i,:] -= factor * A[k, :]
    return A, b
# Example usage:
if __name__ == "__main__":
    A = np.array([[2, 1, -1],
                  [-3, -1, 2],
                  [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    R, b_prime = gaussian_elimination(A, b)
    print("Upper Triangular Matrix R:\n", R)
    print("Modified Right-Hand Side b':\n", b_prime)