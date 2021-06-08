import numpy as np


# Algorithm 16
def CARATHEODORY(P, u):
    """
    :param P: a numpy array P of size n (points) times d (dimensions)
    :param u: weights function
    :return: a Caratheodory set (S, w)
    Computation time: O(n^2 d^2)
    """
    d = P.shape[1]
    while True:
        n = np.count_nonzero(u)
        u_plus_idx = u > 0

        if n <= d + 1:
            return P, u

        A = P[u_plus_idx]
        P1 = np.outer(A[0], np.ones(A.shape[0] - 1))
        A = A[1:].T - P1

        _, _, V = np.linalg.svd(A)
        v = V[-1]
        v = np.insert(v, 0, -1 * sum(v))
        v_plus_idx = v > 0
        alpha = np.min(u[u_plus_idx][v_plus_idx] / v[v_plus_idx])

        w = u[u_plus_idx] - alpha * v
        w[np.argmin(w)] = 0.0
        w_plus_idx = w > 0
        S = P[w_plus_idx]
        w = w[w_plus_idx]
        u = w
        P = S


# Algorithm 1
def FAST_CARATHEODORY_SET(P, u, k):
    d = P.shape[1]
    while True:
        n = np.count_nonzero(u)
        u_plus_idx = u > 0
        P = P[u_plus_idx]
        u = u[u_plus_idx]

        if P.shape[0] <= d + 1:
            return P, u
    # To be continued


# Algorithm 2
def CARATHEODORY_MATRIX(A, k):
    """
    :param A: A matrix in R^{n * d}, whose row vectors are a_i.
    :param k: An integer in range(1, n+1) for numerical accuracy/speed trade-off
    :return: A matrix S in R^{(d^2+1) * d} satisfying A.T @ A = S.T @ S
    """
    n, d = A.shape
    u = np.ones(n) / n
    P = np.einsum("ij,jk->ijk", A, A).reshape(n, -1)
    C, w = FAST_CARATHEODORY_SET(P, u, k)
    idx = w > 0
    S = np.einsum("i,ij->ij", np.sqrt(n * w), A)
    S = S[idx]
    return S


# Algorithm 5
def LMS_CORESET(A, b, m, k):
    """
    This function computes a coreset for LMS solvers that use m-fold cross validation.
    The result satisfies ||Ax - b|| = ||Cx - y||.
    :param A: A matrix in R^{n * d}
    :param b: A vector in R^n
    :param m: A number of cross-validation folds
    :param k: An integer in range(1, n+1) for numerical accuracy/speed trade-off
    :return: A matrix C in R^{O(md^2)*d} and a vector y in R^n
    """
    d = A.shape[1]
    A_prime = np.append(A, b, axis=1)
    batch = A_prime.shape[0] // m

    S = CARATHEODORY_MATRIX(A_prime[:batch], k)
    S = S.T
    for i in range(1, m):
        Ai = A_prime[i * batch:(i + 1) * batch]
        Si = CARATHEODORY_MATRIX(Ai, k)
        S = np.concatenate((S, Si.T))
    S = S.T
    C, y = S[:, :d], S[:, -1]
    return C, y
