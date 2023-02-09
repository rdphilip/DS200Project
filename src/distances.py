import numpy as np


def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    output = np.empty((M, N))
    for i in range(0, M):
        for j in range(0, N):
            d = np.sum(np.square(X[i] - Y[j]))
            output[i, j] = np.sqrt(d)
    return output




def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    output = np.empty((M, N))
    for i in range(0, M):
        for j in range(0, N):
            d = np.sum(np.abs(X[i] - Y[j]))
            output[i, j] = d
    return output


def cosine_distances(X, Y):
    """Compute pairwise Cosine distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    output = np.empty((M, N))
    for i in range(0, M):
        for j in range(0, N):
            s = np.dot(X[i, :], Y[j, :])
            s = s / (np.linalg.norm(X[i, :]) * np.linalg.norm(Y[j, :]) + 10**-8)
            output[i, j] = 1 - s
    return output
