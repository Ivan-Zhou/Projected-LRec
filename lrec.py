import numpy as np
import scipy.sparse as sparse
from scipy.sparse import vstack, hstack
from scipy.sparse.linalg import svds
from scipy.sparse.linalg import inv


def embedded_lirec_items(matrix_train, embeded_matrix=np.empty((0)), lam=80, k_factor=0.33):
    """
    Function used to achieve generalized projected lrec w/o item-attribute embedding
    :param matrix_train: user-item matrix with shape m*n
    :param embeded_matrix: item-attribute matrix with length n (each row represents one item)
    :param lam: parameter of penalty
    :param k_factor: ratio of the latent dimension/number of items
    :return: prediction in sparse matrix
    """
    m = matrix_train.shape[0]
    n = matrix_train.shape[1]
    k = min(int(k_factor * n), m)  # k depends on num items
    matrix_nonzero = matrix_train.nonzero()
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))
    U, sigma, Vt = svds(matrix_input, k=k)
    QV = matrix_input * sparse.csr_matrix(Vt).T
    pre_inv = QV.T.dot(QV) + lam * sparse.identity(k)
    inverse = sparse.csr_matrix(inv(pre_inv))
    sim = inverse.dot(QV.T).dot(matrix_input)
    prediction = QV.dot(sim)
    prediction = prediction[:m, :n]
    prediction[matrix_nonzero] = 0
    return prediction


def embedded_lirec_users(matrix_train, embeded_matrix=np.empty((0)), lam=80, k_factor=0.33):
    """
    Function used to achieve generalized projected lrec w/o item-attribute embedding
    :param matrix_train: user-item matrix with shape m*n
    :param embeded_matrix: user-attribute matrix with length m (each row represents one user)
    :param lam: parameter of penalty
    :param k_factor: ratio of the latent dimension/number of items
    :return: prediction in sparse matrix
    """
    m = matrix_train.shape[0]
    n = matrix_train.shape[1]
    k = min(int(k_factor * m), n)  # k depends on num items
    matrix_nonzero = matrix_train.nonzero()
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = hstack((matrix_input, embeded_matrix))
    U, sigma, Vt = svds(matrix_input, k=k)
    UtQ = sparse.csr_matrix(U).T.dot(matrix_input)
    pre_inv = UtQ.dot(UtQ.T) + lam * sparse.identity(k)
    inverse = sparse.csr_matrix(inv(pre_inv))
    sim = matrix_input.dot(UtQ.T).dot(inverse)
    prediction = sim.dot(UtQ)
    prediction = prediction[:m, :n]
    prediction[matrix_nonzero] = 0
    return prediction
