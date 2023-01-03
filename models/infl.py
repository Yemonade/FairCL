from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import cho_solve, cho_factor


class IFBaseClass(ABC):
    """ Abstract base class for influence function """

    @staticmethod
    def set_sample_weight(n, sample_weight=None):
        """ Set the weight of each sample
        Args:
            n: int, the total number of samples
            sample_weight: list, tuple or ndarray, the weight of each sample

        Returns:
            sample_weight, ndarray
        """
        if sample_weight == None:
            sample_weight = np.ones(n)
        else:
            assert len(sample_weight) == n
            sample_weight = np.array(sample_weight)

        return sample_weight

    @staticmethod
    def check_pos_def(M):
        """ Check the matrix M is positive define or not. """
        pos_def = np.all(np.linalg.eigvals(M) > 0)
        print("Hessian positive definite: %s" % pos_def)
        return pos_def

    @staticmethod
    def get_inv_hvp(hessian, vectors, cho=True):
        # TODO: There is a faster way to calculate hvp (influence function in 2017)
        """ Calculate the hessian^{-1}.dot(vectors)
        hessian: shape = (n, n)
        # TODO: find out the shape of input
        vectors: shape = (n, ?)
        """
        if cho:
            return cho_solve(cho_factor(hessian), vectors)
        else:
            hess_inv = np.linalg.inv(hessian)
            # TODO: why the transpose is used here
            return hess_inv.dot(vectors.T)

    @abstractmethod
    def grad(self, x, y, sample_weight=None, l2_reg=False):
        """ Return the sum of all gradients and every individual gradient. """
        raise NotImplementedError



class NN(IFBaseClass):

    def __init__(self, input_dim, l2_reg, n_iter=10000, lr=1e-3, device="cuda:0", seed=None):
        super(NN, self).__init__(l2_reg)