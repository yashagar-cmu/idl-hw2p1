import numpy as np
from mytorch.nn.imports import *

class Linear:
    def __init__(self, in_features: int, out_features: int, debug=False):
        """
        Initialize the weights and biases with zeros (Hint: check np.zeros method)
        Read the writeup (Hint: Linear Layer Section Table) to identify the right shapes for `W` and `b`.
        """
        self.debug = debug
        self.W = np.zeros(shape=[out_features, in_features])  # out_features x in_features
        self.b = np.zeros(shape=[out_features])  # out_features x 1

        self.in_features = in_features
        self.out_features = out_features

        self.A: Optional[np.ndarray] = None # batch_size x in_features
        self.batch_size: Optional[int] = None
        self.ones: Optional[np.ndarray] = None # batch_size x 1

        self.dLdW: Optional[np.ndarray] = None # out_features x in_features
        self.dLdb: Optional[np.ndarray] = None # out_features x 1

    def forward(self, A: np.ndarray) -> np.ndarray:
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output of linear layer with shape (N, C1)

        Read the writeup (Hint: Linear Layer Section) for implementation details for `Z`
        """
        self.A = A
        assert A.shape[1] == self.in_features, f"A should have {self.in_features} as second dimension, actual={A.shape}"

        if self.batch_size != A.shape[0]:
            self.batch_size = A.shape[0]  # store the batch size parameter of the input A
            # Think how can `self.ones` help in the calculations and uncomment below code snippet.
            self.ones = np.ones((self.batch_size, 1))

        Z = (A @ self.W.T) + (self.ones @ self.b.T) # batch_size x out_features
        assert Z.shape == (self.batch_size, self.out_features)

        return Z

    def backward(self, dldz: np.ndarray) -> np.ndarray:
        """
        :param dldz: Gradient of loss wrt output Z (N, C1)
        :return: Gradient of loss wrt input A (N, C0)

        Read the writeup (Hint: Linear Layer Section) for implementation details below variables.
        """
        dldz_expected = (self.batch_size, self.out_features)
        assert dldz.shape == dldz_expected, f"dLdZ expected shape={dldz_expected}, actual shape= {dldz.shape}"

        dlda: np.ndarray = dldz @ self.W
        assert dlda.shape == (self.batch_size, self.in_features), f"dlda expected shape={(self.batch_size, self.in_features)}, actual shape= {dlda.shape}"

        self.dLdW = dldz.T @ self.A
        assert self.dLdW.shape == (self.out_features, self.in_features), f"dldw expected shape={(self.out_features, self.in_features)}, actual shape={self.dLdW.shape}"

        self.dLdb = dldz.T @ self.ones
        assert self.dLdb.shape == (self.out_features, 1), f"dldb expected shape={(self.out_features, 1)}, actual shape={self.dLdb.shape}"

        if self.debug:
            self.dlda = dlda

        return dlda
