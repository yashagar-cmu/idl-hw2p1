import numpy as np
from mytorch.nn.imports import *
from scipy.special import erf
import scipy


class Activation:
    def __init__(self):
        self.A: Optional[np.ndarray] = None
        self.Z: Optional[np.ndarray] = None

    def forward(self, Z):
        raise NotImplementedError

    def backward(self, Z):
        raise NotImplementedError


### No need to modify Identity class
class Identity(Activation):
    """
    Identity activation function.
    """

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.A = Z
        return self.A

    def backward(self, dLdA: np.ndarray) -> np.ndarray:
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        assert dLdA.shape == self.A.shape, f"dlda expected shape={self.A.shape}, actual={dLdA.shape}"

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ
        return dLdZ


class ActivationElementwise(Activation):

    @staticmethod
    def operate(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def dadz(a: np.ndarray, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
        :return: Output returns the computed output A (N samples, C features).
        """
        self.Z = Z
        self.A = self.operate(Z)
        assert self.A.shape == Z.shape

        return self.A

    def backward(self, dLdA: np.ndarray) -> np.ndarray:
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        assert dLdA.shape == self.A.shape, f"dlda expected shape={self.A.shape}, actual={dLdA.shape}"

        dAdZ = self.dadz(self.A, self.Z)
        assert dAdZ.shape == self.A.shape

        dLdZ = dLdA * dAdZ
        return dLdZ


class Sigmoid(ActivationElementwise):
    """
    Sigmoid activation function.

    TODO:
    On same lines as above, create your own mytorch.nn.Sigmoid!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: Sigmoid Section) for further details on Sigmoid forward and backward expressions.
    """

    @staticmethod
    def operate(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dadz(a: np.ndarray, _) -> np.ndarray:
        return a - a * a


class Tanh(ActivationElementwise):
    """
    Tanh activation function.

    TODO:
    On same lines as above, create your own mytorch.nn.Tanh!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: Tanh Section) for further details on Tanh forward and backward expressions.
    """

    @staticmethod
    def operate(x: np.ndarray) -> np.ndarray:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def dadz(a: np.ndarray, _) -> np.ndarray:
        return 1 - a * a


class ReLU(ActivationElementwise):
    """
    ReLU (Rectified Linear Unit) activation function.

    TODO:
    On same lines as above, create your own mytorch.nn.ReLU!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: ReLU Section) for further details on ReLU forward and backward expressions.
    """

    @staticmethod
    def operate(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def dadz(a: np.ndarray, _) -> np.ndarray:
        return np.where(a > 0, 1, 0)


class GELU(ActivationElementwise):
    """
    GELU (Gaussian Error Linear Unit) activation function.

    TODO:
    On same lines as above, create your own mytorch.nn.GELU!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: GELU Section) for further details on GELU forward and backward expressions.
    Note: Feel free to save any variables from gelu.forward that you might need for gelu.backward.
    """

    @staticmethod
    def operate(z: np.ndarray) -> np.ndarray:
        return 0.5 * z * (1.0 + erf(z / np.sqrt(2)))

    @staticmethod
    def dadz(_, z: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + erf(z / np.sqrt(2))) + z / np.sqrt(2 * np.pi) * np.exp(-z * z / 2)


class Swish(Activation):
    """
    Swish activation function.

    TODO:
    On same lines as above, create your own Swish which is a torch.nn.SiLU with a learnable parameter (beta)!
    Define 'forward' function.
    Define 'backward' function.
    Read the writeup (Hint: Swish Section) for further details on Swish forward and backward expressions.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta: float = beta
        self.inverse_term: Optional[np.ndarray] = None
        self.dLdbeta: Optional[float] = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
                :param Z: Batch of data Z (N samples, C features) to apply activation function to input Z.
                :return: Output returns the computed output A (N samples, C features).
                """
        self.Z = Z
        self.inverse_term = (1 / (1 + np.exp(-self.beta * Z)))
        self.A = Z * self.inverse_term
        assert self.A.shape == Z.shape

        return self.A

    def backward(self, dLdA: np.ndarray) -> np.ndarray:
        """
        :param dLdA: Gradient of loss wrt post-activation output (a measure of how the output A affect the loss L)
        :return: Gradient of loss with respect to pre-activation input (a measure of how the input Z affect the loss L)
        """
        assert dLdA.shape == self.A.shape, f"dlda expected shape={self.A.shape}, actual={dLdA.shape}"

        dAdZ = self.inverse_term + self.beta * self.Z * self.inverse_term * (1 - self.inverse_term)
        assert dAdZ.shape == self.A.shape

        dAdB = self.Z * self.Z * self.inverse_term * (1 - self.inverse_term)
        self.dLdbeta = np.sum(dLdA * dAdB, axis=(0, 1))

        dLdZ = dLdA * dAdZ
        return dLdZ


class Softmax(Activation):
    """
    Softmax activation function.

    ToDO:
    On same lines as above, create your own mytorch.nn.Softmax!
    Complete the 'forward' function.
    Complete the 'backward' function.
    Read the writeup (Hint: Softmax Section) for further details on Softmax forward and backward expressions.
    Hint: You read more about `axis` and `keep_dims` attributes, helpful for future homeworks too.
    """

    @staticmethod
    def soft(z: np.ndarray) -> np.ndarray:
        augz = z - np.max(z)
        exps = np.exp(augz)
        s = exps / np.sum(exps)
        assert s.shape == z.shape

        return s

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Remember that Softmax does not act element-wise.
        It will use an entire row of Z to compute an output element.
        Note: How can we handle large overflow values? Hint: Check numerical stability.
        """
        self.Z = Z
        self.A = np.apply_along_axis(Softmax.soft, axis=1, arr=Z)
        return self.A

    def backward(self, dLdA: np.ndarray) -> np.ndarray:
        assert dLdA.shape == self.A.shape, f"dlda expected shape={self.A.shape}, actual={dLdA.shape}"

        dLdZ = np.zeros_like(dLdA)

        for i in range(len(self.A)):
            a = self.A[i]

            J = -np.outer(a, a) # C x C
            np.fill_diagonal(J, a*(1-a))

            dLdZ[i] = dLdA[i] @ J    # 1 x C

        return dLdZ
