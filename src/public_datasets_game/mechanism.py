from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt


class Mechanism(ABC):
    @abstractmethod
    def __call__(
        self, contributions: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        raise NotImplementedError()


class PrivateFunding(Mechanism):
    def __call__(self, contributions):
        return contributions.sum(axis=0)


class QuadraticFundng(Mechanism):
    def __call__(self, contributions):
        h = (contributions**0.5).sum(axis=0)
        h = h**2
        return h
