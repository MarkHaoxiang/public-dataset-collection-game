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
