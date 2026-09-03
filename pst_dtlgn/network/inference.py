from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class LearnedCircuitBase(ABC):

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def signal_values(self) -> tuple[int, ...]:
        ...

    @staticmethod
    @abstractmethod
    def round_fn(x: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def num_gates(self) -> int:
        ...
