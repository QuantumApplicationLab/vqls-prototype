# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An abstract class for variational linear systems solvers."""

from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np

from qiskit import QuantumCircuit
from qiskit_algorithms.variational_algorithm import VariationalResult


class VariationalLinearSolverResult(VariationalResult):
    """A base class for linear systems results using variational methods

    The  linear systems variational algorithms return an object of the type
    ``VariationalLinearSystemsResult`` with the information about the
    solution obtained.
    """

    def __init__(self) -> None:
        super().__init__()
        # todo: it seems super()._optimizer_time does not get populated by the prototype
        # todo: write tests
        self._cost_function_evals: int | None = None
        self._state: Union[QuantumCircuit, np.ndarray] | None = None
        self._vector: Union[np.ndarray, None] | None = None
        self._transpile_time_local: float | None = None
        self._quantum_time_wall: float | None = None
        self._qpu_job_execution_time: float | None = None
        self._classical_opt_time: float | None = None

    @property
    def cost_function_evals(self) -> Optional[int]:
        """Returns number of cost optimizer evaluations"""
        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:
        """Sets number of cost function evaluations"""
        self._cost_function_evals = value

    @property
    def state(self) -> Union[QuantumCircuit, np.ndarray]:
        """return either the circuit that prepares the solution or the solution
        as a vector"""
        return self._state

    @state.setter
    def state(self, state: Union[QuantumCircuit, np.ndarray]) -> None:
        """Set the solution state as either the circuit that prepares
           it or as a vector.

        Args:
            state: The new solution state.
        """
        self._state = state

    @property
    def vector(self) -> np.ndarray:
        """returns the actual solution of the linear system"""
        return self._vector

    @vector.setter
    def vector(self, vector: np.ndarray) -> None:
        """Set the solution vector of the linear system

        Args:
            vector: The solution vector.
        """
        self._vector = vector

    @property
    def transpile_time_local(self) -> Optional[int]:
        """Returns local transpilation time in seconds."""
        return self._transpile_time_local

    @transpile_time_local.setter
    def transpile_time_local(self, value: int) -> None:
        """Sets local transpilation time in seconds."""
        self._transpile_time_local = value

    @property
    def quantum_time_wall(self) -> Optional[int]:
        """Returns quantum execution wall time in seconds."""
        return self._quantum_time_wall

    @quantum_time_wall.setter
    def quantum_time_wall(self, value: int) -> None:
        """Sets quantum execution wall time in seconds."""
        self._quantum_time_wall = value

    @property
    def qpu_job_execution_time(self) -> Optional[int]:
        """Returns quantum execution time as reported by the QPU job in seconds."""
        return self._qpu_job_execution_time

    @qpu_job_execution_time.setter
    def qpu_job_execution_time(self, value: int) -> None:
        """Sets quantum execution time as reported by the QPU job in seconds"""
        self._qpu_job_execution_time = value

    @property
    def classical_opt_time(self) -> Optional[int]:
        """Returns time taken for classical optimization in seconds."""
        return self._classical_opt_time

    @classical_opt_time.setter
    def classical_opt_time(self, value: int) -> None:
        """Sets time taken for classical optimization in seconds."""
        self._classical_opt_time = value


class VariationalLinearSolver(ABC):
    """An abstract class for linear system solvers in Qiskit."""

    @abstractmethod
    def solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit],
        vector: Union[np.ndarray, QuantumCircuit],
    ) -> VariationalLinearSolverResult:
        """Solve the system and compute the observable(s)

        Args:
            matrix: The matrix specifying the system, i.e. A in Ax=b.
            vector: The vector specifying the right hand side of the equation in Ax=b.

        Returns:
            The result of the linear system.
        """
        raise NotImplementedError
