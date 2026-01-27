"""This module defines the estimator run builder class."""

from typing import Union, List, Dict, Any
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import Estimator, PrimitiveJob
from qiskit_aer.primitives import Estimator as aer_Estimator
from qiskit_aer.primitives import EstimatorV2 as aer_EstimatorV2
from qiskit_ibm_runtime import Estimator as ibm_runtime_Estimator
from qiskit_ibm_runtime import EstimatorV2 as ibm_runtime_EstimatorV2
from qiskit_ibm_runtime import RuntimeJobV2
from .base_run_builder import BasePrimitiveRunBuilder

EstimatorValidType = Union[
    Estimator,
    aer_Estimator,
    aer_EstimatorV2,
    ibm_runtime_Estimator,
    ibm_runtime_EstimatorV2,
]


class EstimatorRunBuilder(BasePrimitiveRunBuilder):  # pylint: disable=abstract-method
    """
    A class to build and configure estimator runs based on their provenance and options.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        estimator: EstimatorValidType,
        circuits: List[QuantumCircuit],
        observables: List[SparsePauliOp],
        parameter_sets: List[List[float]],
        options: Dict[str, Any],
    ):
        """
        Initializes the EstimatorRunBuilder with the given estimator, circuits, observables,
        parameter sets, and options.
        """
        super().__init__(estimator, circuits, parameter_sets, options)
        self.observables = observables

    def _select_run_builder(self) -> Union[PrimitiveJob, RuntimeJobV2]:
        builders = {
            ("qiskit", "Estimator"): self._build_native_qiskit_run,
            ("qiskit_aer", "EstimatorV2"): self._build_v2_run,
            ("qiskit_aer", "Estimator"): self._build_v1_run,
            ("qiskit_ibm_runtime", "EstimatorV2"): self._build_v2_run,
            ("qiskit_ibm_runtime", "EstimatorV1"): self._build_v1_run,
            ("qiskit", "BackendEstimatorV2"): self._build_v2_run,   #todo: more robust implementation of this
        }
        try:
            return builders[self.provenance]
        except KeyError as err:
            raise NotImplementedError(
                f"{self.__class__.__name__} not compatible with {self.provenance}."
            ) from err

    def _build_native_qiskit_run(self) -> PrimitiveJob:
        """Builds a run function for a standard qiskit Estimator."""
        return self.primitive.run(
            self.circuits,
            self.observables,
            self.parameter_sets,
            shots=self.shots,
            seed=self.seed,
        )

    def _build_v2_run(self) -> Union[PrimitiveJob, RuntimeJobV2]:
        """Builds a run function for qiskit-aer and qiskit-ibm-runtime EstimatorV2."""
        backend = self.primitive._backend  # pylint: disable=protected-access
        optimization_level = 1
        pm = generate_preset_pass_manager(optimization_level, backend)
        pubs = []
        for qc, obs, param in zip(self.circuits, self.observables, self.parameter_sets):
            isa_circuit = pm.run(qc)
            isa_obs = obs.apply_layout(isa_circuit.layout)
            pubs.append((isa_circuit, isa_obs, param))
        return self.primitive.run(pubs)
