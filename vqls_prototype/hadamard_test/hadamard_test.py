"""Hadamard test."""

from typing import Optional, List, Union, Tuple
from qiskit import QuantumCircuit
from qiskit_algorithms.exceptions import AlgorithmError
from qiskit.quantum_info import Operator
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import numpy.typing as npt

from qiskit.primitives.estimator import EstimatorResult
from qiskit.primitives.containers import PrimitiveResult
from vqls_prototype.primitives_run_builder import EstimatorRunBuilder


class BatchHadamardTest:
    r"""Class that execute batches of Hadamard Test"""

    def __init__(self, hdmr_list: List):
        """Create a single container that computes many hadamard tests

        Args:
            hdrm_list (List): A list of HadamardTest instances
        """
        self.hdmr_list = hdmr_list
        self.circuits = [c for hdmr in hdmr_list for c in hdmr.circuits]
        self.observable = [hdmr.observable for hdmr in hdmr_list for _ in range(2)]
        self.post_processing = hdmr_list[0].post_processing
        self.shots = hdmr_list[0].shots

    def get_values(self, primitive, parameter_sets: List, zne_strategy=None, return_raw_result: bool = False) -> Union[
        List, Tuple[List, object, object]]:
        """Compute the value(s) of the test.

        If return_raw_result is True, return a tuple (values_array, primitive_result_object, job_object).
        Otherwise return values_array (existing behaviour).
        """
        ncircuits = len(self.circuits)
        all_parameter_sets = [parameter_sets] * ncircuits

        estimator_run_builder = EstimatorRunBuilder(
            primitive,
            self.circuits,
            self.observable,
            all_parameter_sets,
            options={"shots": self.shots},
        )

        try:
            if zne_strategy is None:
                job = estimator_run_builder.build_run()
            else:
                job = primitive.run(
                    self.circuits,
                    self.observable,
                    [parameter_sets] * ncircuits,
                    shots=self.shots,
                    zne_strategy=zne_strategy,
                )

            primitive_result = job.result()
            results = self.post_processing(primitive_result)
        except Exception as exc:
            raise AlgorithmError(
                "The primitive to evaluate the Hadammard Test failed!"
            ) from exc

        results *= np.array([1.0, 1.0j] * (ncircuits // 2))
        results = results.reshape(-1, 2).sum(1).reshape(-1)

        if return_raw_result:
            return results, primitive_result, job
        return results


class HadamardTest:
    r"""Class to compute the Hadamard Test"""

    def __init__(
        self,
        operators: Union[QuantumCircuit, List[QuantumCircuit]],
        use_barrier: Optional[bool] = False,
        apply_control_to_operator: Optional[Union[bool, List[bool]]] = True,
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = False,
        shots: Optional[int] = 4000,
    ):
        r"""Create the quantum circuits required to compute the hadamard test:

        .. math::

            \\langle \\Psi | U | \\Psi \\rangle

        Args:
            operators (Union[QuantumCircuit, List[QuantumCircuit]]): quantum circuit or
                list of quantum circuits representing the U.
            use_barrier (Optional[bool], optional): introduce barriers in the
                description of the circuits. Defaults to False.
            apply_control_to_operator (Optional[bool], optional): Apply control operator to the
                input quantum circuits. Defaults to True.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to create
                |Psi> from |0>. If None, assume that the qubits are alredy in Psi.
            apply_measurement (Optional[bool], optional): apply explicit measurement.
                Defaults to False.

        """

        if isinstance(operators, QuantumCircuit):
            operators = [operators]

        if not isinstance(apply_control_to_operator, list):
            apply_control_to_operator = [apply_control_to_operator] * len(operators)

        if apply_control_to_operator[0]:
            self.num_qubits = operators[0].num_qubits + 1
            if apply_initial_state is not None:
                if apply_initial_state.num_qubits != operators[0].num_qubits:
                    raise ValueError(
                        "The operator and the initial state circuits \
                            have different numbers of qubits"
                    )
        else:
            self.num_qubits = operators[0].num_qubits
            if apply_initial_state is not None:
                if apply_initial_state.num_qubits != operators[0].num_qubits - 1:
                    raise ValueError(
                        "The operator and the initial state circuits \
                            have different numbers of qubits"
                    )

        # classical bit for explicit measurement
        self.num_clbits = 1

        # build the circuits
        self.circuits = self._build_circuit(
            operators,
            use_barrier,
            apply_control_to_operator,
            apply_initial_state,
            apply_measurement,
        )

        # number of circuits required
        self.ncircuits = len(self.circuits)

        # compute the observables
        self.observable = self._build_observable()

        # init the expectation
        self.expect_ops = None

        # number of shots
        self.shots = shots

    def _build_circuit(
        self,
        operators: List[QuantumCircuit],
        use_barrier: bool,
        apply_control_to_operator: List[bool],
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = False,
    ) -> List[QuantumCircuit]:
        """build the quantum circuits

        Args:
            operators (List[QuantumCircuit]): quantum circuit or list of quantum circuits
                representing the U.
            use_barrier (bool): introduce barriers in the description of the circuits.
            apply_control_to_operator (bool): Apply control operator to the
                input quantum circuits.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to
                create |Psi> from |0>. If None, assume that the qubits are alredy in Psi.
                Defaults to None.
            apply_measurement (Optional[bool], optional): apply explicit measurement.
                Defaults to False.

        Returns:
            List[QuantumCircuit]: List of quamtum circuits required to compute the Hadammard Test.
        """
        circuits = []

        for imaginary in [False, True]:
            if apply_measurement:
                circuit = QuantumCircuit(self.num_qubits, self.num_clbits)
            else:
                circuit = QuantumCircuit(self.num_qubits)

            if apply_initial_state is not None:
                circuit.append(apply_initial_state, list(range(1, self.num_qubits)))

            if use_barrier:
                circuit.barrier()

            # hadadmard gate on ctrl qbit
            circuit.h(0)

            # Sdg on ctrl qbit
            if imaginary:
                circuit.sdg(0)

            if use_barrier:
                circuit.barrier()

            # matrix circuit
            for operator, ctrl in zip(operators, apply_control_to_operator):
                if ctrl:
                    circuit.append(operator.control(1), list(range(0, self.num_qubits)))
                else:
                    circuit.append(operator, list(range(0, self.num_qubits)))

            if use_barrier:
                circuit.barrier()

            # hadamard on ctrl circuit
            circuit.h(0)

            # measure
            if apply_measurement:
                circuit.measure(0, 0)

            circuits.append(circuit)

        return circuits

    def _build_observable(self) -> List[Operator]:
        """Create the operator to measure |1> on the control qubit.

        Returns:
            Lis[TensoredOp]: List of two observables to measure
                |1> on the control qubit I^...^I^|1><1|
        """

        proba_0 = "I" * self.num_qubits
        proba_1 = "I" * (self.num_qubits - 1) + "Z"
        one_op_ctrl = SparsePauliOp(
            [proba_0, proba_1], np.array([0.5 + 0.0j, -0.5 + 0.0j])
        )
        return one_op_ctrl

    def post_processing(self, estimator_result) -> npt.NDArray[np.cdouble]:
        """Post process the measurement values

        Args:
            estimator_result (job results): the results of the circuits measurements

        Returns:
            npt.NDArray[np.cdouble]: value of the test
        """
        if isinstance(estimator_result, EstimatorResult):
            return np.array(
                [1.0 - 2.0 * val for val in estimator_result.values]
            ).astype("complex128")

        if isinstance(estimator_result, PrimitiveResult):
            return np.array(
                [1.0 - 2.0 * val.data.evs for val in estimator_result]
            ).astype("complex128")

        raise NotImplementedError(
            f"Cannot post processing for {type(estimator_result)} type class."
            f"Please, refer to {self.__class__.__name__}.post_processing()."
        )

    def get_value(self, estimator, parameter_sets: List, zne_strategy=None) -> List:
        """Compute the value of the test

        Args:
            estimator (Estimator): an estimator instance
            parameter_sets (List): The list of parameter values for the circuit

        Returns:
            List: value of the test
        """

        ncircuits = len(self.circuits)
        all_parameter_sets = [parameter_sets] * ncircuits
        all_observables = [self.observable] * ncircuits

        estimator_run_builder = EstimatorRunBuilder(
            estimator,
            self.circuits,
            all_observables,
            all_parameter_sets,
            options={"shots": self.shots},
        )

        try:
            if zne_strategy is None:
                job = estimator_run_builder.build_run()
            else:
                job = estimator.run(
                    self.circuits,
                    [self.observable] * ncircuits,
                    [parameter_sets] * ncircuits,
                    shots=self.shots,
                    zne_strategy=zne_strategy,
                )
            results = self.post_processing(job.result())
        except Exception as exc:
            raise AlgorithmError(
                "The primitive to evaluate the Hadammard Test failed!"
            ) from exc

        results *= np.array([1.0, 1.0j])
        return results.sum()
