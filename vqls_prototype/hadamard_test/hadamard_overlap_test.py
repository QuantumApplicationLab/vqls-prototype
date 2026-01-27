from typing import Optional, List
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_algorithms.exceptions import AlgorithmError
import numpy as np
import numpy.typing as npt

from qiskit.primitives.sampler import SamplerResult
from qiskit.primitives.containers import PrimitiveResult
from vqls_prototype.primitives_run_builder import SamplerRunBuilder


class BatchHadammardOverlapTest:
    r"""Class that execute batches of Hadamard Test"""

    def __init__(self, hdmr_list: List):
        """Create a single container that computes many hadamard tests

        Args:
            hdrm_list (List): A list of DirectHadamardTest instances
        """
        self.hdmr_list = hdmr_list
        self.circuits = [c for hdmr in hdmr_list for c in hdmr.circuits]
        self.post_processing = hdmr_list[0].post_processing
        self.shots = hdmr_list[0].shots

    def get_values(self, sampler, parameter_sets: List, zne_strategy=None) -> List:
        """Compute the value of the test

        Args:
            sampler (Sampler): an sampler instance
            parameter_sets (List): The list of parameter values for the circuit

        Returns:
            List: values of the batched Hadammard Tests
        """

        ncircuits = len(self.circuits)
        all_parameter_sets = [parameter_sets] * ncircuits

        sampler_run_builder = SamplerRunBuilder(
            sampler,
            self.circuits,
            all_parameter_sets,
            options={"shots": self.shots},
        )

        try:
            if zne_strategy is None:
                job = sampler_run_builder.build_run()
            else:
                job = sampler.run(
                    self.circuits,
                    [parameter_sets] * ncircuits,
                    shots=self.shots,
                    zne_strategy=zne_strategy,
                )

            results = self.post_processing(job.result())
        except Exception as exc:
            print(exc)
            raise AlgorithmError(
                "The primitive to evaluate the Hadammard Test failed!"
            ) from exc

        results *= np.array([1.0, 1.0j] * (ncircuits // 2))
        return results.reshape(-1, 2).sum(1).reshape(-1)


class HadamardOverlapTest:
    r"""Class to compute the Hadamard Test"""

    def __init__(
        self,
        operators: List[QuantumCircuit],
        use_barrier: Optional[bool] = False,
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = True,
        shots: Optional[int] = 4000,
    ):
        r"""Create the quantum circuits required to compute the hadamard test:

        .. math::

            \\langle 0 | U^\dagger A_l V | 0 \\rangle \\langle V^\dagger A_m^\dagger U | 0 \\rangle

        Args:
            operators (List[QuantumCircuit]): List of quantum circuits representing
                the operators [U, A_l, A_m].
            use_barrier (Optional[bool], optional): introduce barriers in the
                description of the circuits.  Defaults to False.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to create
                |Psi> from |0>. If None, assume that the qubits of the firsr
                register are alredy in Psi.
            apply_measurement (Optional[bool], optional): apply explicit measurement.
                Defaults to False.

        Returns:
            List[QuantumCircuit]: List of quamtum circuits required to compute the Hadammard Test.
        """

        self.operator_num_qubits = operators[0].num_qubits
        self.num_qubits = 2 * operators[0].num_qubits + 1
        if apply_initial_state is not None:
            if apply_initial_state.num_qubits != operators[0].num_qubits:
                raise ValueError(
                    "The operator and the initial state circuits have different numbers of qubits"
                )

        # classical bit for explicit measurement
        self.num_clbits = self.num_qubits

        # build the circuits
        self.circuits = self._build_circuit(
            operators,
            use_barrier,
            apply_initial_state,
            apply_measurement,
        )

        # number of circuits required
        self.ncircuits = len(self.circuits)

        # post processing coefficients
        self.post_process_coeffs = self.compute_post_processing_coefficients()

        # var for iterator
        self.iiter = None

        # number of shots
        self.shots = shots

    def _build_circuit(
        self,
        operators: List[QuantumCircuit],
        use_barrier: bool,
        apply_initial_state: Optional[QuantumCircuit] = None,
        apply_measurement: Optional[bool] = False,
    ) -> List[QuantumCircuit]:
        """build the quantum circuits

        Args:
            operators (List[QuantumCircuit]): quantum circuit or list of quantum circuits
                representing the [U, Al, Am].
            use_barrier (bool): introduce barriers in the description of the circuits.
            apply_initial_state (Optional[QuantumCircuit], optional): Quantum Circuit to create
                |Psi> from |0>. If None, assume that the qubits are alredy in Psi.
                Defaults to None.
            apply_measurement (Optional[bool], optional): apply explicit measurement.
                Defaults to False.

        Returns:
            List[QuantumCircuit]: List of quamtum circuits required to compute the Hadammard Test.
        """

        circuits = []
        op_umat, op_al, op_am = operators

        for imaginary in [False, True]:
            qctrl = QuantumRegister(1, "qctrl")
            qreg0 = QuantumRegister(op_al.num_qubits, "qr0")
            qreg1 = QuantumRegister(op_am.num_qubits, "qr1")
            circuit = QuantumCircuit(qctrl, qreg0, qreg1)

            # hadadmard gate on ctrl qbit
            circuit.h(qctrl)

            # prepare psi on the first register
            if apply_initial_state is not None:
                circuit.append(apply_initial_state, qreg0)

            # apply U on the second register
            circuit.append(op_umat, qreg1)

            if use_barrier:
                circuit.barrier()

            # apply Al on the first qreg
            idx = [0] + list(range(1, op_al.num_qubits + 1))
            circuit.append(op_al.control(1), idx)

            # apply Am^\dagger on the second reg
            idx = [0] + list(range(op_al.num_qubits + 1, 2 * op_al.num_qubits + 1))
            circuit.append(op_am.inverse().control(1), idx)

            if use_barrier:
                circuit.barrier()

            # apply the cnot gate
            for qubit_0, qubit_1 in zip(qreg0, qreg1):
                circuit.cx(qubit_0, qubit_1)

            # Sdg on ctrl qbit
            if imaginary:
                circuit.rz(-np.pi / 2, qctrl)

            if use_barrier:
                circuit.barrier()

            # hadamard on ctrl circuit
            circuit.h(qctrl)
            for qubit_0 in qreg0:
                circuit.h(qubit_0)

            # measure
            if apply_measurement:
                circuit.measure_all(inplace=True)

            circuits.append(circuit)

        return circuits

    def compute_post_processing_coefficients(self):
        """Compute the coefficients for the postprocessing"""

        # compute [1,1,1,-1] \otimes n
        # these are the coefficients if the qubits of register A and B
        # are ordered as A0 B0 A1 B1 .... AN BN
        coeff_0 = np.array([1, 1, 1, -1])
        coeffs = np.array([1, 1, 1, -1])
        for _ in range(1, self.operator_num_qubits):
            coeffs = np.tensordot(coeffs, coeff_0, axes=0).flatten()

        # create all the possible bit strings of a single register
        bit_strings = []
        for i in range(2 ** (self.operator_num_qubits)):
            bit_strings.append(f"{i:b}".zfill(self.operator_num_qubits))

        # coeff in the A0 A1 .. AN B0 B1 ... BN
        reordered_coeffs = np.zeros_like(coeffs)

        # Reorder the coefficients from
        # A0 B0 A1 B1 ... AN BN => A0 A1 .. AN B0 B1 ... BN
        for bs1 in bit_strings:
            for bs2 in bit_strings:
                idx = int(bs1 + bs2, 2)
                new_bit_string = "".join([i + j for i, j in zip(bs1, bs2)])
                idx_ori = int(new_bit_string, 2)
                reordered_coeffs[idx] = coeffs[idx_ori]

        return reordered_coeffs

    def post_processing(self, sampler_result) -> npt.NDArray[np.cdouble]:
        """Post process the sampled values of the circuits

        Args:
            sampler_result: Result of the sampler

        Returns:
            Complex numpy array: value of the overlap hadamard test
        """
        if isinstance(sampler_result, SamplerResult):
            quasi_dist = sampler_result.quasi_dists

        elif isinstance(sampler_result, PrimitiveResult):
            quasi_dist = [
                {
                    key: value / result.data.meas.num_shots
                    for key, value in result.data.meas.get_int_counts().items()
                }
                for result in sampler_result
            ]

        else:
            raise NotImplementedError(
                f"Cannot post processing for {type(sampler_result)} type class."
                f"Please, refer to {self.__class__.__name__}.post_processing()."
            )

        output = []

        for qdist in quasi_dist:
            # add missing keys
            val = np.array(
                [qdist[k] if k in qdist else 0 for k in range(2**self.num_qubits)]
            )

            value_0, value_1 = val[0::2], val[1::2]
            proba_0 = (value_0 * self.post_process_coeffs).sum()
            proba_1 = (value_1 * self.post_process_coeffs).sum()

            output.append(proba_0 - proba_1)

        return np.array(output).astype("complex128")

    def get_value(self, sampler, parameter_sets: List) -> float:
        """Compute and return the value of Hadmard overlap test

        Args:
            sampler: a Sampler primitive to extract the output of the circuits
            parameter_sets: the parameters of the variational circuits

        Returns:
            float: value of the overlap hadamard test
        """
        ncircuits = len(self.circuits)
        all_parameter_sets = [parameter_sets] * ncircuits

        sampler_run_builder = SamplerRunBuilder(
            sampler,
            self.circuits,
            all_parameter_sets,
            options={"shots": self.shots},
        )
        job = sampler_run_builder.build_run()
        results = self.post_processing(job.result())

        results *= np.array([1.0, 1.0j])
        return results.sum()
