# Variational Quantum Linear Solver
# Ref :
# Tutorial :

"""Variational Quantum Linear Solver

See https://arxiv.org/abs/1909.05820
"""
from typing import Optional, Union, List, Callable, Dict, Tuple
import numpy as np
import time
import logging

from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV1, BaseSamplerV1
from qiskit_algorithms.utils import validate_bounds
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import Minimizer, Optimizer
from qiskit_algorithms.gradients import BaseEstimatorGradient
from qiskit import transpile

from .variational_linear_solver import (
    VariationalLinearSolverResult,
)
from ..matrix_decomposition.matrix_decomposition import (
    SymmetricDecomposition,
    MatrixDecomposition,
    PauliDecomposition,
)

from ..matrix_decomposition.optimized_matrix_decomposition import (
    OptimizedPauliDecomposition,
    ContractedPauliDecomposition,
)
from ..hadamard_test.hadamard_test import (
    HadamardTest,
    BatchHadamardTest,
)

from ..hadamard_test.hadamard_overlap_test import (
    HadamardOverlapTest,
    BatchHadammardOverlapTest,
)

from ..hadamard_test.direct_hadamard_test import (
    DirectHadamardTest,
    BatchDirectHadamardTest,
)
from .validation import validate_initial_point
from .base_solver import BaseSolver

# module logger
logger = logging.getLogger(__name__)


class VQLS(BaseSolver):
    r"""Systems of linear equations arise naturally in many real-life applications in a wide range
    of areas, such as in the solution of Partial Differential Equations, the calibration of
    financial models, fluid simulation or numerical field calculation. The problem can be defined
    as, given a matrix :math:`A\in\mathbb{C}^{N\times N}` and a vector
    :math:`\vec{b}\in\mathbb{C}^{N}`, find :math:`\vec{x}\in\mathbb{C}^{N}` satisfying
    :math:`A\vec{x}=\vec{b}`.
    """

    def __init__(
        self,
        estimator: BaseEstimatorV1,
        ansatz: QuantumCircuit,
        optimizer: Union[Optimizer, Minimizer],
        sampler: Optional[Union[BaseSamplerV1, None]] = None,
        initial_point: Optional[Union[np.ndarray, None]] = None,
        gradient: Optional[Union[BaseEstimatorGradient, Callable, None]] = None,
        max_evals_grouped: Optional[int] = 1,
        options: Optional[Union[Dict, None]] = None,
    ) -> None:
        super().__init__(
            estimator,
            ansatz,
            optimizer,
            sampler,
            initial_point,
            gradient,
            max_evals_grouped,
        )

        self.default_solve_options = {
            "use_overlap_test": False,
            "use_local_cost_function": False,
            "matrix_decomposition": "symmetric",
            "shots": None,
            "reuse_matrix": False,
            "verbose": False,
        }
        self.options = self._validate_solve_options(options)

        # adding a dict to keep timing information of various tasks for benchmarks. Appended to results.
        self._timing = {
        # todo: it could be interesting to split the transpile time to gain more insight, e.g. transpile(vector_circuit)?
            "transpile_time_local": 0.0,  # local transpile time
            "quantum_time_wall": 0.0,  # wall-clock time around get_values/cost_evaluation
            "qpu_job_execution_time": 0.0,  # server-side execution as reported by the backend, if available
            "classical_opt_time": 0.0,  # time spent in classical optimizer
        }
        self.circuits_constructed = False

        self.supported_decomposition = {
            "pauli": PauliDecomposition,
            "contracted_pauli": ContractedPauliDecomposition,
            "optimized_pauli": OptimizedPauliDecomposition,
            "symmetric": SymmetricDecomposition,
        }

        self.supported_decomposition_list = tuple(
            v for _, v in self.supported_decomposition.items()
        )

    def construct_circuit(  # pylint: disable=too-many-branches
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List],
        vector: Union[np.ndarray, QuantumCircuit],
        transpilation_function=None, **transpile_kwargs
    ) -> Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
        """Returns the list of circuits required to compute the expectation value

        Args:
            matrix (Union[np.ndarray, QuantumCircuit, List]): matrix of the linear system
            vector (Union[np.ndarray, QuantumCircuit]): rhs of the linear system
            transpilation_function (Callable, optional): function to do the transpilation of the circuit

        Raises:
            ValueError: if vector and matrix have different size
            ValueError: if vector and matrix have different number of qubits
            ValueError: the input matrix is not a numpy array nor a quantum circuit

        Returns:
            List[QuantumCircuit]: Quantum Circuits required to compute the cost function
        """

        # state preparation
        if isinstance(vector, QuantumCircuit):
            nqbit = vector.num_qubits
            self.vector_circuit = vector

        elif isinstance(vector, np.ndarray):
            # ensure the vector is double
            vector = vector.astype("float64")

            if vector.ndim == 2:
                vector = vector.flatten()

            # create the circuit
            nqbit = int(np.log2(len(vector)))
            self.vector_circuit = QuantumCircuit(nqbit, name="Ub")

            # prep the vector if its norm is non nul
            vec_norm = np.linalg.norm(vector)
            if vec_norm != 0:
                self.vector_circuit.prepare_state(vector / vec_norm)
            else:
                raise ValueError("Norm of b vector is null!")
        else:
            raise ValueError("Format of the input vector not recognized")

        # Reuse the matrix if we reinit with a different rhs
        if (self.options["reuse_matrix"] is True) and (
            self.matrix_circuits is not None
        ):
            print("\t VQLS : Reusing matrix decomposition for new RHS")

        # recreate a decomposition for the matrix
        else:

            # general np array
            if isinstance(matrix, np.ndarray):
                # ensure the matrix is double
                matrix = matrix.astype("float64")

                if matrix.shape[0] != 2**self.vector_circuit.num_qubits:
                    raise ValueError(
                        "Input vector dimension does not match input "
                        "matrix dimension! Vector dimension: "
                        + str(self.vector_circuit.num_qubits)
                        + ". Matrix dimension: "
                        + str(matrix.shape[0])
                    )
                decomposition = self.supported_decomposition[
                    self.options["matrix_decomposition"]
                ]
                self.matrix_circuits = decomposition(matrix=matrix)

            # a pregenerated decomposition
            elif isinstance(matrix, self.supported_decomposition_list):
                self.matrix_circuits = matrix  # type: ignore[assignment]

            # a single circuit
            elif isinstance(matrix, QuantumCircuit):
                if matrix.num_qubits != self.vector_circuit.num_qubits:
                    raise ValueError(
                        "Matrix and vector circuits have different numbers of qubits."
                    )
                self.matrix_circuits = MatrixDecomposition(circuits=matrix)

            # if it's a list of (coefficients, circuits)
            elif isinstance(matrix, List):
                assert isinstance(matrix[0][0], (float, complex))
                assert isinstance(matrix[0][1], QuantumCircuit)
                self.matrix_circuits = MatrixDecomposition(
                    circuits=[m[1] for m in matrix], coefficients=[m[0] for m in matrix]
                )

            else:
                raise ValueError(
                    "Format of the input matrix not recognized", type(matrix)
                )

        # create only the circuit for <psi|psi> =  <0|V A_n ^* A_m V|0>
        # with n != m as the diagonal terms (n==m) always give a proba of 1.0
        hdmr_tests_norm = self._get_norm_circuits()

        # create the circuits for <b|psi>
        # local cost function
        if self.options["use_local_cost_function"]:
            hdmr_tests_overlap = self._get_local_circuits()

        # global cost function
        else:
            hdmr_tests_overlap = self._get_global_circuits()

        # all circuits are constructed, transpile for backend
        backend_for_transpile = getattr(getattr(self, "estimator", None), "backend", None)
        if backend_for_transpile is not None:
            t0 = time.perf_counter()
            try:
                hdmr_tests_norm = self.manually_transpile(hdmr_tests_norm, backend_for_transpile, transpilation_function, **transpile_kwargs)
                hdmr_tests_overlap = self.manually_transpile(hdmr_tests_overlap, backend_for_transpile, transpilation_function, **transpile_kwargs)
            finally:
                self._timing["transpile_time_local"] += time.perf_counter() - t0

        self.hdmr_tests_norm = hdmr_tests_norm
        self.hdmr_tests_overlap = hdmr_tests_overlap
        self.circuits_constructed = True
        return hdmr_tests_norm, hdmr_tests_overlap

    @staticmethod
    def manually_transpile(tests, backend, transpilation_function=None, **transpile_kwargs):
        """
        Transpile only raw QuantumCircuit items, leave wrapper objects unchanged.

        If `transpilation_function` is provided it will be called as
        `transpilation_function(circuit, backend, **transpile_kwargs)`.
        Otherwise, Qiskit's `transpile` is used: `transpile(circuit, backend=backend, **transpile_kwargs)`.
        """
        transpiled = []
        for t in tests:
            if isinstance(t, QuantumCircuit):
                if transpilation_function is None:
                    transpiled.append(transpile(t, backend=backend, **transpile_kwargs))
                else:
                    transpiled.append(transpilation_function(t, backend, **transpile_kwargs))
            else:
                transpiled.append(t)
        return transpiled

    def _get_norm_circuits(self) -> List[QuantumCircuit]:
        """construct the circuit for the norm

        Returns:
            List[QuantumCircuit]: quantum circuits needed for the norm
        """

        hdmr_tests_norm = []

        # use measurement optimized Pauli decomposition
        if isinstance(self.matrix_circuits, OptimizedPauliDecomposition):
            for (
                circ
            ) in self.matrix_circuits.optimized_measurement.shared_basis_transformation:
                hdmr_tests_norm.append(
                    DirectHadamardTest(
                        operators=circ,
                        apply_initial_state=self._ansatz,
                        shots=self.options["shots"],
                    )
                )

        # use contracted Pauli Decomposition: SHOULD IT BE DIRECTHADAMARDTEST ?!
        elif isinstance(self.matrix_circuits, ContractedPauliDecomposition):
            for circ in self.matrix_circuits.contracted_circuits:
                hdmr_tests_norm.append(
                    HadamardTest(  # type: ignore[arg-type]
                        operators=[circ],
                        apply_initial_state=self._ansatz,
                        apply_measurement=False,
                        shots=self.options["shots"],
                    )
                )

        # if we use the bare decomposition to create the circuits
        else:
            for ii_mat in range(len(self.matrix_circuits)):
                mat_i = self.matrix_circuits[ii_mat]

                for jj_mat in range(ii_mat + 1, len(self.matrix_circuits)):
                    mat_j = self.matrix_circuits[jj_mat]
                    hdmr_tests_norm.append(
                        HadamardTest(  # type: ignore[arg-type]
                            operators=[mat_i.circuit.inverse(), mat_j.circuit],
                            apply_initial_state=self._ansatz,
                            apply_measurement=False,
                            shots=self.options["shots"],
                        )
                    )

        return hdmr_tests_norm

    def _get_local_circuits(self) -> List[QuantumCircuit]:
        """construct the circuits needed for the local cost function

        Returns:
            List[QuantumCircuit]: quantum circuits for the local cost function
        """

        hdmr_tests_overlap = []
        num_z = self.matrix_circuits[0].circuit.num_qubits

        # create the circuits for <0| U^* A_l V(Zj . Ij|) V^* Am^* U|0>
        for ii_mat in range(len(self.matrix_circuits)):
            mat_i = self.matrix_circuits[ii_mat]

            for jj_mat in range(ii_mat, len(self.matrix_circuits)):
                mat_j = self.matrix_circuits[jj_mat]

                for iqubit in range(num_z):
                    # circuit for the CZ operation on the iqth qubit
                    qc_z = QuantumCircuit(num_z + 1)
                    qc_z.cz(0, iqubit + 1)

                    # create Hadammard circuit
                    hdmr_tests_overlap.append(
                        HadamardTest(
                            operators=[
                                mat_i.circuit,
                                self.vector_circuit.inverse(),
                                qc_z,
                                self.vector_circuit,
                                mat_j.circuit.inverse(),
                            ],
                            apply_control_to_operator=[True, True, False, True, True],
                            apply_initial_state=self.ansatz,
                            apply_measurement=False,
                            shots=self.options["shots"],
                        )
                    )
        return hdmr_tests_overlap

    def _get_global_circuits(self) -> List[QuantumCircuit]:
        """construct circuits needed for the global cost function


        Returns:
            List[QuantumCircuit]: quantum circuits needed for the global cost function
        """

        # create the circuits for <0|U^* A_l V|0\rangle\langle 0| V^* Am^* U|0>
        # either using overal test or hadammard test
        if self.options["use_overlap_test"]:
            hdmr_overlap_tests = []
            for ii_mat in range(len(self.matrix_circuits)):
                mat_i = self.matrix_circuits[ii_mat]

                for jj_mat in range(ii_mat, len(self.matrix_circuits)):
                    mat_j = self.matrix_circuits[jj_mat]

                    hdmr_overlap_tests.append(
                        HadamardOverlapTest(
                            operators=[
                                self.vector_circuit,
                                mat_i.circuit,
                                mat_j.circuit,
                            ],
                            apply_initial_state=self.ansatz,
                            apply_measurement=True,
                            shots=self.options["shots"],
                        )
                    )
            return hdmr_overlap_tests

        # or using the normal Hadamard tests

        # Note there is an issue if we direcly pass self.vector_circuit.inverse()
        # as an operator to the HadammardTest.
        # therefore we first create the controlled version of self.vector_circuit.inverse()
        # and pass that to Hadammard test requiring not to apply control

        # precompute the controlled version of the inverse vector circuit
        qc_u = QuantumCircuit(self.vector_circuit.num_qubits + 1, name="c_U")
        qc_u.append(
            self.vector_circuit.inverse().control(1),
            list(range(self.vector_circuit.num_qubits + 1)),
        )

        # create the tests
        hdmr_tests = []
        for mat_i in self.matrix_circuits:
            hdmr_tests.append(
                HadamardTest(
                    operators=[self.ansatz, mat_i.circuit, qc_u],
                    apply_control_to_operator=[True, True, False],
                    apply_measurement=False,
                    shots=self.options["shots"],
                )
            )
        return hdmr_tests

    @staticmethod
    def get_coefficient_matrix(coeffs) -> np.ndarray:
        """Compute all the vi* vj terms

        Args:
            coeffs (np.ndarray): list of complex coefficients
        """
        return coeffs[:, None].conj() @ coeffs[None, :]

    def _assemble_cost_function(
        self,
        hdmr_values_norm: np.ndarray,
        hdmr_values_overlap: np.ndarray,
        coefficient_matrix: np.ndarray,
    ) -> float:
        """Computes the value of the cost function

        Args:
            hdmr_values_norm (np.ndarray): values of the hadamard test for the norm
            hdmr_values_overlap (np.ndarray): values of the hadamard tests for the overlap
            coefficient_matrix (np.ndarray): expansion coefficients of the matrix

        Returns:
            float: value of the cost function
        """

        # compute all the terms in <\phi|\phi> = \sum c_i* cj <0|V Ai* Aj V|0>
        norm = self._compute_normalization_term(coefficient_matrix, hdmr_values_norm)

        if self.options["use_local_cost_function"]:
            # compute all terms in
            # \sum c_i* c_j 1/n \sum_n <0|V* Ai U Zn U* Aj* V|0>
            sum_terms = self._compute_local_terms(
                coefficient_matrix, hdmr_values_overlap, norm
            )

        else:
            # compute all the terms in
            # |<b|\phi>|^2 = \sum c_i* cj <0|U* Ai V|0><0|V* Aj* U|0>
            sum_terms = self._compute_global_terms(
                coefficient_matrix, hdmr_values_overlap
            )

        # overall cost
        cost = 1.0 - np.real(sum_terms / norm)

        return cost

    def get_cost_evaluation_function(
        self,
        hdmr_tests_norm: List,
        hdmr_tests_overlap: List,
        coefficient_matrix: np.ndarray,
    ) -> Callable[[np.ndarray], Union[float, List[float]]]:
        """Generate the cost function of the minimization process

        Args:
            hdmr_tests_norm (List): list of quantum circuits needed to compute the norm
            hdmr_tests_overlap (List): list of quantum circuits needed to compute the norm
            coefficient_matrix (np.ndarray): the matrix values of the c_n^* c_m coefficients

        Raises:
            RuntimeError: If the ansatz is not parametrizable

        Returns:
            Callable[[np.ndarray], Union[float, List[float]]]: the cost function
        """

        num_parameters = self.ansatz.num_parameters
        if num_parameters == 0:
            raise RuntimeError(
                "The ansatz must be parameterized, but has 0 free parameters."
            )

        def cost_evaluation(parameters):
            t0 = time.perf_counter()
            primitive = self.estimator

            # compute the values of the norm with optimized/contracted Pauli decomposition
            if type(self.matrix_circuits) in [
                ContractedPauliDecomposition,
                OptimizedPauliDecomposition,
            ]:
                # switch to sampler primitive if we do measurement optimization
                BatchTest = BatchHadamardTest
                if isinstance(self.matrix_circuits, OptimizedPauliDecomposition):
                    primitive = self.sampler
                    BatchTest = BatchDirectHadamardTest

                # compute the hadamard values of the unique circuits
                hdmr_values_norm = BatchTest(hdmr_tests_norm).get_values(
                    primitive, parameters
                )

                # postprocess the values
                hdmr_values_norm = (
                    self.matrix_circuits.post_process_contracted_norm_values(
                        hdmr_values_norm
                    )
                )

            # compute the norm with other decomposition
            else:
                # estimate the expected values of the norm circuits
                hdmr_values_norm = BatchHadamardTest(hdmr_tests_norm).get_values(
                    primitive, parameters
                )

            # switch primitive to sampler if we do overlap test
            primitive = self.estimator
            BatchTest = BatchHadamardTest
            if self.options["use_overlap_test"]:
                primitive = self.sampler
                BatchTest = BatchHadammardOverlapTest

            # estimate the expected values of the overlap circuits
            hdmr_values_overlap = BatchTest(hdmr_tests_overlap).get_values(
                primitive, parameters
            )

            # compute the total cost
            cost = self._assemble_cost_function(
                hdmr_values_norm, hdmr_values_overlap, coefficient_matrix
            )

            # get the intermediate results if required
            if self._callback is not None:
                self._eval_count += 1
                self._callback(self._eval_count, cost, parameters)
            else:
                self._eval_count += 1
            if self.options["verbose"]:
                print(
                    f"VQLS Iteration {self._eval_count} Cost {cost:.3e}",
                    end="\r",
                    flush=True,
                )
            self._timing["quantum_time_wall"] +=  time.perf_counter() - t0
            return cost

        return cost_evaluation

    def _validate_solve_options(self, options: Union[Dict, None]) -> Dict:
        """Validate the options used for the solve methods

        Args:
            options (Union[Dict, None]): options
        """
        valid_keys = self.default_solve_options.keys()

        if options is None:
            options = self.default_solve_options

        else:
            for k in options.keys():
                if k not in valid_keys:
                    raise ValueError(
                        "Option {k} not recognized, valid keys are {valid_keys}"
                    )
            for k in valid_keys:
                if k not in options.keys():
                    options[k] = self.default_solve_options[k]

        if options["use_overlap_test"] and options["use_local_cost_function"]:
            raise ValueError(
                "Local cost function cannot be used with Hadamard Overlap test"
            )

        if options["use_overlap_test"] and self.sampler is None:
            raise ValueError(
                "Please provide a sampler primitives when using Hadamard Overlap test"
            )

        valid_matrix_decomposition = [
            "symmetric",
            "pauli",
            "contracted_pauli",
            "optimized_pauli",
        ]
        if options["matrix_decomposition"] not in valid_matrix_decomposition:
            raise ValueError(
                f"matrix decomposition {k} invalid, valid keys: {valid_matrix_decomposition}"
            )

        return options

    def _solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List[QuantumCircuit]],
        vector: Union[np.ndarray, QuantumCircuit],
    ) -> VariationalLinearSolverResult:
        """Solve the linear system

        Args:
            matrix (Union[List, np.ndarray, QuantumCircuit]): matrix of the linear system
            vector (Union[np.ndarray, QuantumCircuit]): rhs of the linear system

        Returns:
            VariationalLinearSolverResult: Result of the optimization
                and solution vector of the linear system
        """

        # compute the circuits needed for the hadamard tests
        if not self.circuits_constructed:
            self.construct_circuit(matrix, vector)

        # compute he coefficient matrix
        coefficient_matrix = self.get_coefficient_matrix(
            np.array([mat_i.coeff for mat_i in self.matrix_circuits])
        )

        # set an expectation for this algorithm run (will be reset to None at the end
        initial_point = validate_initial_point(self.initial_point, self.ansatz)
        bounds = validate_bounds(self.ansatz)

        # Convert the gradient operator into a callable function that is compatible with the
        # optimization routine.
        gradient = self._gradient
        self._eval_count = 0

        # get the cost evaluation function
        cost_evaluation = self.get_cost_evaluation_function(
            self.hdmr_tests_norm, self.hdmr_tests_overlap, coefficient_matrix
        )

        opt_start = time.perf_counter()
        if callable(self.optimizer):
            opt_result = self.optimizer(fun=cost_evaluation, x0=initial_point, jac=gradient, bounds=bounds) # pylint: disable=not-callable
        else:
            opt_result = self.optimizer.minimize(fun=cost_evaluation, x0=initial_point, jac=gradient, bounds=bounds)
        opt_time = time.perf_counter() - opt_start
        # opt.minimize() contains all evaluations of the quantum cost function, therefore need to subtract those here.
        opt_time = opt_time - self._timing["quantum_time_wall"]
        self._timing["classical_opt_time"] = self._timing["classical_opt_time"] + opt_time


        # create the solution
        solution = VariationalLinearSolverResult()

        # optimization data
        solution.optimal_point = opt_result.x
        solution.optimal_parameters = dict(zip(self.ansatz.parameters, opt_result.x))
        solution.optimal_value = opt_result.fun
        # todo: should this not rather be solution.optimizer_evals ?
        solution.cost_function_evals = opt_result.nfev
        #todo: solution.optimizer_result ?

        # final ansatz
        # todo: should this not rather be solution.optimal_circuit ?
        solution.state = self.ansatz.assign_parameters(solution.optimal_parameters)

        # solution vector
        solution.vector = np.real(Statevector(solution.state).data)

        # attach a copy of the timing dict to the results for the caller's inspection, e.g. benchmarking
        solution.transpile_time_local = self._timing["transpile_time_local"]
        solution.quantum_time_wall = self._timing["quantum_time_wall"]
        solution.qpu_job_execution_time = self._timing["qpu_job_execution_time"]
        # todo: should this rather be solution.optimizer_time?
        solution.classical_opt_time = self._timing["classical_opt_time"]

        return solution
