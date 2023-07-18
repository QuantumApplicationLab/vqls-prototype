# Variational Quantum Linear Solver
# Ref :
# Tutorial :


"""Variational Quantum Linear Solver

See https://arxiv.org/abs/1909.05820
"""

from qiskit.algorithms.optimizers import Minimizer, Optimizer
from typing import Optional, Union, List, Callable, Dict, Tuple
import numpy as np
from qiskit.opflow.gradients import GradientBase
from qiskit.primitives import BaseEstimator, BaseSampler
from qiskit import Aer
from qiskit import QuantumCircuit

from qiskit.algorithms.minimum_eigen_solvers.vqe import (
    _validate_bounds,
    _validate_initial_point,
)

from qiskit.quantum_info import Operator

from .variational_linear_solver import (
    VariationalLinearSolverResult,
)


from .matrix_decomposition.optimized_matrix_decomposition import (
    OptimizedPauliDecomposition,
)

from .vqls import VQLS

from .hadamard_test.direct_hadamard_test import (
    DirectHadamardTest,
    BatchDirectHadammardTest,
)
from .tomography.qst import FullQST
from .tomography.simulator_qst import SimulatorQST
from .tomography.real_qst import RealQST


class EVQLS(VQLS):
    r"""Systems of linear equations arise naturally in many real-life applications in a wide range
    of areas, such as in the solution of Partial Differential Equations, the calibration of
    financial models, fluid simulation or numerical field calculation. The problem can be defined
    as, given a matrix :math:`A\in\mathbb{C}^{N\times N}` and a vector
    :math:`\vec{b}\in\mathbb{C}^{N}`, find :math:`\vec{x}\in\mathbb{C}^{N}` satisfying
    :math:`A\vec{x}=\vec{b}`.

    Examples:

        .. jupyter-execute:

            from qalcore.qiskit.vqls.vqls import VQLS, VQLSLog
            from qiskit.circuit.library.n_local.real_amplitudes import RealAmplitudes
            from qiskit.algorithms import optimizers as opt
            from qiskit import Aer, BasicAer
            import numpy as np

            from qiskit.quantum_info import Statevector
            import matplotlib.pyplot as plt
            from qiskit.primitives import Estimator, Sampler, BackendEstimator

            # create random symmetric matrix
            A = np.random.rand(4, 4)
            A = A + A.T

            # create rhight hand side
            b = np.random.rand(4)

            # solve using numpy
            classical_solution = np.linalg.solve(A, b / np.linalg.norm(b))
            ref_solution = classical_solution / np.linalg.norm(classical_solution)

            # define the wave function ansatz
            ansatz = RealAmplitudes(2, entanglement="full", reps=3, insert_barriers=False)

            # define backend
            backend = BasicAer.get_backend("statevector_simulator")

            # define an estimator primitive
            estimator = Estimator()

            # define the logger
            log = VQLSLog([],[])

            # create the solver
            vqls = VQLS(
                estimator,
                ansatz,
                opt.CG(maxiter=200),
                callback=log.update
            )

            # solve
            res = vqls.solve(A, b, opt)
            vqls_solution = np.real(Statevector(res.state).data)

            # plot solution
            plt.scatter(ref_solution, vqls_solution)
            plt.plot([-1, 1], [-1, 1], "--")
            plt.show()

            # plot cost function
            plt.plot(log.values)
            plt.ylabel('Cost Function')
            plt.xlabel('Iterations')
            plt.show()

    References:

        [1] Carlos Bravo-Prieto, Ryan LaRose, M. Cerezo, Yigit Subasi, Lukasz Cincio,
        Patrick J. Coles. Variational Quantum Linear Solver
        `arXiv:1909.05820 <https://arxiv.org/abs/1909.05820>`
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        ansatz: QuantumCircuit,
        optimizer: Union[Optimizer, Minimizer],
        sampler: Union[BaseSampler, None],
        initial_point: Optional[Union[np.ndarray, None]] = None,
        gradient: Optional[Union[GradientBase, Callable, None]] = None,
        max_evals_grouped: Optional[int] = 1,
        callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
    ) -> None:
        r"""
        Args:
            estimator: an Estimator primitive to compute the expected values of the
                quantum circuits needed for the cost function
            ansatz: A parameterized circuit used as Ansatz for the wave function.
            optimizer: A classical optimizer. Can either be a Qiskit optimizer or a callable
                that takes an array as input and returns a Qiskit or SciPy optimization result.
            sampler: a Sampler primitive to sample the output of some quantum circuits needed to
                compute the cost function. This is only needed if overal Hadammard tests are used.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQLS will look to the ansatz for a preferred
                point and if not will simply compute a random one.
            gradient: An optional gradient function or operator for optimizer.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time. Deprecated if a gradient operator or function is
                given.
            callback: a callback that can access the intermediate data during the optimization.
                Three parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the cost and the parameters for the ansatz
        """
        super().__init__(
            estimator,
            ansatz,
            optimizer,
            sampler,
            initial_point,
            gradient,
            max_evals_grouped,
            callback,
        )

        self.tomography_calculator = None
        self.default_solve_options = {
            "use_overlap_test": False,
            "use_local_cost_function": False,
            "matrix_decomposition": "optimized_pauli",
            "tomography": "real_qst",
            "shots": 4000,
        }

    def construct_circuit(
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List],
        vector: Union[np.ndarray, QuantumCircuit],
        options: Dict,
    ) -> Tuple[List[QuantumCircuit], List[QuantumCircuit]]:
        """Returns the a list of circuits required to compute the expectation value

        Args:
            matrix (Union[np.ndarray, QuantumCircuit, List]): matrix of the linear system
            vector (Union[np.ndarray, QuantumCircuit]): rhs of thge linear system
            options (Dict): Options to compute define the quantum circuits
                that compute the cost function

        Raises:
            ValueError: if vector and matrix have different size
            ValueError: if vector and matrix have different number of qubits
            ValueError: the input matrix is not a numoy array nor a quantum circuit

        Returns:
            List[QuantumCircuit]: Quantum Circuits required to compute the cost function
        """

        # state preparation
        if isinstance(vector, QuantumCircuit):
            raise NotImplementedError("We didn't do that yet")

        elif isinstance(vector, np.ndarray):
            self.vector_norm = np.linalg.norm(vector)
            self.vector_amplitude = vector / self.vector_norm

        # general numpy matrix
        if isinstance(matrix, np.ndarray):
            self.matrix_circuits = OptimizedPauliDecomposition(matrix, vector)

        # a single circuit
        elif issubclass(matrix, OptimizedPauliDecomposition):
            self.matrix_circuits = matrix

        else:
            raise ValueError(
                "matrix should be a np.array or a OptimizedPauliDecomposition"
            )

        return self._get_norm_circuits(options), self._get_overlap_circuits(options)

    def _get_norm_circuits(self, options) -> List[QuantumCircuit]:
        """construct the circuit for the norm

        Returns:
            List[QuantumCircuit]: quantum circuits needed for the norm
        """

        circuits = []
        for (
            circ
        ) in self.matrix_circuits.optimized_measurement.shared_basis_transformation:
            circuits.append(
                DirectHadamardTest(
                    operators=circ,
                    apply_initial_state=self._ansatz,
                    shots=options["shots"],
                )
            )
        return circuits

    def _get_overlap_circuits(self, options) -> List[QuantumCircuit]:
        """_summary_

        Args:
            options (_type_): _description_

        Raises:
            RuntimeError: _description_
            ValueError: _description_

        Returns:
            List[QuantumCircuit]: _description_
        """
        circuits = []
        circuits.append(
            DirectHadamardTest(
                operators=self._ansatz,
                shots=options["shots"],
            )
        )

        return circuits

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
        options: Dict,
    ) -> float:
        """Computes the value of the cost function

        Args:
            hdmr_values_norm (np.ndarray): values of the hadamard test for the norm
            hdmr_values_overlap (np.ndarray): values of the hadamard tests for the overlap
            coefficient_matrix (np.ndarray): exapnsion coefficients of the matrix
            options (Dict): options to compute cost function

        Returns:
            float: value of the cost function
        """

        # compute all the terms in <\phi|\phi> = \sum c_i* cj <0|V Ai* Aj V|0>
        norm = self._compute_normalization_term(coefficient_matrix, hdmr_values_norm)

        # compute the overlap terms <b|AV|0>
        sum_terms = self._compute_global_terms(
            coefficient_matrix, hdmr_values_overlap, options
        )

        # overall cost
        cost = 1.0 - np.real(sum_terms / norm)

        return cost

    def get_cost_evaluation_function(
        self,
        norm_circuits: List,
        overlap_circuits: List,
        coefficient_matrix: np.ndarray,
        options: Dict,
    ) -> Callable[[np.ndarray], Union[float, List[float]]]:
        """Generate the cost function of the minimazation process

        Args:
            hdmr_tests_norm (List): list of quantum circuits needed to compute the norm
            hdmr_tests_overlap (List): list of quantum circuits needed to compute the norm
            coefficient_matrix (np.ndarray): the matrix values of the c_n^* c_m coefficients
            options (Dict): Option to compute the cost function

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
            num_norm_circuits = len(norm_circuits)
            circuits = norm_circuits + overlap_circuits

            # sample the unique circuits
            samples = BatchDirectHadammardTest(circuits).get_values(
                self.sampler, parameters
            )

            # postprocess the values for the norm
            hdmr_values_norm = self.matrix_circuits.get_norm_values(
                samples[:num_norm_circuits]
            )

            # post process the values for the overlap
            sign_ansatz = self.tomography_calculator.get_relative_amplitude_sign(
                parameters
            )
            hdmr_values_overlap = self.matrix_circuits.get_overlap_values(
                samples[num_norm_circuits:], sign_ansatz
            )

            # compute the total cost
            cost = self._assemble_cost_function(
                hdmr_values_norm, hdmr_values_overlap, coefficient_matrix, options
            )

            # get the intermediate results if required
            if self._callback is not None:
                self._eval_count += 1
                self._callback(self._eval_count, cost, parameters)
            else:
                self._eval_count += 1
                print(
                    f"VQLS Iteration {self._eval_count} Cost {cost}",
                    end="\r",
                    flush=True,
                )

            return cost

        return cost_evaluation

    def _validate_solve_options(self, options: Union[Dict, None]) -> Dict:
        """validate the options used for the solve methods

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

        if options["use_overlap_test"] != False:
            raise ValueError("Overlap test not implemented for evqls")
        if options["use_local_cost_function"] != False:
            raise ValueError("local cost function not implemented for evqls")
        if options["matrix_decomposition"] != "optimized_pauli":
            raise ValueError("Matrix decomposition must be optimied pauli for evqls")

        return options

    def _init_tomography(self, tomography: str):
        """initialize the tomography calculator

        Args:
            tomography (str): the name of the tomography
        """
        if tomography == "simulator":
            self.tomography_calculator = SimulatorQST(self._ansatz)
        elif tomography == "real_qst":
            self.tomography_calculator = RealQST(self._ansatz, self.sampler)
        elif tomography == "full_qst":
            self.tomography_calculator = FullQST(
                self._ansatz, Aer.get_backend("statevector_simulator")
            )
        else:
            raise ValueError("tomography method not recognized")

    def solve(
        self,
        matrix: Union[np.ndarray, QuantumCircuit, List[QuantumCircuit]],
        vector: Union[np.ndarray, QuantumCircuit],
        options: Union[Dict, None] = None,
    ) -> VariationalLinearSolverResult:
        """Solve the linear system

        Args:
            matrix (Union[List, np.ndarray, QuantumCircuit]): matrix of the linear system
            vector (Union[np.ndarray, QuantumCircuit]): rhs of the linear system
            options (Union[Dict, None]): options for the calculation of the cost function

        Returns:
            VariationalLinearSolverResult: Result of the optimization
                and solution vector of the linear system
        """

        # validate the options
        options = self._validate_solve_options(options)

        # intiialize the tomography
        self._init_tomography(options["tomography"])

        # compute the circuits needed for the hadamard tests
        norm_circuits, overlap_circuits = self.construct_circuit(
            matrix, vector, options
        )

        # compute he coefficient matrix
        coefficient_matrix = self.get_coefficient_matrix(
            np.array([mat_i.coeff for mat_i in self.matrix_circuits])
        )

        # set an expectation for this algorithm run (will be reset to None at the end)
        initial_point = _validate_initial_point(self.initial_point, self.ansatz)
        bounds = _validate_bounds(self.ansatz)

        # Convert the gradient operator into a callable function that is compatible with the
        # optimization routine.
        gradient = self._gradient
        self._eval_count = 0

        # get the cost evaluation function
        cost_evaluation = self.get_cost_evaluation_function(
            norm_circuits, overlap_circuits, coefficient_matrix, options
        )

        if callable(self.optimizer):
            opt_result = self.optimizer(  # pylint: disable=not-callable
                fun=cost_evaluation, x0=initial_point, jac=gradient, bounds=bounds
            )
        else:
            opt_result = self.optimizer.minimize(
                fun=cost_evaluation, x0=initial_point, jac=gradient, bounds=bounds
            )

        # create the solution
        solution = VariationalLinearSolverResult()

        # optimization data
        solution.optimal_point = opt_result.x
        solution.optimal_parameters = dict(zip(self.ansatz.parameters, opt_result.x))
        solution.optimal_value = opt_result.fun
        solution.cost_function_evals = opt_result.nfev

        # final ansatz
        solution.state = self.ansatz.assign_parameters(solution.optimal_parameters)

        return solution
