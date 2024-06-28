"""Methods to decompose a matrix into quantum circuits"""

from itertools import chain, combinations
from collections import namedtuple
from itertools import product
from typing import Optional, Union, List, Tuple, TypeVar, cast, Iterator


import numpy as np
from numpy.testing import assert_
import numpy.typing as npt
import scipy.linalg as spla
import scipy.sparse as spsp

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp
from tqdm import tqdm


complex_type = TypeVar("complex_type", float, complex)
complex_array_type = npt.NDArray[np.cdouble]


class MatrixDecomposition:
    """Base class for the decomposition of a matrix in quantum circuits."""

    CircuitElement = namedtuple("CircuitElement", ["coeff", "circuit"])

    @staticmethod
    def _as_complex(
        num_or_arr: Union[float, List[float], complex, List[complex]]
    ) -> npt.NDArray[np.cdouble]:
        """Converts a number or a list of numbers to a complex array.

        Args:
            num_or_arr (Union[complex_type, List[complex_type]]): array of number to convert

        Returns:
            complex_array_type: array of complex numbers
        """
        arr = num_or_arr if isinstance(num_or_arr, List) else [num_or_arr]
        return np.array(arr, dtype=np.cdouble)

    def __init__(
        self,
        matrix: Optional[npt.NDArray] = None,
        circuits: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        coefficients: Optional[
            Union[float, complex, List[float], List[complex]]
        ] = None,
        load: Optional[str] = None,
    ):
        """Decompose a matrix representing quantum circuits

        Args:
            matrix (Optional[npt.NDArray], optional): Array to decompose;
                only relevant in derived classes where
                `self.decompose_matrix()` has been implemented. Defaults to None.
            circuits (Optional[Union[QuantumCircuit, List[QuantumCircuit]]], optional):
                quantum circuits representing the matrix. Defaults to None.
            coefficients (Optional[ Union[float, complex, List[float], List[complex]] ], optional):
                coefficients associated with the input quantum circuits; `None` is
                valid only for a circuit with 1 element. Defaults to None.
            load (Optional[str]): filename to load the decomposition from
        """

        if load is not None:
            self.load(load)

        elif matrix is not None:  # ignore circuits & coefficients
            self.sparse_matrix = spsp.issparse(matrix)
            self._matrix, self.num_qubits = self._validate_matrix(matrix)
            self._coefficients, self._matrices, self._circuits = self.decompose_matrix()

        elif circuits is not None:
            self._circuits = circuits if isinstance(circuits, list) else [circuits]

            assert_(
                isinstance(self._circuits[0], QuantumCircuit),
                f"{circuits}: invalid circuit",
            )
            if coefficients is None:
                if len(self._circuits) == 1:
                    self._coefficients = self._as_complex(1.0)
                else:
                    raise ValueError("coefficients mandatory for multiple circuits")
            else:
                self._coefficients = self._as_complex(coefficients)

            if len(self._circuits) != len(self._coefficients):
                raise ValueError("number of coefficients and circuits do not match")

            self.num_qubits = self._circuits[0].num_qubits
            if not all(map(lambda ct: ct.num_qubits == self.num_qubits, self.circuits)):
                _num_qubits = [ct.num_qubits for ct in self.circuits]
                raise ValueError(f"mismatched number of qubits: {_num_qubits}")

            self._matrices = [Operator(qc).data for qc in self.circuits]
            # self._matrix = self.recompose()
        else:
            raise ValueError(
                f"inconsistent arguments: matrix={matrix}, \
                    coefficients={coefficients}, circuits={circuits}"
            )

        self.num_circuits = len(self._circuits)
        self.iiter = 0

    @classmethod
    def _compute_circuit_size(cls, matrix: npt.NDArray) -> int:
        """Compute the size of the circuit represented by the matrix

        Args:
            matrix (npt.NDArray): matrix representing the circuit

        Returns:
            int: circuit size
        """
        return int(np.log2(matrix.shape[0]))

    @classmethod
    def _validate_matrix(
        cls, matrix: complex_array_type
    ) -> Tuple[complex_array_type, int]:
        """Check the size of the matrix

        Args:
            matrix (complex_array_type): input matrix

        Raises:
            ValueError: if the matrix is not square
            ValueError: if the matrix size is not a power of 2
            ValueError: if the matrix is not symmetric

        Returns:
            Tuple[complex_array_type, int]: matrix and the number of qubits required
        """
        if len(matrix.shape) == 2 and matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Input matrix must be square: matrix.shape={matrix.shape}"
            )
        num_qubits = cls._compute_circuit_size(matrix)
        if num_qubits % 1 != 0:
            raise ValueError(
                f"Input matrix dimension is not a power of 2: {num_qubits}"
            )
        # if not np.allclose(matrix, matrix.conj().T):
        #     raise ValueError(f"Input matrix isn't symmetric:\n{matrix}")

        return matrix, num_qubits

    @property
    def matrix(self) -> np.ndarray:
        """matrix of the decomposition"""
        return self._matrix

    @property
    def circuits(self) -> List[QuantumCircuit]:
        """circuits of the decomposition"""
        return self._circuits

    @circuits.setter
    def circuits(self, circuits: List[QuantumCircuit]) -> None:
        """Set new circuits

        Args:
            circuits (List[QuantumCircuit]): new circuits
        """
        self._circuits = circuits

    @property
    def coefficients(self) -> complex_array_type:
        """coefficients of the decomposition."""
        return self._coefficients

    @coefficients.setter
    def coefficients(self, new_coefficients: np.ndarray) -> None:
        """Set the values of the coefficients

        Args:
            new_coefficients (np.ndarray): new values
        """
        self._coefficients = new_coefficients

    @property
    def matrices(self) -> List[complex_array_type]:
        """return the unitary matrices"""
        return self._matrices

    def __iter__(self):
        self.iiter = 0
        return self

    def __next__(self):
        if self.iiter < self.num_circuits:
            out = self.CircuitElement(
                self._coefficients[self.iiter], self._circuits[self.iiter]
            )
            self.iiter += 1
            return out
        raise StopIteration

    def __len__(self):
        return len(self._circuits)

    def __getitem__(self, index):
        return self.CircuitElement(self._coefficients[index], self._circuits[index])

    def recompose(self) -> complex_array_type:
        """Rebuilds the original matrix from the decomposed one.

        Returns:
            complex_array_type: The recomposed matrix.
        """
        coeffs, matrices = self.coefficients, self.matrices
        return (coeffs.reshape(len(coeffs), 1, 1) * matrices).sum(axis=0)

    def decompose_matrix(
        self,
    ) -> Tuple[complex_array_type, List[complex_array_type], List[QuantumCircuit]]:
        raise NotImplementedError(f"can't decompose in {self.__class__.__name__!r}")

    def update_matrix(self, new_matrix: npt.NDArray) -> None:
        """Update the decomposition with a new matrix

        Args:
            new_matrix (npt.NDArray): new input matrix
        """
        self.sparse_matrix = spsp.issparse(new_matrix)
        self._matrix, self.num_qubits = self._validate_matrix(new_matrix)
        self._coefficients, self._matrices, self._circuits = self.decompose_matrix()

    def save(self, filename) -> None:
        """save the decomposition for future use

        Args:
            filename (str): name of the file
        """
        raise NotImplementedError("Save method not implemented for this decomposition")

    def load(self, filename) -> None:
        """load a decomposition from file

        Args:
            filename (str): name of the file
        """
        raise NotImplementedError("Load method not implemented for this decomposition")


class SymmetricDecomposition(MatrixDecomposition):
    """
    A class that represents the symmetric decomposition of a matrix.
    For the mathematical background for the decomposition, see the following
    math.sx answer: https://math.stackexchange.com/a/1710390
    """

    def _create_circuits(
        self, unimatrices: List[np.ndarray], names: List[str]
    ) -> List[QuantumCircuit]:
        """Construct the quantum circuits from unitary matrices

        Args:
            unimatrices (List[np.ndarray]): list of unitary matrices of the decomposition.
            names (List[str]): names of the circuits

        Returns:
            List[QuantumCircuit]: quantum circuits
        """

        def make_qc(mat: complex_array_type, name: str) -> QuantumCircuit:
            circuit = QuantumCircuit(self.num_qubits, name=name)
            circuit.unitary(mat, circuit.qubits)
            return circuit

        return [make_qc(mat, name) for mat, name in zip(unimatrices, names)]

    @staticmethod
    def auxilliary_matrix(
        x: Union[npt.NDArray[np.float64], complex_array_type]
    ) -> complex_array_type:
        """Returns the auxiliary matrix for the decomposition of size n.
           and derfined as defined as : i * sqrt(I - x^2)

        Args:
            x (Union[npt.NDArray[np.float_], complex_array_type]): original matrix.

        Returns:
            complex_array_type: The auxiliary matrix.
        """
        mat = np.eye(len(x)) - x @ x
        mat = cast(npt.NDArray[Union[np.float64, np.cdouble]], spla.sqrtm(mat))
        return 1.0j * mat

    def decompose_matrix(
        self,
    ) -> Tuple[complex_array_type, List[complex_array_type], List[QuantumCircuit]]:
        """Decompose a generic numpy matrix into a sum of unitary matrices.

        Returns:
            Tuple[complex_array_type, List[complex_array_type], List[QuantumCircuit]]:
                A tuple containing the list of coefficients numpy matrices,
                and quantum circuits of the decomposition.
        """

        # Normalize
        norm = np.linalg.norm(self._matrix)
        mat = self._matrix / norm

        mat_real = np.real(mat)
        mat_imag = np.imag(mat)

        coef_real = norm * 0.5
        coef_imag = coef_real * 1j

        # Get the matrices
        unitary_matrices, unitary_coefficients = [], []
        if not np.allclose(mat_real, 0.0):
            aux_mat = self.auxilliary_matrix(mat_real)
            unitary_matrices += [mat_real + aux_mat, mat_real - aux_mat]
            unitary_coefficients += [coef_real] * 2

        if not np.allclose(mat_imag, 0.0):
            aux_mat = self.auxilliary_matrix(mat_imag)
            unitary_matrices += [mat_imag + aux_mat, mat_imag - aux_mat]
            unitary_coefficients += [coef_imag] * 2
        unit_coeffs = np.array(unitary_coefficients, dtype=np.cdouble)

        # create the circuits
        names = ["A+", "A-"]
        circuits = self._create_circuits(unitary_matrices, names)

        return unit_coeffs, unitary_matrices, circuits

    def recompose(self) -> complex_array_type:
        """Rebuilds the original matrix from the decomposed one.

        Returns:
            complex_array_type: The recomposed matrix.
        """
        coeffs, matrices = self.coefficients, self.matrices
        return (coeffs.reshape(len(coeffs), 1, 1) * matrices).sum(axis=0)

    def save(self, filename) -> None:
        """save the decomposition for future use

        Args:
            filename (str): name of the file
        """
        raise NotImplementedError(
            "Save method not implemented for this Symmetric decomposition"
        )

    def load(self, filename) -> None:
        """load a decomposition from file

        Args:
            filename (str): name of the file
        """
        raise NotImplementedError(
            "Load method not implemented for this Symmetric decomposition"
        )


class PauliDecomposition(MatrixDecomposition):
    """A class that represents the Pauli decomposition of a matrix.
    Could be replaced by SparsePauliOp.from_operator(A)"""

    basis = "IXYZ"

    def __init__(
        self,
        matrix: Optional[Union[npt.NDArray, spsp.csr_array]] = None,
        circuits: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
        coefficients: Optional[
            Union[float, complex, List[float], List[complex]]
        ] = None,
        load: Optional[str] = None,
        sparse: Optional[bool] = False,
    ):
        """Decompose a matrix representing quantum circuits

        Args:
            matrix (Optional[npt.NDArray], optional): Array to decompose;
                only relevant in derived classes where
                `self.decompose_matrix()` has been implemented. Defaults to None.
            circuits (Optional[Union[QuantumCircuit, List[QuantumCircuit]]], optional):
                quantum circuits representing the matrix. Defaults to None.
            coefficients (Optional[ Union[float, complex, List[float], List[complex]] ], optional):
                coefficients associated with the input quantum circuits; `None` is
                valid only for a circuit with 1 element. Defaults to None.
            load (Optional[str]): filename to load the decomposition from
            sparse (optional[bool]): use a sparse decomposition
        """
        self.use_sparse = sparse
        super().__init__(matrix, circuits, coefficients, load)

    @staticmethod
    def _create_circuit(pauli_string: str) -> QuantumCircuit:
        """creates a quantum circuit for a given pauli string

        Args:
            pauli_string (str): the input pauli string

        Returns:
            QuantumCircuit: quantum circuit for the string
        """
        num_qubit = len(pauli_string)
        circuit = QuantumCircuit(num_qubit, name=pauli_string)
        for iqbit, gate in enumerate(pauli_string[::-1]):
            if (
                gate.upper() != "I"
            ):  # identity gate cannot be controlled by ancillary qubit
                getattr(circuit, gate.lower())(iqbit)
        return circuit

    def get_possible_pauli_strings(self) -> List:
        """Return a list of all possible Pauli strings

        Returns:
            List: list of pauli strings
        """

        # if we use the sparse decomposition
        if self.use_sparse:
            # for now convert to coo and extract indices
            assert isinstance(self._matrix, spsp.spmatrix)
            coo_mat = self._matrix.tocoo()
            idx_row, idx_col = coo_mat.row, coo_mat.col

            # get the diagonal pauli strings
            matrix_size = coo_mat.shape[0]
            possible_pauli_strings = get_diagonal_elements_pauli_string(matrix_size)

            # add off diagonal pauli strings
            for irow, icol in zip(idx_row, idx_col):
                if irow != icol:
                    possible_pauli_strings += get_off_diagonal_element_pauli_strings(
                        irow, icol, matrix_size
                    )

            return list(set(possible_pauli_strings))

        # if we use the full decomposition
        return list(product(self.basis, repeat=self.num_qubits))

    def decompose_matrix(
        self,
    ) -> Tuple[complex_array_type, List[complex_array_type], List[QuantumCircuit]]:
        """Decompose a generic numpy matrix into a sum of Pauli strings.

        Returns:
            Tuple[complex_array_type, List[complex_array_type]]:
                A tuple containing the list of coefficients and
                the numpy matrix of the decomposition.
        """

        prefactor = 1.0 / (2**self.num_qubits)
        unit_mats: List[complex_array_type] = []
        coeffs, circuits = [], []
        self.strings = []
        possible_pauli_strings = self.get_possible_pauli_strings()
        for pauli_gates in tqdm(possible_pauli_strings):
            pauli_string = "".join(pauli_gates)
            coef: complex_array_type = self._get_pauli_coefficient(
                self.matrix, pauli_string, self.sparse_matrix
            )
            if not np.isclose(coef * np.conj(coef), 0):
                self.strings.append(pauli_string)
                coeffs.append(prefactor * coef)
                circuits.append(self._create_circuit(pauli_string))

        return np.array(coeffs, dtype=np.cdouble), unit_mats, circuits

    def update_matrix(self, new_matrix: npt.NDArray) -> None:
        """Update the coefficients using a new matrix

        Args:
            new_matrix (npt.NDArray): new input matrix

        """
        prefactor = 1.0 / (2**self.num_qubits)
        coeffs = []
        for pauli_string in tqdm(self.strings):
            coef = self._get_pauli_coefficient(
                new_matrix, pauli_string, self.sparse_matrix
            )
            coeffs.append(prefactor * coef)

        self._matrix = new_matrix
        self.coefficients = np.array(coeffs, dtype=np.cdouble)

    @staticmethod
    def _get_pauli_coefficient(
        matrix: npt.ArrayLike, pauli_string: str, sparse_matrix: bool
    ) -> complex_array_type:
        """Compute the pauli coefficient of a given pauli string

        Args:
            matrix (npt.ArrayLike): input matrix
            pauli_string (str): a given pauli string
            sparse_matrix (bool): if the matrix is sparse or not

        Returns:
            complex_array_type: pauli coefficient
        """
        pauli_op = SparsePauliOp(pauli_string)  # Pauli(pauli_string)
        if sparse_matrix:
            coef = (pauli_op.to_matrix(sparse=True) @ matrix).trace()
        else:
            coef = np.einsum("ij,ji", pauli_op, matrix)  # type: ignore
        return coef

    def recompose(self) -> complex_array_type:
        """Recompose the full matrix

        Returns:
            complex_array_type: recomposed matrix
        """
        return SparsePauliOp(self.strings, self.coefficients).to_matrix()

    def save(self, filename: str) -> None:
        """save the decomposition for future use

        Args:
            filename (str): name of the file
        """
        np.save(filename, np.stack((self.strings, self.coefficients)))

    def load(self, filename) -> None:
        """load a decomposition from file

        Args:
            filename (str): name of the file
        """
        pauli_strings, coeffs = np.load(filename)
        self.strings = pauli_strings
        self.coefficient = coeffs
        self.circuits = []
        for pauli_string in self.strings:
            self.circuits.append(self._create_circuit(pauli_string))


def get_off_diagonal_element_pauli_strings(
    idx_row: int, idx_col: int, matrix_size: int
) -> List:
    """Get the pauli strings associated with the index (i,j) of a matrix element of size matrix_size

    Args:
        idx_row (int): row index of the element
        idx_col (int): column index of the element
        matrix_size (int): size of the matrix

    Returns:
        List: list of pauli strings associated with that element
    """

    x_matrix = np.array([[0, 1], [1, 0]])
    # shift = 0

    def powerset(iterable: List) -> Iterator:
        """Create a powerset (0,2) -> [(), (0), (2), (0,2)]

        Args:
            iterable (list): indices

        Returns:
            iterator: powerset
        """
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def iz_sub(xi_string: str) -> List:
        """Returns a list of strings containing all substituion of I by Z gates

        Args:
            xi_string (str): Pauli string cointaining only I and X gates

        Returns:
            List: List of combinations
        """
        strings = []
        index_id = [i for i, v in enumerate(list(xi_string)) if v == "I"]
        pset = powerset(index_id)
        next(pset)
        for idx in pset:
            new_string = list(xi_string)
            for r in idx:
                new_string[r] = "Z"
            strings.append("".join(new_string))
        return strings

    def xy_sub(xi_string: str) -> List:
        """Returns a list of strings containing all even substituions of X by Y gates

        Args:
            xi_string (str): Pauli string cointaining only I and X gates

        Returns:
            List: List of combinations
        """
        strings = []
        index_id = [i for i, v in enumerate(list(xi_string)) if v == "X"]
        pset = [ps for ps in powerset(index_id) if len(ps) % 2 == 0]
        for idx in pset[1:]:
            new_string = list(xi_string)
            for r in idx:
                new_string[r] = "Y"
            strings.append("".join(new_string))
        return strings

    def get_val_xi_string(i, j, shift, size):
        """Get the int value of the binary representation of the
            XI string associated with the element (i,j)
            The XI string is a strign containing only
            X and I gate and that has a non null element at (i,j)

        Args:
            i (int): row index
            j (int): column index
            shift (int): value of the shift
            size (int): size of the matrix

        Returns:
            int: value of the bin repr of the string e.g.:
                    1 = 001 => IIX
                    5 = 101 => XIX
        """

        if size == 2:
            return x_matrix[i, j] + shift

        # prepare the next iteration
        shift += int((i >= (size // 2)) ^ (j >= size // 2)) * (size // 2)
        return get_val_xi_string(i % (size // 2), j % (size // 2), shift, (size // 2))

    def val2xistring(val_xi_string: int, size: int) -> str:
        """convert the value of the bin repr of the xi sting into the xi string

        Args:
            val (int): value of the bin repr of the xi_str

        Returns:
            str: xi string
        """
        return (
            np.binary_repr(val_xi_string, int(np.log2(size)))
            .replace("0", "I")
            .replace("1", "X")
        )

    # get the val of the bin rep of the xi string
    val_xi_string = get_val_xi_string(idx_row, idx_col, 0, matrix_size)

    # convert value to pauli string repre
    xi_string = val2xistring(val_xi_string, matrix_size)

    # get the strings with I->Z substitutions
    iz_sub_strings = iz_sub(xi_string)

    # get the strings with XX -> YY substitutions
    xy_sub_strings = xy_sub(xi_string)

    return [xi_string] + iz_sub_strings + xy_sub_strings


def get_diagonal_elements_pauli_string(matrix_size: int) -> List:
    """Get the combination of I and Z gates to encode the diagonal

    Args:
        matrix_size (int): size of the matrix
    """
    num_qubits = int(np.log2(matrix_size))
    return list(product("IZ", repeat=num_qubits))
