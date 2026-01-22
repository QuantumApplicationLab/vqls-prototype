# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test VQLS """


import unittest

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import Estimator, Sampler

from qiskit_algorithms.optimizers import ADAM

from qiskit_aer.primitives import EstimatorV2 as aer_EstimatorV2
from qiskit_aer.primitives import SamplerV2 as aer_SamplerV2

from vqls_prototype import VQLS

# 8-11-2023
# Overlap Hadamard test do not work with BasicAer  primitives anymore
# this test case is skipped for now


class TestVQLS(unittest.TestCase):
    """Test VQLS"""

    def setUp(self):
        super().setUp()

        self.options = (
            {"use_local_cost_function": False, "use_overlap_test": False},
            {"use_local_cost_function": True, "use_overlap_test": False},
            {"use_local_cost_function": False, "use_overlap_test": True},
        )

        self.estimators = (
            Estimator(),
            aer_EstimatorV2(),
            # AerEstimator(),
        )

        self.samplers = (
            Sampler(),
            aer_SamplerV2(),
            # AerSampler(),
        )

    def test_numpy_input(self):
        """Test the VQLS on matrix input using statevector simulator."""

        matrix = np.array(
            [
                [0.50, 0.25, 0.10, 0.00],
                [0.25, 0.50, 0.25, 0.10],
                [0.10, 0.25, 0.50, 0.25],
                [0.00, 0.10, 0.25, 0.50],
            ]
        )

        rhs = np.array([0.1] * 4)
        ansatz = real_amplitudes(num_qubits=2, reps=3, entanglement="full")

        for _, (estimator, sampler) in enumerate(zip(self.estimators, self.samplers)):
            for _, opt in enumerate(self.options):
                vqls = VQLS(
                    estimator,
                    ansatz,
                    ADAM(maxiter=2),
                    options=opt,
                    sampler=sampler,
                )
                _ = vqls.solve(matrix, rhs)

    def test_circuit_input_statevector(self):
        """Test the VQLS on circuits input using statevector simulator."""

        num_qubits = 2
        ansatz = real_amplitudes(num_qubits=num_qubits, reps=3, entanglement="full")

        rhs = QuantumCircuit(num_qubits)
        rhs.h(0)
        rhs.h(1)

        qc1 = QuantumCircuit(num_qubits)
        qc1.x(0)
        qc1.x(1)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit(num_qubits)
        qc2.h(0)
        qc2.x(1)
        qc2.cx(0, 1)

        for _, (estimator, sampler) in enumerate(zip(self.estimators, self.samplers)):
            for _, opt in enumerate(self.options):
                vqls = VQLS(
                    estimator,
                    ansatz,
                    ADAM(maxiter=2),
                    sampler=sampler,
                    options=opt,
                )
                _ = vqls.solve([[0.5, qc1], [0.5, qc2]], rhs)


if __name__ == "__main__":
    unittest.main()
