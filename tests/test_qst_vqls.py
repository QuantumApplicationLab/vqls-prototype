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
from qiskit.circuit.library import real_amplitudes

from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator, Sampler
from vqls_prototype import QST_VQLS, VQLSLog


class TestQSTVQLS(unittest.TestCase):
    """Test VQLS"""

    def setUp(self):
        super().setUp()

        self.options = ({"use_local_cost_function": False, "use_overlap_test": False},)

        self.estimators = (
            Estimator(),
            # AerEstimator(),
        )

        self.samplers = (
            Sampler(),
            # AerSampler(),
        )

        self.log = VQLSLog([], [])

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

        for estimator, sampler in zip(self.estimators, self.samplers):
            for opt in self.options:
                vqls = QST_VQLS(
                    estimator,
                    ansatz,
                    COBYLA(maxiter=10, disp=True),
                    options=opt,
                    sampler=sampler,
                )
                _ = vqls.solve(matrix, rhs)


if __name__ == "__main__":
    unittest.main()
