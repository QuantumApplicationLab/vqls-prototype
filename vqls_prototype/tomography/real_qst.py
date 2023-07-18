import qiskit
import numpy as np
import treelib


class RealQST:
    def __init__(self, circuit, sampler):
        """Perform a QST for real valued state vector
        This needs only N additional circuits but require some posprocesing

        Args:
            circuit (QuantumCircuit): the base circuit we want to evaluate
            sampler (Sampler): A sampler primitive
        """
        self.root = []
        self.leaf = []

        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.size = 2**self.num_qubits
        self.tree = self.get_tree()
        self.path_to_node = self.get_path()
        self.sampler = sampler
        self.list_circuits = self.get_circuits()
        self.ncircuits = len(self.list_circuits)

    def get_tree(self):
        """Compute the tree"""

        def init_tree():
            """_summary_

            Returns:
                _type_: _description_
            """
            trees = []
            level_root, level_leaf = [], []
            for i in range(int(self.size / 2)):
                tree = treelib.Tree()
                a, b = 2 * (i), 2 * (i) + 1
                tree.create_node(a, a, data=1)
                tree.create_node(b, b, parent=a)
                trees.append(tree)

                level_leaf.append(b)
                level_root.append(a)
            return trees, level_root, level_leaf

        def link_trees(trees):
            """_summary_

            Args:
                trees (_type_): _description_
            """
            ntree = len(trees)
            level_root, level_leaf = [], []
            for iter in range(1, self.num_qubits):
                root, leaf = [], []
                for iroot in range(0, int(ntree), 2**iter):
                    new_root = trees[iroot].root
                    new_leaf = iroot + 2 ** (iter - 1)
                    trees[iroot].paste(new_root, trees[new_leaf])

                    root.append(trees[iroot].root)
                    leaf.append(trees[new_leaf].root)

                level_root.append(root)
                level_leaf.append(leaf)
            return trees[0], level_root, level_leaf

        tree_list, root, leaf = init_tree()
        self.root.append(root)
        self.leaf.append(leaf)
        tree, root, leaf = link_trees(tree_list)
        self.root += root
        self.leaf += leaf

        return tree

    def get_path(self):
        """_summary_"""
        paths = []
        for inode in range(self.size):
            paths.append(list(self.tree.rsearch(inode)))
        return paths

    def get_circuits(self):
        """_summary_

        Args:
            circuits (_type_): _description_
        """
        list_circuits = [self.circuit.measure_all(inplace=False)]

        for iq in range(self.num_qubits):
            qc = qiskit.QuantumCircuit(self.num_qubits)
            qc.append(self.circuit, range(self.num_qubits))
            qc.h(iq)
            list_circuits.append(qc.measure_all(inplace=False))
        return list_circuits

    def get_samples(self, parameters):
        """_summary_

        Args:
            sampler (_type_): _description_
            circuits (_type_): _description_
        """
        results = (
            self.sampler.run(self.list_circuits, [parameters] * self.ncircuits)
            .result()
            .quasi_dists
        )
        samples = []
        for res in results:
            proba = np.zeros(2**self.num_qubits)
            for k, v in res.items():
                proba[k] = v
            samples.append(proba)
        return samples

    def get_weight(self, samples):
        """_summary_

        Args:
            samples (_type_): _description_
        """
        # init the weights
        # signs = np.sign(2 * samples[1] - samples[0])
        # weights = np.zeros_like(signs)
        # weights[1::2] = signs[::2]

        # root
        weights = np.zeros_like(samples[0])
        weights[0] = 1

        # link the weights
        # nblocks = self.size // 2
        for iter in range(0, self.num_qubits):
            # signs = np.sign(2 * samples[iter + 1] - samples[0])
            # for i, iroot in enumerate(range(0, nblocks, 2**iter)):
            # new_root = 2 * iroot
            # new_leaf = 2 * (iroot + 2 ** (iter - 1))
            # weights[new_leaf] = signs[new_root]
            roots = self.root[iter]
            leafs = self.leaf[iter]
            signs = np.sign(
                2 * samples[iter + 1][roots] - samples[0][roots] - samples[0][leafs]
            )
            weights[leafs] = signs

        return weights

    def get_relative_amplitude_sign(self, parameters):
        """_summary_

        Args:
            circuit (_type_): _description_
            parameters (_type_): _description_
            backend (_type_): _description_
        """
        samples = self.get_samples(parameters)
        weights = self.get_weight(samples)
        signs = np.zeros_like(weights)
        for ip, path in enumerate(self.path_to_node):
            signs[ip] = weights[path].prod()
        return signs