from typing import Any

from neat.activations import sigmoid_activation
from neat.genetics.genome import Genome


"""
INPUT:
node_evals = [(node_key, activation_function, aggregation_function, bias, response, inputs)]
r = RecurrentNetwork(inputs, outputs, node_evals)

Example:
node_evals = [(0, activations.sigmoid_activation, sum, 0.0, 1.0, [])]
r = RecurrentNetwork([], [0], node_evals)
"""


def required_for_output(inputs: list[int], outputs: list[int], connections: dict[tuple[int, int], Any]) -> set[int]:
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.
    Returns a set of identifiers of required nodes.
    """
    assert not set(inputs).intersection(outputs)

    required = set(outputs)
    s = set(outputs)
    while 1:
        # Find nodes not in s whose output is consumed by a node in s.
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


class RecurrentNetwork(object):
    def __init__(
        self,
        inputs: list[int],
        outputs: list[int],
        node_evals: list[tuple[int, float, list[tuple[int, float]]]]
    ):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.response = 1.0

        self.values = [{}, {}]
        for val_dict in self.values:
            for key in [*inputs, *outputs]:
                val_dict[key] = 0.0

            for node_key, _bias, links in self.node_evals:
                val_dict[node_key] = 0.0
                for i, _w in links:
                    val_dict[i] = 0.0
        self.active = 0

    def reset(self):
        self.values = [dict((k, 0.0) for k in v) for v in self.values]
        self.active = 0

    def activate(self, inputs: list[float]) -> list[float]:
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        ivalues = self.values[self.active]
        ovalues = self.values[1 - self.active]
        self.active = 1 - self.active

        for i, v in zip(self.input_nodes, inputs):
            ivalues[i] = v
            ovalues[i] = v

        for node, bias, links in self.node_evals:
            node_inputs = [ivalues[i] * w for i, w in links]
            s = sum(node_inputs)
            ovalues[node] = sigmoid_activation(bias + self.response * s)

        return [ovalues[i] for i in self.output_nodes]

    @staticmethod
    def create(genome: Genome):
        """ Receives a genome and returns its phenotype (a RecurrentNetwork). """
        required = required_for_output(list(genome.inputs.keys()), genome.output_keys, genome.connections)

        # Gather inputs and expressed connections.
        node_inputs = {}
        for conn_gene in genome.connections.values():
            if not conn_gene.enabled:
                continue

            in_key, out_key = conn_gene.node_in_idx, conn_gene.node_out_idx
            if out_key not in required and in_key not in required:
                continue

            if out_key not in node_inputs:
                node_inputs[out_key] = [(in_key, conn_gene.weight)]
            else:
                node_inputs[out_key].append((in_key, conn_gene.weight))

        node_evals = []
        for node_key, inputs in node_inputs.items():
            node = genome.nodes[node_key]
            node_evals.append((node_key, node.bias, inputs))

        return RecurrentNetwork(list(genome.inputs.keys()), genome.output_keys, node_evals)
