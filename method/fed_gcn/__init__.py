"""
FedGCN: Convergence and Communication Tradeoffs in Federated Training of Graph Convolutional Networks
"""

from distributed_learning_simulation import (
    AggregationServer,
    AlgorithmRepository,
    GraphAlgorithm,
)

from .worker import FedGCNWorker

AlgorithmRepository.register_algorithm(
    algorithm_name="fed_gcn",
    client_cls=FedGCNWorker,
    server_cls=AggregationServer,
    algorithm_cls=GraphAlgorithm,
)
