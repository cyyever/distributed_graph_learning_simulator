"""
Subgraph Federated Learning with Missing Neighbor Generation
"""

from distributed_learning_simulation import AlgorithmRepository, GraphAlgorithm

from .server import FedSagePlusServer
from .worker import FedSagePlusWorker

AlgorithmRepository.register_algorithm(
    algorithm_name="fed_sage_plus",
    client_cls=FedSagePlusWorker,
    server_cls=FedSagePlusServer,
    algorithm_cls=GraphAlgorithm,
)
