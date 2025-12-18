"""
Our work
"""

from distributed_learning_simulation import AlgorithmRepository

from .algorithm import FedAGSAlgorithm
from .server import FedAGSServer
from .worker import FedAGSWorker

AlgorithmRepository.register_algorithm(
    algorithm_name="fed_ags",
    client_cls=FedAGSWorker,
    server_cls=FedAGSServer,
    algorithm_cls=FedAGSAlgorithm,
)
