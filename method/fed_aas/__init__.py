"""
Our work
"""

from distributed_learning_simulation import (
    AlgorithmRepository,
    DifferentialPrivacyEmbeddingEndpoint,
    GraphAlgorithm,
)

from .server import FedAASServer
from .worker import FedAASWorker

AlgorithmRepository.register_algorithm(
    algorithm_name="fed_aas",
    client_cls=FedAASWorker,
    server_cls=FedAASServer,
    algorithm_cls=GraphAlgorithm,
)
AlgorithmRepository.register_algorithm(
    algorithm_name="fed_aas_dp",
    client_cls=FedAASWorker,
    server_cls=FedAASServer,
    algorithm_cls=GraphAlgorithm,
    client_endpoint_cls=DifferentialPrivacyEmbeddingEndpoint,
)
