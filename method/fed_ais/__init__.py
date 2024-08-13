"""
Our work
"""

from ..fed_aas.server import \
    FedAASServer
from distributed_learning_simulation import (
    CentralizedAlgorithmFactory, DifferentialPrivacyEmbeddingEndpoint,
    GraphAlgorithm)

from .worker import FedAISWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_ais",
    client_cls=FedAISWorker,
    server_cls=FedAASServer,
    algorithm_cls=GraphAlgorithm,
)

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="fed_ais_dp",
    client_cls=FedAISWorker,
    server_cls=FedAASServer,
    algorithm_cls=GraphAlgorithm,
    client_endpoint_cls=DifferentialPrivacyEmbeddingEndpoint,
)
