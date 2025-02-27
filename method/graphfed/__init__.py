from distributed_learning_simulation import AlgorithmRepository

from .algorithm import GraphFedAlgorithm
from .server import GraphFedServer
from .worker import GraphFedWorker

AlgorithmRepository.register_algorithm(
    algorithm_name="graphfed",
    client_cls=GraphFedWorker,
    server_cls=GraphFedServer,
    algorithm_cls=GraphFedAlgorithm,
)
