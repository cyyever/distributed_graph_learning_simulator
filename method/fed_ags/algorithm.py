from distributed_learning_simulation import (
    CompositeAggregationAlgorithm,
    FedAVGAlgorithm,
    GraphTopologyAlgorithm,
)


class FedAGSAlgorithm(CompositeAggregationAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.append_algorithm(GraphTopologyAlgorithm())
        fed_avg_algorithm = FedAVGAlgorithm()
        fed_avg_algorithm.aggregate_loss = True
        self.append_algorithm(fed_avg_algorithm)
