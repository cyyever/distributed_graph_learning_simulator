from distributed_learning_simulation import Message, ParameterMessageBase
from distributed_learning_simulation.graph_worker import \
    NodeSelectionMixin

from ..fed_aas.worker import FedAASWorker


class FedAISWorker(FedAASWorker, NodeSelectionMixin):
    def _load_result_from_server(self, result: Message) -> None:
        after_aggregation = True
        if not isinstance(result, ParameterMessageBase):
            after_aggregation = False
        super()._load_result_from_server(result=result)
        if not after_aggregation:
            return
        self.append_node_selection_hook()
