from typing import Any

from distributed_learning_simulation import Message
from distributed_learning_simulation.graph_worker import (
    GraphWorker, NodeSelectionMixin)


class FedAGSWorker(GraphWorker, NodeSelectionMixin):
    def __init__(self, **kwargs: Any) -> None:
        config = kwargs["config"]
        config.algorithm_kwargs["share_feature"] = False
        super().__init__(**kwargs)
        assert not self._share_feature
        self._send_loss = True
        self.disable_choosing_model_by_validation()

    def _load_result_from_server(self, result: Message) -> None:
        local_epoch = result.other_data.pop("local_epoch")
        super()._load_result_from_server(result=result)
        self.trainer.hyper_parameter.epoch = local_epoch
        self.remove_node_selection_hook()

        if local_epoch > 1:
            self.append_node_selection_hook()

    def _stopped(self) -> bool:
        return self._force_stop
