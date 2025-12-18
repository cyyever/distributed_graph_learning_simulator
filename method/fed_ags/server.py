import math
from typing import Any

from cyy_naive_lib.log import log_warning
from distributed_learning_simulation import AggregationServer, ParameterMessageBase


class FedAGSServer(AggregationServer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__total_epoch = self.config.algorithm_kwargs["total_epoch"]
        self.__fixed_epoch = self.config.algorithm_kwargs.get("fixed_epoch", None)
        self.__initial_training_loss: float | None = None
        self._need_init_performance = True
        self.__used_epochs: int = 0
        self.__last_epochs: int | None = None
        self._warmup_rounds: int | None = self.config.algorithm_kwargs.get(
            "warmup_rounds", None
        )

    def _stopped(self) -> bool:
        return self._stop

    def __allocate_epoch(self, result: Any) -> int:
        if self.__fixed_epoch is not None:
            log_warning("use fixed local epoch %s", self.__fixed_epoch)
            return self.__fixed_epoch
        min_epoch = self.config.algorithm_kwargs["min_epoch"]

        if result.is_initial:
            log_warning("initial local epoch %s", min_epoch)
            return min_epoch

        last_training_loss = result.other_data.pop("training_loss")
        if self._warmup_rounds is not None and self.round_index < self._warmup_rounds:
            self.__initial_training_loss = last_training_loss
            log_warning("warm-up local epoch %s", min_epoch)
            return min_epoch

        if self.__initial_training_loss is None:
            self.__initial_training_loss = last_training_loss
            log_warning("initial training loss %s", last_training_loss)
            log_warning("use initial local epoch %s the second time", min_epoch)
            return min_epoch
        epochs = int(
            math.sqrt(last_training_loss / self.__initial_training_loss)
            * (self.__total_epoch - self.__used_epochs)
        )
        max_epoch = self.config.algorithm_kwargs.get("max_epoch", None)
        if max_epoch is not None and epochs > max_epoch:
            log_warning("reduce current epochs %s to %s", epochs, max_epoch)
            epochs = max_epoch
        epochs = max(epochs, min_epoch)
        if self.__last_epochs is not None:
            epochs = min(epochs, self.__last_epochs)
        self.__last_epochs = epochs
        log_warning(
            "initial loss is %s,last training loss is %s, current epoch is %s, used epochs are %s",
            self.__initial_training_loss,
            last_training_loss,
            epochs,
            self.__used_epochs,
        )
        return epochs

    def _before_send_result(self, result: Any) -> None:
        super()._before_send_result(result=result)
        if not isinstance(result, ParameterMessageBase):
            return
        if self.__used_epochs >= self.__total_epoch:
            result.end_training = True
            result.other_data["local_epoch"] = 1
            self._stop = True
            return
        local_epoch = self.__allocate_epoch(result)
        local_epoch = min(local_epoch, self.__total_epoch - self.__used_epochs)
        self._set_stat(key="local_epoch", value=local_epoch, message=result)
        result.other_data["local_epoch"] = local_epoch
        self.__used_epochs += local_epoch
