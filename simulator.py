import os

import hydra
import method  # noqa: F401
from distributed_learning_simulation.config import DistributedTrainingConfig
from distributed_learning_simulation.config import load_config as __load_config
from distributed_learning_simulation.training import train

os.putenv("no_pipe", "1")
global_config: DistributedTrainingConfig = DistributedTrainingConfig()


@hydra.main(config_path="./conf", version_base=None)
def load_config(conf) -> None:
    global global_config
    global_config = __load_config(
        conf,
        os.path.join(os.path.dirname(__file__), "conf", "global.yaml"),
        import_libs=False,
    )


if __name__ == "__main__":
    load_config()
    train(config=global_config)
