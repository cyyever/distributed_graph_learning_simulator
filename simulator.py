import os

os.putenv("no_pipe", "1")

import method  # noqa: F401
from cyy_naive_lib.log import redirect_stdout_to_logger
from distributed_learning_simulation import load_config, train

if __name__ == "__main__":
    with redirect_stdout_to_logger():
        config_path = os.path.join(os.path.dirname(__file__), "conf")
        config = load_config(
            config_path=config_path,
            global_conf_path=os.path.join(config_path, "global.yaml"),
            import_libs=False
        )
        train(config=config, single_task=True)
