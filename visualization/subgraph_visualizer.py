import os

import hydra
import networkx as nx
import torch_geometric.data
import torch_geometric.utils
from cyy_torch_graph import GraphDatasetUtil
from cyy_torch_toolbox import MachineLearningPhase
from distributed_learning_simulation.config import (
    DistributedTrainingConfig,
    import_dependencies,
)
from distributed_learning_simulation.config import load_config as __load_config

global_config: DistributedTrainingConfig = DistributedTrainingConfig()
import_dependencies()


config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "conf"))


@hydra.main(config_path=config_path, version_base=None)
def load_config(conf) -> None:
    global global_config
    global_config = __load_config(conf, os.path.join(config_path, "global.yaml"))


if __name__ == "__main__":
    load_config()
    practitioners = global_config.create_practitioners()
    sub_training_node_indices = set()
    y = []
    for practitioner_idx, practitioner in enumerate(practitioners):
        dc = practitioner.create_dataset_collection(global_config)
        util = dc.get_dataset_util(phase=MachineLearningPhase.Training)
        assert isinstance(util, GraphDatasetUtil)
        raw_graph = util.get_original_graph(0)
        raw_graph_dict = raw_graph.to_dict()
        y = raw_graph.y.tolist()
        node_mask = (
            util.get_mask()[0].clone()
            | dc.get_dataset_util(phase=MachineLearningPhase.Validation)
            .get_mask()[0]
            .clone()
        )

        # node_drop_rate = 0.5
        # dropout_mask = torch.bernoulli(
        #     torch.full(training_node_mask.size(), 1 - node_drop_rate)
        # ).to(dtype=torch.bool)
        # training_node_mask &= dropout_mask
        practitioner_node_indices = torch_geometric.utils.mask_to_index(node_mask)
        # sub_training_node_indices.update(practitioner_node_indices.tolist())
        # # edge_mask = (
        #     training_node_mask[raw_graph_dict["edge_index"][0]]
        #     & training_node_mask[raw_graph_dict["edge_index"][1]]
        # )
        # raw_graph_dict["edge_index"] = raw_graph_dict["edge_index"][:, edge_mask]
        raw_graph = type(raw_graph)(**raw_graph_dict)
        G = torch_geometric.utils.to_networkx(data=raw_graph, node_attrs=["y"])
        # Separate by group
        G = G.subgraph(practitioner_node_indices.tolist())
        nx.nx_agraph.write_dot(
            G,
            f"{global_config.dc_config.dataset_name}_{global_config.worker_number}.dot",
        )
        break
