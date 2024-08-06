import os

import hydra
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch_geometric.data
import torch_geometric.utils
from cyy_torch_graph import GraphDatasetUtil
from cyy_torch_toolbox import MachineLearningPhase
from distributed_learning_simulation.config import (DistributedTrainingConfig,
                                                    import_dependencies)
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
    assert practitioners
    raw_graph_dict = {}
    node_client = None
    raw_graph = None
    sub_training_node_indices = set()
    total_edge_mask = None
    assert len(practitioners) == 2
    training_node_masks = []
    y = []
    for practitioner_idx, practitioner in enumerate(practitioners):
        dc = practitioner.create_dataset_collection(global_config)
        util = dc.get_dataset_util(phase=MachineLearningPhase.Training)
        assert isinstance(util, GraphDatasetUtil)
        if not raw_graph_dict:
            raw_graph = util.get_original_graph(0)
            raw_graph_dict = raw_graph.to_dict()
            y = raw_graph.y.tolist()
        training_node_mask = util.get_mask()[0].clone()

        node_drop_rate = 0.95
        dropout_mask = torch.bernoulli(
            torch.full(training_node_mask.size(), 1 - node_drop_rate)
        ).to(dtype=torch.bool)
        training_node_mask &= dropout_mask
        training_node_masks.append(training_node_mask.clone())
        practitioner_node_indices = torch_geometric.utils.mask_to_index(
            training_node_mask
        )
        sub_training_node_indices.update(practitioner_node_indices.tolist())
        if node_client is None:
            node_client = torch.zeros_like(training_node_mask, dtype=torch.long)
        for node_idx in practitioner_node_indices:
            node_client[node_idx] = practitioner_idx
    assert len(training_node_masks) == 2
    edge_mask = (
        training_node_masks[0][raw_graph_dict["edge_index"][0]]
        & training_node_masks[1][raw_graph_dict["edge_index"][1]]
    )
    raw_graph_dict["edge_index"] = raw_graph_dict["edge_index"][:, edge_mask]
    raw_graph = type(raw_graph)(**raw_graph_dict, bipartite=node_client)
    # raw_graph = raw_graph.subgraph(
    #     subset=torch.tensor(list(sub_training_node_indices), dtype=torch.long)
    # )
    # sub_training_node_indices = set(raw_graph.edge_index.view(-1).tolist())
    # raw_graph = raw_graph.subgraph(
    #     subset=torch.tensor(list(sub_training_node_indices), dtype=torch.long)
    # )
    G = torch_geometric.utils.to_networkx(data=raw_graph, node_attrs=["y", "bipartite"])
    # Separate by group
    G = G.subgraph(sub_training_node_indices)
    print(G)
    pos = {}

    # # Update position for node from each group
    # pos.update(
    #     (node, (1, index))
    #     for index, node in enumerate(
    #         torch_geometric.utils.mask_to_index(training_node_masks[0]).tolist()
    #     )
    # )
    # pos.update(
    #     (node, (2, index))
    #     for index, node in enumerate(
    #         torch_geometric.utils.mask_to_index(training_node_masks[1]).tolist()
    #     )
    # )
    pos = nx.bipartite_layout(
        G,
        torch_geometric.utils.mask_to_index(training_node_masks[0]).tolist(),
        scale=10,
    )
    nx.nx_agraph.write_dot(
        G, f"{global_config.dc_config.dataset_name}_{global_config.worker_number}.dot"
    )

    nodes = list(G)
    node_colors = []
    for node in nodes:
        node_colors.append(y[node])
    nx.draw(
        G,
        pos=pos,
        with_labels=False,
        node_color=node_colors,
        node_size=30,
        arrowstyle="-",
        arrowsize=0.1,
        cmap="tab10",
    )
    plt.tight_layout()
    plt.savefig(
        f"{global_config.dc_config.dataset_name}_{global_config.worker_number}.png"
    )
