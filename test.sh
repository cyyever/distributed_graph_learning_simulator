# Graph
python3 ./simulator.py --config-name fed_gnn/cs.yaml ++fed_gnn.round=1 ++fed_gnn.epoch=1 ++fed_gnn.worker_number=2 ++fed_gnn.debug=False
python3 ./simulator.py --config-name fed_aas/yelp.yaml ++fed_aas.round=1 ++fed_aas.epoch=1 ++fed_aas.worker_number=2 ++fed_aas.debug=False
python3 ./simulator.py --config-name fed_cluster_gcn/cs.yaml ++fed_cluster_gcn.round=1 ++fed_cluster_gcn.epoch=1  ++fed_cluster_gcn.worker_number=6  ++fed_cluster_gcn.algorithm_kwargs.cluster_number=2
