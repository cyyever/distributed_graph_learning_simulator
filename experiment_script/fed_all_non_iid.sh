# 所有节点nodes均参与构成batch，不涉及对节点采样
for configname in cs.yaml PubMed.yaml yelp.yaml reddit.yaml ogbn-products.yaml; do
  for alpha in 0.5; do
    python3 ./simulator.py --config-name fed_gnn/${configname} ++fed_gnn.exp_name="fed_all" ++fed_gnn.algorithm_kwargs.node_sample_percent=1.0 ++fed_gnn.algorithm_kwargs.node_random_selection=true ++fed_gnn.worker_number=100 ++fed_gnn.round=100 ++fed_gnn.epoch=5 ++fed_gnn.algorithm_kwargs.edge_drop_rate=0.5 ++fed_gnn.algorithm_kwargs.share_feature=false ++fed_gnn.model_name=TwoGATCN ++fed_gnn.dataloader_kwargs.batch_number=10 ++fed_gnn.learning_rate=0.001 ++fed_gnn.fake_weight_decay=0.001 ++fed_gnn.dataset_sampling="dirichlet_split" ++fed_gnn.dataset_sampling_kwargs.concentration=$alpha
  done
done
