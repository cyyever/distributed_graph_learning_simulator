# fedlocal
for configname in cs.yaml PubMed.yaml yelp.yaml reddit.yaml ogbn-products.yaml; do
  python3 ./simulator.py --config-name fed_gnn/${configname} ++fed_gnn.worker_number=100 ++fed_gnn.round=100 ++fed_gnn.model_name=TwoGATCN ++fed_gnn.epoch=5 ++fed_gnn.dataloader_kwargs.batch_number=10 ++fed_gnn.learning_rate=0.001 ++fed_gnn.algorithm_kwargs.edge_drop_rate=0.5 ++fed_gnn.algorithm_kwargs.share_feature=false ++fed_gnn.exp_name="fed_local" ++fed_gnn.weight_decay=0.001 ++fed_gnn.algorithm_kwargs.sample_percent=0.7
done
