for configname in cs.yaml PubMed.yaml yelp.yaml reddit.yaml; do
  python3 ./simulator.py --config-name fed_gnn/${configname} ++fed_gnn.worker_number=100 ++fed_gnn.round=100 ++fed_gnn.model_name=TwoGATCN ++fed_gnn.epoch=1 ++fed_gnn.dataloader_kwargs.batch_number=10 ++fed_gnn.learning_rate=0.001 ++fed_gnn.algorithm_kwargs.edge_drop_rate=0.5 ++fed_gnn.algorithm_kwargs.share_feature=true ++fed_gnn.exp_name="fed_all" ++fed_gnn.weight_decay=0.001
done
