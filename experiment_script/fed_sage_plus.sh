for configname in amazonproducts.yaml reddit.yaml cs.yaml PubMed.yaml yelp.yaml; do
  python3 ./simulator.py --config-name fed_sage_plus/${configname} ++fed_sage_plus.worker_number=100 ++fed_sage_plus.round=100 ++fed_sage_plus.model_name=TwoGATCN ++fed_sage_plus.epoch=1 ++fed_sage_plus.dataloader_kwargs.batch_number=10 ++fed_sage_plus.learning_rate=0.001 ++fed_sage_plus.algorithm_kwargs.edge_drop_rate=0.5 ++fed_sage_plus.exp_name="fed_sage_plus" ++fed_sage_plus.weight_decay=0.001 ++fed_sage_plus.log_level=INFO
  for alpha in 0.5; do
    python3 ./simulator.py --config-name fed_sage_plus/${configname} ++fed_sage_plus.worker_number=100 ++fed_sage_plus.round=100 ++fed_sage_plus.model_name=TwoGATCN ++fed_sage_plus.epoch=1 ++fed_sage_plus.dataloader_kwargs.batch_number=10 ++fed_sage_plus.learning_rate=0.001 ++fed_sage_plus.algorithm_kwargs.edge_drop_rate=0.5 ++fed_sage_plus.exp_name="fed_sage_plus" ++fed_sage_plus.weight_decay=0.001 ++fed_sage_plus.log_level=INFO ++fed_sage_plus.dataset_sampling="dirichlet_split" ++fed_sage_plus.dataset_sampling_kwargs.concentration=$alpha
  done
done
