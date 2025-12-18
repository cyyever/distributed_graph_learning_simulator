for configname in cs.yaml PubMed.yaml yelp.yaml reddit.yaml ogbn-products.yaml; do
  python3 ./simulator.py --config-name fed_ags/${configname} ++fed_ags.worker_number=100 ++fed_ags.exp_name="fed_ags" ++fed_ags.algorithm_kwargs.sample_percent=0.7 ++fed_ags.algorithm_kwargs.edge_drop_rate=0.5 ++fed_ags.algorithm_kwargs.max_epoch=50 ++fed_ags.algorithm_kwargs.warmup_rounds=1 ++fed_ags.algorithm_kwargs.total_epoch=500
done
