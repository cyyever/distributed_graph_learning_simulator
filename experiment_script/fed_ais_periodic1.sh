for configname in cs.yaml PubMed.yaml yelp.yaml reddit.yaml; do
  for percent in 0.7; do
    python3 ./simulator.py --config-name fed_ais/${configname} ++fed_ais.worker_number=100 ++fed_ais.round=100 ++fed_ais.model_name=TwoGATCN ++fed_ais.epoch=1 ++fed_ais.dataloader_kwargs.batch_number=10 ++fed_ais.learning_rate=0.001 ++fed_ais.weight_decay=0.001 ++fed_ais.exp_name="fed_ais_periodic1" ++fed_ais.algorithm_kwargs.sample_percent=$percent ++fed_ais.algorithm_kwargs.fixed_sharing_interval=1 ++fed_ais.algorithm_kwargs.edge_drop_rate=0.5
  done
done
