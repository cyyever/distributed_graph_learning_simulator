# fed_ags
for configname in cs.yaml PubMed.yaml reddit.yaml yelp.yaml; do
  python3 ./simulator.py --config-name fed_ags/${configname} ++fed_ags.worker_number=50 ++fed_ags.round=50 ++fed_ags.model_name=TwoGATCN ++fed_ags.epoch=1 ++fed_ags.dataloader_kwargs.batch_number=10 ++fed_ags.learning_rate=0.001 ++fed_ags.algorithm_kwargs.edge_drop_rate=0.99 ++fed_ags.fake_weight_decay=0.001 ++fed_ags.exp_name="fed_ags" ++fed_ags.algorithm_kwargs.min_sharing_interval=2
done
