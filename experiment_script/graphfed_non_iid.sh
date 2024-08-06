for configname in cs.yaml yelp.yaml reddit.yaml ogbn-products.yaml; do
  for alpha in 0.5; do
    python3 ./simulator.py --config-name graphfed/${configname} ++graphfed.exp_name="graphfed" ++graphfed.algorithm_kwargs.node_sample_percent=0.7 ++graphfed.algorithm_kwargs.node_random_selection=true ++graphfed.round=100 ++graphfed.epoch=5 ++graphfed.worker_number=100 ++graphfed.model_name=TwoGATCN ++graphfed.dataloader_kwargs.batch_number=10 ++graphfed.learning_rate=0.001 ++graphfed.fake_weight_decay=0.001 ++graphfed.algorithm_kwargs.share_feature=false ++graphfed.algorithm_kwargs.edge_drop_rate=0.5 ++graphfed.dataset_sampling="dirichlet_split" ++graphfed.dataset_sampling_kwargs.concentration=$alpha
  done
done
