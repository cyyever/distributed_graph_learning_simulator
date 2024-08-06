for configname in cs.yaml PubMed.yaml yelp.yaml reddit.yaml; do
    python3 ./simulator.py --config-name graphfed/${configname} ++graphfed.worker_number=100 ++graphfed.round=100 ++graphfed.model_name=TwoGATCN ++graphfed.epoch=1 ++graphfed.dataloader_kwargs.batch_number=10 ++graphfed.learning_rate=0.001 ++graphfed.weight_decay=0.001 ++graphfed.exp_name="graphfed" ++graphfed.algorithm_kwargs.min_sharing_interval=2 ++graphfed.algorithm_kwargs.edge_drop_rate=0.5 ++graphfed.algorithm_kwargs.share_feature=true
done
