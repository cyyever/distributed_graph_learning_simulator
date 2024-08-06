for configname in cs.yaml PubMed.yaml yelp.yaml reddit.yaml amazonproducts.yaml; do
  env no_pipe=1 python3 ./simulator.py --config-name fed_cluster_gcn/${configname} ++fed_cluster_gcn.worker_number=100 ++fed_cluster_gcn.round=100 ++fed_cluster_gcn.model_name=TwoGATCN ++fed_cluster_gcn.epoch=1 ++fed_cluster_gcn.dataloader_kwargs.batch_number=10 ++fed_cluster_gcn.learning_rate=0.001 ++fed_cluster_gcn.algorithm_kwargs.edge_drop_rate=0.5 ++fed_cluster_gcn.algorithm_kwargs.share_feature=true ++fed_cluster_gcn.exp_name="fed_cluster_gcn" ++fed_cluster_gcn.weight_decay=0.001 ++fed_cluster_gcn.algorithm_kwargs.cluster_number=5
done
for configname in cs.yaml PubMed.yaml yelp.yaml reddit.yaml amazonproducts.yaml; do
  env no_pipe=1 python3 ./simulator.py --config-name fed_cluster_gcn/${configname} ++fed_cluster_gcn.worker_number=100 ++fed_cluster_gcn.round=100 ++fed_cluster_gcn.model_name=TwoGATCN ++fed_cluster_gcn.epoch=1 ++fed_cluster_gcn.dataloader_kwargs.batch_number=10 ++fed_cluster_gcn.learning_rate=0.001 ++fed_cluster_gcn.algorithm_kwargs.edge_drop_rate=0.5 ++fed_cluster_gcn.algorithm_kwargs.share_feature=true ++fed_cluster_gcn.exp_name="fed_cluster_gcn" ++fed_cluster_gcn.weight_decay=0.001 ++fed_cluster_gcn.algorithm_kwargs.cluster_number=10
done
