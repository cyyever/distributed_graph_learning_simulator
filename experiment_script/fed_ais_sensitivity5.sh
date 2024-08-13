for configname in reddit.yaml; do
  for neib in 20 30 50; do
    # fed_ais
    for percent in 0.7; do
      python3 ./simulator.py --config-name fed_ais/${configname} ++fed_ais.worker_number=100 ++fed_ais.round=100 ++fed_ais.model_name=TwoGATCN ++fed_ais.epoch=1 ++fed_ais.dataloader_kwargs.batch_number=10 ++fed_ais.learning_rate=0.001 ++fed_ais.weight_decay=0.001 ++fed_ais.exp_name="fed_ais" ++fed_ais.algorithm_kwargs.sample_percent=$percent ++fed_ais.algorithm_kwargs.min_sharing_interval=2 ++fed_ais.algorithm_kwargs.edge_drop_rate=0.5 ++fed_ais.dataloader_kwargs.num_neighbor=${neib}
      python3 ./simulator.py --config-name fed_ais/${configname} ++fed_ais.worker_number=100 ++fed_ais.round=100 ++fed_ais.model_name=TwoGATCN ++fed_ais.epoch=1 ++fed_ais.dataloader_kwargs.batch_number=10 ++fed_ais.learning_rate=0.001 ++fed_ais.weight_decay=0.001 ++fed_ais.exp_name="fed_ais_random" ++fed_ais.algorithm_kwargs.sample_percent=$percent ++fed_ais.algorithm_kwargs.min_sharing_interval=2 ++fed_ais.algorithm_kwargs.edge_drop_rate=0.5 ++fed_ais.algorithm_kwargs.random_selection=true ++fed_ais.dataloader_kwargs.num_neighbor=${neib}
    done
    #fed_all
    python3 ./simulator.py --config-name fed_gnn/${configname} ++fed_gnn.worker_number=100 ++fed_gnn.round=100 ++fed_gnn.model_name=TwoGATCN ++fed_gnn.epoch=1 ++fed_gnn.dataloader_kwargs.batch_number=10 ++fed_gnn.learning_rate=0.001 ++fed_gnn.algorithm_kwargs.edge_drop_rate=0.5 ++fed_gnn.algorithm_kwargs.share_feature=true ++fed_gnn.exp_name="fed_all" ++fed_gnn.weight_decay=0.001 ++fed_gnn.dataloader_kwargs.num_neighbor=${neib}

  done
done
