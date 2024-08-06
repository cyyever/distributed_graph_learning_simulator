# fedaas
for configname in cora_full.yaml; do
  python3 ./simulator.py --config-name fed_aas/${configname} ++fed_aas.worker_number=50 ++fed_aas.round=50 ++fed_aas.model_name=TwoGATCN ++fed_aas.epoch=1 ++fed_aas.algorithm_kwargs.batch_number=10 ++fed_aas.learning_rate=0.001 ++fed_aas.algorithm_kwargs.edge_drop_rate=0.99 ++fed_aas.weight_decay=0.001 ++fed_aas.exp_name="fed_aas" ++fed_aas.algorithm_kwargs.min_sharing_interval=2
done

for alpha in 0.5; do
  for configname in cora_full.yaml; do
    python3 ./simulator.py --config-name fed_aas/${configname} ++fed_aas.worker_number=50 ++fed_aas.round=50 ++fed_aas.model_name=TwoGATCN ++fed_aas.epoch=1 ++fed_aas.algorithm_kwargs.batch_number=10 ++fed_aas.learning_rate=0.001 ++fed_aas.algorithm_kwargs.edge_drop_rate=0.99 ++fed_aas.weight_decay=0.001 ++fed_aas.exp_name="fed_aas" ++fed_aas.algorithm_kwargs.min_sharing_interval=2 ++fed_aas.dataset_sampling="dirichlet_split" ++fed_aas.dataset_sampling_kwargs.concentration=$alpha
  done
done

for configname in cora_full.yaml; do
  python3 ./simulator.py --config-name fed_gnn/${configname} ++fed_gnn.worker_number=50 ++fed_gnn.round=50 ++fed_gnn.model_name=TwoGATCN ++fed_gnn.epoch=1 ++fed_gnn.algorithm_kwargs.batch_number=10 ++fed_gnn.learning_rate=0.001 ++fed_gnn.algorithm_kwargs.edge_drop_rate=0.99 ++fed_gnn.algorithm_kwargs.share_feature=true ++fed_gnn.exp_name="fed_all" ++fed_gnn.weight_decay=0.001
done
for alpha in 0.5; do
  for configname in cora_full.yaml; do
    python3 ./simulator.py --config-name fed_gnn/${configname} ++fed_gnn.worker_number=50 ++fed_gnn.round=50 ++fed_gnn.model_name=TwoGATCN ++fed_gnn.epoch=1 ++fed_gnn.algorithm_kwargs.batch_number=10 ++fed_gnn.learning_rate=0.001 ++fed_gnn.algorithm_kwargs.edge_drop_rate=0.99 ++fed_gnn.algorithm_kwargs.share_feature=true ++fed_gnn.exp_name="fed_all" ++fed_gnn.weight_decay=0.001 ++fed_gnn.dataset_sampling="dirichlet_split" ++fed_gnn.dataset_sampling_kwargs.concentration=$alpha
  done
done

for configname in cora_full.yaml; do
  python3 ./simulator.py --config-name fed_gnn/${configname} ++fed_gnn.worker_number=50 ++fed_gnn.round=50 ++fed_gnn.model_name=TwoGATCN ++fed_gnn.epoch=1 ++fed_gnn.algorithm_kwargs.batch_number=10 ++fed_gnn.learning_rate=0.001 ++fed_gnn.algorithm_kwargs.edge_drop_rate=0.99 ++fed_gnn.algorithm_kwargs.share_feature=false ++fed_gnn.exp_name="fed_local" ++fed_gnn.weight_decay=0.001
done

for alpha in 0.5; do
  for configname in cora_full.yaml; do
    python3 ./simulator.py --config-name fed_gnn/${configname} ++fed_gnn.worker_number=50 ++fed_gnn.round=50 ++fed_gnn.model_name=TwoGATCN ++fed_gnn.epoch=1 ++fed_gnn.algorithm_kwargs.batch_number=10 ++fed_gnn.learning_rate=0.001 ++fed_gnn.algorithm_kwargs.edge_drop_rate=0.99 ++fed_gnn.algorithm_kwargs.share_feature=false ++fed_gnn.exp_name="fed_local" ++fed_gnn.weight_decay=0.001 ++fed_gnn.dataset_sampling="dirichlet_split" ++fed_gnn.dataset_sampling_kwargs.concentration=$alpha
  done
done

for configname in cora_full.yaml; do
  python3 ./simulator.py --config-name fed_aas/${configname} ++fed_aas.worker_number=50 ++fed_aas.round=50 ++fed_aas.model_name=TwoGATCN ++fed_aas.epoch=1 ++fed_aas.algorithm_kwargs.batch_number=10 ++fed_aas.learning_rate=0.001 ++fed_aas.algorithm_kwargs.edge_drop_rate=0.99 ++fed_aas.algorithm_kwargs.fixed_sharing_interval=2 ++fed_aas.weight_decay=0.001 ++fed_aas.exp_name="fed_aas_periodic"
done
for alpha in 0.5; do
  for configname in cora_full.yaml; do
    python3 ./simulator.py --config-name fed_aas/${configname} ++fed_aas.worker_number=50 ++fed_aas.round=50 ++fed_aas.model_name=TwoGATCN ++fed_aas.epoch=1 ++fed_aas.algorithm_kwargs.batch_number=10 ++fed_aas.learning_rate=0.001 ++fed_aas.algorithm_kwargs.edge_drop_rate=0.99 ++fed_aas.algorithm_kwargs.fixed_sharing_interval=2 ++fed_aas.weight_decay=0.001 ++fed_aas.exp_name="fed_aas_periodic" ++fed_aas.dataset_sampling="dirichlet_split" ++fed_aas.dataset_sampling_kwargs.concentration=$alpha
  done
done

for configname in cora_full.yaml; do
  python3 ./simulator.py --config-name fed_aas/${configname} ++fed_aas.worker_number=50 ++fed_aas.round=50 ++fed_aas.model_name=TwoGATCN ++fed_aas.epoch=1 ++fed_aas.algorithm_kwargs.batch_number=10 ++fed_aas.learning_rate=0.001 ++fed_aas.algorithm_kwargs.edge_drop_rate=0.99 ++fed_aas.algorithm_kwargs.fixed_sharing_interval=1 ++fed_aas.weight_decay=0.001 ++fed_aas.exp_name="fed_aas_periodic1"
done
for alpha in 0.5; do
  for configname in cora_full.yaml; do
    python3 ./simulator.py --config-name fed_aas/${configname} ++fed_aas.worker_number=50 ++fed_aas.round=50 ++fed_aas.model_name=TwoGATCN ++fed_aas.epoch=1 ++fed_aas.algorithm_kwargs.batch_number=10 ++fed_aas.learning_rate=0.001 ++fed_aas.algorithm_kwargs.edge_drop_rate=0.99 ++fed_aas.algorithm_kwargs.fixed_sharing_interval=1 ++fed_aas.weight_decay=0.001 ++fed_aas.exp_name="fed_aas_periodic1" ++fed_aas.dataset_sampling="dirichlet_split" ++fed_aas.dataset_sampling_kwargs.concentration=$alpha
  done
done
