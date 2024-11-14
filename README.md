# distributed_graph_learning_simulator

This is a simulator for distributed Graph Federated Learning on a single host. It implements our works and some graph learning methods as benchmarks.

## Installation

This is a Python project. The third party dependencies are listed in **requirements.txt**. If you use pip, it should be easy to install:

```
python3 -m pip install . --user
```

## Our Works

### Historical Embedding-Guided Efficient Large-Scale Federated Graph Learning

To run the experiments of [Historical Embedding-Guided Efficient Large-Scale Federated Graph Learning](https://dl.acm.org/doi/10.1145/3654947), use this command

```
bash fed_aas.sh
```

#### Reference

If you find this work useful, feel free to cite it:

```
@article{li2024historical,
  title={Historical Embedding-Guided Efficient Large-Scale Federated Graph Learning},
  author={Li, Anran and Chen, Yuanyuan and Zhang, Jian and Cheng, Mingfei and Huang, Yihao and Wu, Yueming and Luu, Anh Tuan and Yu, Han},
  journal={Proceedings of the ACM on Management of Data},
  volume={2},
  number={3},
  pages={1--24},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```
