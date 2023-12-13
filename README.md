# AHNS
PyTorch Implementation for Adaptive Hardness Negative Sampling for Collaborative Filtering, AAAI2024

--- Based on "MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems", https://github.com/huangtinglin/MixGCF


#### Environment Requirement

The code has been tested running under Python 3.7.6. The required packages are as follows:

- pytorch == 1.7.0
- numpy == 1.20.2
- scipy == 1.6.3
- sklearn == 0.24.1
- prettytable == 2.1.0



#### Training

The training commands are as following:

```
python main.py --dataset ml --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 0 --ns ahns --alpha 0.1 --beta 0.4 --n_negs 16
```

```
python main.py --dataset phone --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 0 --ns ahns --alpha 1.0 --beta 0.1 --n_negs 32
```

```
python main.py --dataset sport --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 0 --ns ahns --alpha 0.5 --beta 0.1 --n_negs 32
```

```
python main.py --dataset tool --gnn lightgcn --dim 64 --lr 0.001 --batch_size 2048 --gpu_id 0 --context_hops 0 --ns ahns --alpha 1.0 --beta 0.1 --n_negs 32
```

The ipynb files are also provided.



#### Datasets

We use four processed datasets: MovieLens-1M, Amazon-Phones, Amazon-Sports and Amazon-Tools. The processing code is also provided.

|        | \#user | \#item | \#inter. | avg. inter. |
| :----: | :----: | :----: | :------: | :---------: |
| ML-1M  |  6.0k  |  3.7k  | 1000.2k  |    165.6    |
| Phones | 27.9k  | 10.4k  |  194.4k  |     7.0     |
| Sports | 35.6k  | 18.4k  |  296.3k  |     8.3     |
| Tools  | 16.6k  | 10.2k  |  134.5k  |     8.1     |
